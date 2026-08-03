"""scripts/serving/api_server.py - FastAPI 추론 서버.

엔드포인트:
  GET  /health            서버/모델 상태 확인 — {"ok": true, "model": "...", "classes": {...}}
  POST /detect            [교육생 앱 공식 규격] 이미지 1장 추론 — 「YOLO 부품인식 API 인터페이스 요청서(2탄)」(2026-07-29) 준수
  POST /predict           [내부/개발용] 이미지 1장 추론 — bbox 포함 상세 포맷 (README·대시보드용)

── /detect (요청서 2탄 규격 + bbox 추가 제공) ──
요청: multipart/form-data — 필드 "image"(JPEG) + "camera_id"(기기 식별자, 예: TAB-01)
앱(태블릿/ML2)이 초당 2~5회 연속 호출. "인식됨" 이벤트 판단은 앱이 응답을 보고 수행.
응답(감지 없어도 항상 200 + 동일 구조):
  {
    "timestamp": "2026-07-29T14:30:15.123456",   # ISO 8601
    "camera_id": "TAB-01",                        # 요청 값 그대로 반환
    "detection_count": 1,                         # 없으면 0
    "detections": [
      {"detection_class": "bolt", "detection_code": 0, "confidence": 0.923,
       "bbox": {"x": 512.0, "y": 244.5, "w": 96.0, "h": 128.0}}
    ]
  }
규칙: confidence < 0.7 은 detections 에 미포함 / confidence 는 number 3자리(문자열 금지).
bbox: 2탄 규격에는 없는 **추가 제공 필드** — 앱이 안 쓰면 무시해도 됨(파서 하위호환).
      x,y = 박스 좌상단 픽셀 / w,h = 너비·높이 픽셀 (앱이 보낸 원본 이미지 좌표 기준).
      XR 화면에 부품 위치 표시가 필요해지는 시점에 바로 받아쓸 수 있게 항상 실어 보낸다.
오류: 400 {"error": "invalid_image"} / 500 {"error": "inference_failed"}
문서: 서버 기동 후 http://<host>:8000/docs (Swagger UI — 필드 정의·예시·오류 규격 자동 문서)

── /predict (내부용, 기존 유지) ──
요청 필드 "file". 응답에 bbox(좌상단 x,y + w,h 픽셀)·image 크기 포함 — 육안 검증·대시보드에서 사용.

실행:
  python scripts/serving/api_server.py
  # 테스트(2탄 자가 테스트): curl -X POST http://localhost:8000/detect -F "image=@test.jpg" -F "camera_id=TEST-01"
  #                          curl http://localhost:8000/health
"""
import io
import threading
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from ultralytics import YOLO

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ 공용(config 등)
import config

# 모델은 무겁다 -> 요청마다 로드하지 않고 프로세스 시작 시 1회만 로드해 전역 공유한다.
STATE = {}

# 요청서 2탄 규격: confidence 0.7 미만은 detections 에 넣지 않는다 (/detect 전용 임계값).
DETECT_CONF = 0.7

# ultralytics 모델 객체는 내부 predictor 상태를 공유해 스레드 동시 호출 시 결과가 섞일 수 있다.
# 여러 기기가 동시에 POST 해도 안전하도록 추론 구간만 직렬화한다
# (2~10 fps 부하에서 추론 1회 수십 ms 라 병목 아님).
INFER_LOCK = threading.Lock()


# ── /detect 응답 스키마 (Pydantic) ──
# dict 를 그대로 반환하면 Swagger(/docs)에 응답 구조가 안 뜬다.
# 모델로 선언해야 필드·타입·설명·예시가 자동 문서화되어, 앱 쪽이 문서만 보고 파싱을 짤 수 있다.

class BBox(BaseModel):
    """감지 박스 위치 — 앱이 보낸 원본 이미지 좌표(픽셀) 기준."""
    x: float = Field(description="박스 좌상단 x (픽셀)", examples=[512.0])
    y: float = Field(description="박스 좌상단 y (픽셀)", examples=[244.5])
    w: float = Field(description="박스 너비 (픽셀)", examples=[96.0])
    h: float = Field(description="박스 높이 (픽셀)", examples=[128.0])


class Detection(BaseModel):
    detection_class: str = Field(description="부품명 (부품표 확정 시 그 표 기준)", examples=["bolt"])
    detection_code: int = Field(description="부품 번호 (학습 클래스 순서 0..k-1)", examples=[0])
    confidence: float = Field(description="신뢰도 0~1, 소수점 3자리. 0.7 미만은 목록에 미포함", examples=[0.923])
    bbox: BBox = Field(description="[추가 제공] 박스 위치. 요청서 2탄 규격 외 필드 — 위치 표시가 필요 없으면 무시해도 됨")


class DetectResponse(BaseModel):
    """감지가 없어도 항상 이 구조로 200 응답 (detection_count=0, detections=[])."""
    timestamp: str = Field(description="추론 시각, ISO 8601", examples=["2026-07-29T14:30:15.123456"])
    camera_id: str = Field(description="요청에서 받은 값 그대로 반환 (앱이 자기 결과인지 확인용)", examples=["TAB-01"])
    detection_count: int = Field(description="감지된 객체 수. 없으면 0", examples=[1])
    detections: list[Detection] = Field(description="감지 목록. 없으면 빈 배열")


class ErrorBody(BaseModel):
    error: str = Field(examples=["invalid_image"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 수명주기 훅: 시작 시 모델 1회 로드, 종료 시 정리."""
    model_path = str(config.SERVE_MODEL)
    if not config.SERVE_MODEL.exists():
        # 모델이 없으면 즉시 알 수 있게 명확히 실패시킨다(요청 때 미스터리 에러 방지).
        raise RuntimeError(
            f"서빙 모델이 없습니다: {model_path}\n"
            f"먼저 scripts/training/train_pipeline.py 로 new_model.pt 를 만드세요."
        )
    print(f"[서버] 모델 로드: {model_path}")
    STATE["model"] = YOLO(model_path)        # .pt / .onnx 모두 로드 가능
    # 워밍업 1회: 첫 실제 요청이 CUDA 초기화/그래프 빌드 비용(수 초)을 물지 않게 한다.
    STATE["model"].predict(source=Image.new("RGB", (config.IMG_SIZE, config.IMG_SIZE)),
                           verbose=False)
    yield
    STATE.clear()


app = FastAPI(title="XR AutoLearning Parts Detector", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    """서버가 살아 있고 모델이 로드됐는지, 어떤 클래스를 아는지 반환.

    "ok"/"model" 키 = 요청서 2탄 규격. "classes" 는 부품표(클래스↔번호) 확인용 추가 정보.
    """
    model = STATE.get("model")
    return {
        "ok": model is not None,
        "model": config.SERVE_MODEL.name,
        "classes": model.names if model else {},
    }


# 요청서 2탄 오류 규격: /detect 는 필수 필드 누락도 400 {"error": "invalid_image"} 로 응답.
# (FastAPI 기본은 422 라서 /detect 경로만 400 으로 변환한다.)
@app.exception_handler(RequestValidationError)
async def validation_handler(request, exc):
    if request.url.path == "/detect":
        return JSONResponse(status_code=400, content={"error": "invalid_image"})
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


# async 가 아니라 일반 def: 동기 함수인 model.predict() 가 이벤트루프를 막지 않도록
# FastAPI 가 스레드풀에서 실행한다 (추론 중에도 /health 등 다른 요청 수신 가능).
@app.post(
    "/detect",
    response_model=DetectResponse,
    responses={  # Swagger 오류 규격 문서화 (요청서 2탄 §3)
        400: {"model": ErrorBody, "description": "image 필드 없음 / JPEG 디코드 실패",
              "content": {"application/json": {"example": {"error": "invalid_image"}}}},
        500: {"model": ErrorBody, "description": "서버 내부 오류 (모델 등)",
              "content": {"application/json": {"example": {"error": "inference_failed"}}}},
    },
)
def detect(image: UploadFile = File(...), camera_id: str = Form("UNKNOWN")):
    """[교육생 앱 공식 규격] 이미지 1장 추론 — 요청서 2탄(2026-07-29) 준수 + `bbox` 추가 제공.

    앱이 초당 2~5회 연속 호출한다. 감지가 없어도 200 + 빈 배열로 응답한다
    (앱이 "서버 정상 + 인식 안 됨"과 "서버 오류"를 구분하는 기준).
    `bbox`는 규격 외 추가 필드 — 위치 표시가 필요 없으면 무시해도 된다.
    """
    raw = image.file.read()
    if not raw:
        return JSONResponse(status_code=400, content={"error": "invalid_image"})
    try:
        # convert("RGB")로 채널 통일(흑백/RGBA/EXIF 회전 대비)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return JSONResponse(status_code=400, content={"error": "invalid_image"})

    try:
        with INFER_LOCK:  # 여러 기기 동시 요청 대비 추론 직렬화
            r = STATE["model"].predict(source=img, imgsz=config.IMG_SIZE,
                                       conf=DETECT_CONF, verbose=False)[0]

        names = r.names
        detections = []
        if r.boxes is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()  # 원본 이미지 픽셀 좌표 [x1,y1,x2,y2]
            clss = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), cls_id, conf in zip(xyxy, clss, confs):
                detections.append({
                    # 클래스명·번호 = 학습 data.yaml 순서(0..k-1). 부품표 확정 시 그 표와 일치.
                    "detection_class": names.get(int(cls_id), str(int(cls_id))),
                    "detection_code": int(cls_id),
                    "confidence": round(float(conf), 3),  # number, 소수점 3자리
                    # [추가 제공] 좌상단 x,y + 너비/높이 w,h — 앱은 필요할 때만 사용
                    "bbox": {
                        "x": round(float(x1), 2),
                        "y": round(float(y1), 2),
                        "w": round(float(x2 - x1), 2),
                        "h": round(float(y2 - y1), 2),
                    },
                })
    except Exception:
        return JSONResponse(status_code=500, content={"error": "inference_failed"})

    return {
        "timestamp": datetime.now().isoformat(),  # 추론 시각 ISO 8601
        "camera_id": camera_id,                   # 요청 값 그대로 반환(앱이 자기 결과 확인용)
        "detection_count": len(detections),
        "detections": detections,
    }


# (주의) async def 로 선언하면 동기 함수인 model.predict() 가 이벤트루프를 통째로
# 막아 동시 요청이 전부 직렬화된다. 일반 def 로 두면 FastAPI 가 스레드풀에서
# 실행해 추론 중에도 다른 요청(/health 등)을 받을 수 있다.
@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    conf: float = Query(config.API_CONF, ge=0.0, le=1.0, description="confidence 임계값"),
    iou: float = Query(config.API_IOU, ge=0.0, le=1.0, description="NMS IoU 임계값"),
):
    """[내부/개발용] 업로드 이미지를 추론해 bbox 포함 상세 JSON 으로 반환한다.

    교육생 앱 공식 인터페이스는 /detect (요청서 2탄 규격) — 이쪽은 육안 검증·대시보드용으로 유지.
    """
    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    try:
        # PIL 로 디코드. convert("RGB") 로 채널을 통일(흑백/RGBA/EXIF 회전 대비).
        # ultralytics 에 PIL.Image 를 직접 넘기면 RGB 로 올바르게 처리하므로
        # numpy(BGR/RGB) 변환에서 생기는 색상 채널 혼동을 피한다.
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="이미지를 열 수 없습니다(지원 형식 확인).")

    model = STATE["model"]
    names = model.names

    # 단일 이미지 추론. verbose=False 로 콘솔 로그를 끈다. (/detect 와 같은 락으로 직렬화)
    with INFER_LOCK:
        results = model.predict(source=image, conf=conf, iou=iou, verbose=False)
    r = results[0]

    detections = []
    if r.boxes is not None:
        # xyxy(원본 이미지 픽셀 좌표) -> (x, y, w, h) 좌상단+너비/높이 로 변환.
        xyxy = r.boxes.xyxy.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, cf in zip(xyxy, clss, confs):
            detections.append({
                "class_id": int(c),
                "class_name": names.get(int(c), str(int(c))),
                "confidence": round(float(cf), 4),
                "bbox": {
                    "x": round(float(x1), 2),
                    "y": round(float(y1), 2),
                    "w": round(float(x2 - x1), 2),
                    "h": round(float(y2 - y1), 2),
                },
            })

    return {
        "filename": file.filename,
        "image": {"width": image.width, "height": image.height},
        "count": len(detections),
        "detections": detections,
    }


if __name__ == "__main__":
    import uvicorn
    # app 객체를 직접 전달(파일명이 숫자로 시작해 "module:app" 문자열 import 가 까다롭기 때문).
    # 이렇게 하면 reload 없이 모델을 단 한 번만 로드해 메모리를 아낀다.
    uvicorn.run(app, host=config.HOST, port=config.PORT)
