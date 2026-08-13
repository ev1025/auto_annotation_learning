# -*- coding: utf-8 -*-
"""server.py - YOLO 부품인식 연동 테스트 서버 (자립형, USB 전달용).

「YOLO 사물인식 서버 인터페이스 요청서 (2탄)」(2026-07-29) 규격 그대로 동작한다.
오토러닝으로 학습한 실제 부품 모델(2클래스)을 서빙한다:

  ┌─────────────────┬────────────────┬──────────────────────────────┐
  │ detection_class │ detection_code │ 테스트 방법                   │
  ├─────────────────┼────────────────┼──────────────────────────────┤
  │ gearbox         │ 0              │ 카메라에 기어박스를 비춘다     │
  │ a_test          │ 1              │ 카메라에 a_test 부품을 비춘다  │
  └─────────────────┴────────────────┴──────────────────────────────┘

실행:
  pip install -r requirements.txt   (최초 1회)
  python server.py                  (또는 run.bat 더블클릭)
  → http://0.0.0.0:9412  (문서: http://localhost:9412/docs)

자가 테스트:
  curl http://localhost:9412/health
  curl -X POST http://localhost:9412/detect -F "image=@아무사진.jpg" -F "camera_id=TEST-01"
"""
import io
import os
import sys
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from ultralytics import YOLO

# Windows 콘솔 한글 출력 깨짐 방지
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, Exception):
    pass

# ── 설정 ─────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
# 서빙 모델 파일. 오토러닝 최종 부품 모델(2클래스: gearbox·a_test). model.pt·model.onnx 둘 다 동봉.
# 기본 = model.pt (이식성 좋고 onnxruntime 불필요, CPU 로 충분).
# ── .onnx 로 전환하려면 ── 아래 "model.pt" 를 "model.onnx" 로 바꾸거나(한 줄),
#    실행 시 환경변수로: SERVE_MODEL_FILE=model.onnx  (onnxruntime 는 requirements 에 포함됨)
MODEL_PATH = BASE_DIR / os.getenv("SERVE_MODEL_FILE", "model.pt")
# 포트 9412 = Bell 412 에서 따옴. 8000 은 흔해서 다른 프로젝트 서버와 충돌한다(실제 충돌 이력).
HOST = os.getenv("YOLO_SERVER_HOST", "0.0.0.0")
PORT = int(os.getenv("YOLO_SERVER_PORT", "9412"))
IMG_SIZE = 640
CONFIDENCE_THRESHOLD = 0.7             # 요청서 규격: 0.7 미만은 detections 미포함

# 실제 부품 모델은 자기 클래스(0=gearbox, 1=a_test)만 출력하므로 별도 필터가 필요 없다.
# detection_class/code = 모델 클래스명/인덱스를 그대로 내보낸다(아래 detect 참고).
# (예전 COCO 임시모델에서 person/cell phone 만 통과시키던 TARGET_CLASSES 필터는 제거)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

STATE = {}

# ultralytics 모델은 동시 호출 시 내부 상태가 섞일 수 있어 추론 구간만 직렬화
# (초당 3장 순차 전송 조건에서 병목 아님)
INFER_LOCK = threading.Lock()


# ── 응답 스키마 (Swagger /docs 자동 문서화용) ──

class BBox(BaseModel):
    """[추가 제공] 감지 박스 위치 — 앱이 보낸 원본 이미지 좌표(픽셀) 기준. 무시해도 됨."""
    x: float = Field(description="박스 좌상단 x (픽셀)", examples=[512.0])
    y: float = Field(description="박스 좌상단 y (픽셀)", examples=[244.5])
    w: float = Field(description="박스 너비 (픽셀)", examples=[96.0])
    h: float = Field(description="박스 높이 (픽셀)", examples=[128.0])


class Detection(BaseModel):
    detection_class: str = Field(description="클래스명 (테스트: person / cell phone)", examples=["person"])
    detection_code: int = Field(description="클래스 번호 (person=0, cell phone=1)", examples=[0])
    confidence: float = Field(description="신뢰도 0~1, 소수점 3자리. 0.7 미만 미포함", examples=[0.923])
    bbox: BBox = Field(description="[추가 제공] 요청서 2탄 규격 외 필드 — 무시 가능")


class DetectResponse(BaseModel):
    """감지가 없어도 항상 이 구조로 200 응답 (detection_count=0, detections=[])."""
    timestamp: str = Field(description="추론 시각, ISO 8601", examples=["2026-07-30T14:30:15.123456"])
    camera_id: str = Field(description="요청에서 받은 값 그대로 반환", examples=["TAB-01"])
    detection_count: int = Field(description="감지된 객체 수. 없으면 0", examples=[1])
    detections: list[Detection] = Field(description="감지 목록. 없으면 빈 배열")


class ErrorBody(BaseModel):
    error: str = Field(examples=["invalid_image"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """기동 시 모델 1회 로드 + 워밍업(첫 요청도 빠르게)."""
    logging.info(f"모델 로드: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    model.predict(source=Image.new("RGB", (IMG_SIZE, IMG_SIZE)), verbose=False)  # 워밍업
    STATE["model"] = model
    logging.info(f"준비 완료 — 부품 클래스: {model.names}")
    yield
    STATE.clear()


app = FastAPI(title="YOLO 연동 테스트 서버 (요청서 2탄 규격)",
              version="test-1.0", lifespan=lifespan)


@app.get("/health")
def health():
    """생존 확인 — {"ok": true, "model": ..., "classes": 테스트 대상 클래스표}."""
    model = STATE.get("model")
    return {
        "ok": model is not None,
        "model": MODEL_PATH.name,
        "classes": {str(i): n for i, n in model.names.items()} if model else {},
    }


# 요청서 2탄 오류 규격: 필수 필드 누락도 400 invalid_image (FastAPI 기본 422를 변환)
@app.exception_handler(RequestValidationError)
async def validation_handler(request, exc):
    if request.url.path == "/detect":
        return JSONResponse(status_code=400, content={"error": "invalid_image"})
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


# 일반 def: 동기 추론이 이벤트루프를 막지 않게 FastAPI 가 스레드풀에서 실행
@app.post(
    "/detect",
    response_model=DetectResponse,
    responses={
        400: {"model": ErrorBody, "description": "image 필드 없음 / JPEG 디코드 실패",
              "content": {"application/json": {"example": {"error": "invalid_image"}}}},
        500: {"model": ErrorBody, "description": "서버 내부 오류",
              "content": {"application/json": {"example": {"error": "inference_failed"}}}},
    },
)
def detect(image: UploadFile = File(...), camera_id: str = Form("UNKNOWN")):
    """이미지 1장 추론 — 요청서 2탄 규격. 감지 없어도 200 + 빈 배열."""
    raw = image.file.read()
    if not raw:
        return JSONResponse(status_code=400, content={"error": "invalid_image"})
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return JSONResponse(status_code=400, content={"error": "invalid_image"})

    try:
        with INFER_LOCK:
            r = STATE["model"].predict(source=img, imgsz=IMG_SIZE,
                                       conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

        names = r.names
        detections = []
        if r.boxes is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), cls_id, conf in zip(xyxy, clss, confs):
                cls_name = names.get(int(cls_id), str(int(cls_id)))
                detections.append({
                    "detection_class": cls_name,
                    "detection_code": int(cls_id),
                    "confidence": round(float(conf), 3),
                    "bbox": {
                        "x": round(float(x1), 2),
                        "y": round(float(y1), 2),
                        "w": round(float(x2 - x1), 2),
                        "h": round(float(y2 - y1), 2),
                    },
                })
    except Exception as e:
        logging.error(f"추론 실패: {e}")
        return JSONResponse(status_code=500, content={"error": "inference_failed"})

    result = {
        "timestamp": datetime.now().isoformat(),
        "camera_id": camera_id,
        "detection_count": len(detections),
        "detections": detections,
    }
    if detections:
        logging.info(f"감지: {camera_id} → "
                     f"{[(d['detection_class'], d['confidence']) for d in detections]}")
    return result


@app.post("/reload")
def reload_model():
    """오토러닝이 새 모델(model.pt/onnx)을 배포(덮어쓰기)한 뒤 호출 → 모델 핫 리로드(서버 재시작 불필요).
    오토러닝의 '신규 모델 적용'이 이 엔드포인트를 자동 호출한다."""
    with INFER_LOCK:
        model = YOLO(str(MODEL_PATH))
        model.predict(source=Image.new("RGB", (IMG_SIZE, IMG_SIZE)), verbose=False)  # 워밍업
        STATE["model"] = model
    logging.info(f"모델 리로드: {MODEL_PATH} — 부품 클래스: {model.names}")
    return {"ok": True, "model": MODEL_PATH.name,
            "classes": {str(i): n for i, n in model.names.items()}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
