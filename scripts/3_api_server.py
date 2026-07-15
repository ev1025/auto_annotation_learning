"""3_api_server.py - FastAPI 추론 서버 (타 부서 제공용 REST 포맷).

엔드포인트:
  GET  /health            서버/모델 상태 확인
  POST /predict           이미지 1장 추론(multipart/form-data, 필드명 = "file")

응답 JSON 예시:
  {
    "filename": "part.jpg",
    "image": {"width": 1280, "height": 720},
    "count": 2,
    "detections": [
      {"class_id": 0, "class_name": "bolt", "confidence": 0.91,
       "bbox": {"x": 120.5, "y": 80.0, "w": 64.0, "h": 64.0}},
      ...
    ]
  }

bbox 좌표 약속(중요):
  x, y = 박스 좌상단(top-left) 픽셀 좌표
  w, h = 박스 너비 / 높이(픽셀)
  -> XR/프론트엔드가 화면에 바로 사각형을 그릴 수 있는 가장 보편적인 형태로 통일했다.

실행:
  python 3_api_server.py
  # 또는: uvicorn 3_api_server:app --host 0.0.0.0 --port 8000   <- 파일명이 숫자로 시작해 import 명이 어색하므로 직접 실행 권장
  # 테스트: curl -X POST -F "file=@sample.jpg" http://localhost:8000/predict
"""
import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image
from ultralytics import YOLO

import config

# 모델은 무겁다 -> 요청마다 로드하지 않고 프로세스 시작 시 1회만 로드해 전역 공유한다.
STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 수명주기 훅: 시작 시 모델 1회 로드, 종료 시 정리."""
    model_path = str(config.SERVE_MODEL)
    if not config.SERVE_MODEL.exists():
        # 모델이 없으면 즉시 알 수 있게 명확히 실패시킨다(요청 때 미스터리 에러 방지).
        raise RuntimeError(
            f"서빙 모델이 없습니다: {model_path}\n"
            f"먼저 2_train_pipeline.py 로 new_model.pt 를 만드세요."
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
    """서버가 살아 있고 모델이 로드됐는지, 어떤 클래스를 아는지 반환."""
    model = STATE.get("model")
    return {
        "status": "ok",
        "model": str(config.SERVE_MODEL),
        "classes": model.names if model else {},
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
    """업로드 이미지를 추론해 탐지 결과를 JSON 으로 반환한다."""
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

    # 단일 이미지 추론. verbose=False 로 콘솔 로그를 끈다.
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
