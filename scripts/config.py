"""config.py - 파이프라인 전역 설정 (경로 / 클래스 / 임계값).

실행 스크립트들(labeling/auto_labeling, training/train_pipeline, serving/api_server)가
공통으로 참조하는 값을 한 곳에 모은다.
- 경로를 각 스크립트에 하드코딩하지 않고 여기서만 바꾸면 전체 파이프라인이 따라오도록 모듈화.
- 숫자로 시작하는 실행 스크립트(1_, 2_, 3_)는 파이썬 식별자 규칙상 import 가 불가능하므로,
  '공유 상수'는 import 가능한 이 모듈(config.py)에 둔다.
"""
import os
from pathlib import Path

# 프로젝트 루트 절대경로(이 파일은 scripts/ 안에 있으므로 한 단계 위).
# 어느 작업 디렉토리에서 실행해도 경로가 깨지지 않게 한다.
BASE_DIR = Path(__file__).resolve().parent.parent

# --- 모델 가중치 ---
MODELS_DIR = BASE_DIR / "models"
NEW_MODEL_PT = MODELS_DIR / "model.pt"      # 서빙 모델(본 학습 결과 best.pt 를 복사). 고정 이름 = 갱신 시 덮어쓰기
NEW_MODEL_ONNX = MODELS_DIR / "model.onnx"  # ONNX 변환 산출물(Thor TensorRT / 비파이썬 런타임용)

# --- YOLO 추론 서버(배포 타깃) ---  오토러닝 '신규 모델 적용' 시 여기로 자동 배포(복사+리로드)
YOLO_SERVER_DIR = BASE_DIR / "backend" / "yolo_server"
# 우리 YOLO 추론서버 포트 = 9412 (Bell 412 에서 따옴).
# 8000 은 흔해서 다른 프로젝트 서버(화재/연기 API 등)와 충돌한다 — 실제로 충돌해 /reload 가 남의 서버로 갔었다.
YOLO_SERVER_PORT = int(os.environ.get("YOLO_SERVER_PORT", "9412"))
YOLO_SERVER_URL = os.environ.get("YOLO_SERVER_URL", f"http://127.0.0.1:{YOLO_SERVER_PORT}")   # /reload 호출용(서버 안 떠 있으면 파일만 배포)

# --- 데이터셋 경로 ---
DATA_DIR = BASE_DIR / "data"          # 모든 데이터(원본·데이터셋·등록영상)의 단일 루트
DATA_YAML = BASE_DIR / "data.yaml"

# --- 임계값 ---
AUTO_LABEL_CONF = 0.6   # 자동 라벨링: 이 값 이상만 '정답'으로 신뢰

# --- 모델 릴리스(버전 보관·배포 게이트·롤백) ---

# --- 학습 하이퍼파라미터 기본값 ---
# 부품 34클래스 재벤치(results/bench/20260805_173149, v8·11·26 × n·s·m)로 선정:
# yolo11s가 mAP50 1위(0.9931)이자 기존 yolo26s 대비 전 항목 우위(크기·GPU속도·학습시간)
# 경량·엣지 배포가 필요하면 yolov8n / yolo11n 으로 교체
PRETRAINED = "yolo11s.pt"  # base_model 이 없을 때 전이학습 시작점(자동 다운로드)
EPOCHS = 100
IMG_SIZE = 640
BATCH = 16

# --- 서빙 모델 ---
# (구 api_server 용 HOST/PORT 상수는 제거했다. 사용처가 없고 8000 이라 새 포트 9412 와 충돌하는 정보였음.
#  각 서버는 자기 포트를 직접 정한다: 대시보드 7862, YOLO 추론 YOLO_SERVER_PORT=9412)
SERVE_MODEL = NEW_MODEL_PT  # 서빙에 쓸 가중치(.pt 또는 .onnx 로 바꿔도 동작)

# 지원 이미지 확장자(소문자 비교용)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
