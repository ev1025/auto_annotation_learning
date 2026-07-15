"""config.py - 파이프라인 전역 설정 (경로 / 클래스 / 임계값).

세 실행 스크립트(1_auto_labeling, 2_train_pipeline, 3_api_server)가
공통으로 참조하는 값을 한 곳에 모은다.
- 경로를 각 스크립트에 하드코딩하지 않고 여기서만 바꾸면 전체 파이프라인이 따라오도록 모듈화.
- 숫자로 시작하는 실행 스크립트(1_, 2_, 3_)는 파이썬 식별자 규칙상 import 가 불가능하므로,
  '공유 상수'는 import 가능한 이 모듈(config.py)에 둔다.
"""
from pathlib import Path

# 프로젝트 루트 절대경로(이 파일은 scripts/ 안에 있으므로 한 단계 위).
# 어느 작업 디렉토리에서 실행해도 경로가 깨지지 않게 한다.
BASE_DIR = Path(__file__).resolve().parent.parent

# --- 모델 가중치 ---
MODELS_DIR = BASE_DIR / "models"
BASE_MODEL = MODELS_DIR / "base_model.pt"       # 소량 데이터로 미리 학습한 초기 가중치
NEW_MODEL_PT = MODELS_DIR / "new_model.pt"      # 본 학습 결과(best.pt)를 복사해 둘 위치
NEW_MODEL_ONNX = MODELS_DIR / "new_model.onnx"  # ONNX 변환 산출물(Unity/C# 등 비파이썬 런타임용)

# --- 데이터셋 경로 ---
DATASETS_DIR = BASE_DIR / "datasets"
UNLABELED_DIR = DATASETS_DIR / "unlabeled_images"  # 자동 라벨링 입력(원본)
IMAGES_DIR = DATASETS_DIR / "images"   # 라벨링된 이미지(ultralytics 가 labels 와 짝지음)
LABELS_DIR = DATASETS_DIR / "labels"   # YOLO 포맷 .txt 정답
DATA_YAML = BASE_DIR / "data.yaml"

# --- 임계값 ---
AUTO_LABEL_CONF = 0.6   # 자동 라벨링: 이 값 이상만 '정답'으로 신뢰
API_CONF = 0.25         # 추론 API 기본 confidence 임계값
API_IOU = 0.45          # 추론 API 기본 NMS IoU 임계값

# --- 학습 하이퍼파라미터 기본값 ---
PRETRAINED = "yolov8n.pt"  # base_model 이 없을 때 전이학습 시작점(자동 다운로드). n=경량, 데모/엣지용
EPOCHS = 100
IMG_SIZE = 640
BATCH = 16
VAL_RATIO = 0.2  # 학습 / 검증 분할 비율(val 비율)

# --- API 서버 ---
HOST = "0.0.0.0"
PORT = 8000
SERVE_MODEL = NEW_MODEL_PT  # 서빙에 쓸 가중치(.pt 또는 .onnx 로 바꿔도 동작)

# 지원 이미지 확장자(소문자 비교용)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
