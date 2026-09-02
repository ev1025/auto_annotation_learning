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
#
# 2026-09-01 11s -> 11m 으로 올림. 위 벤치의 mAP 는 학습셋과 검증셋이 같은 값이라
# 크기 차이가 안 드러났다. 사람이 그린 GT 40장으로 다시 재니 기체 장착 상태에서 갈렸다.
#   11s  구간3 2/10 · 구간4 0/10      11m  구간3 8/10 · 구간4 1/10   (conf 0.25, 36클래스)
# 서비스 중인 모델도 11m 이라 대시보드 재학습으로 11s 가 올라가면 성능이 내려간다.
PRETRAINED = "yolo11m.pt"  # base_model 이 없을 때 전이학습 시작점(자동 다운로드)
EPOCHS = 100
IMG_SIZE = 640
BATCH = 16
# 대시보드 학습(부품 멀티클래스)의 배치. 위 BATCH 는 구 파이프라인용이라 따로 둔다.
# 기본값은 실제로 학습을 돌리는 기기(Thor 통합메모리 122GB, 서버 RTX5090 32GB)에 맞춘다.
# 메모리가 작은 개발 PC 에서는 환경변수로 낮춰서 띄운다.
#   Thor / RTX 5090 + 11m : 32      RTX 4060 8GB + 11m : 4 (XR_TRAIN_BATCH=4)
# 넘치면 CUDA out of memory 로 학습이 죽는다.
TRAIN_BATCH = 32
# 대시보드 학습의 배경 합성 증강 배수(실촬 대비). 지금 서비스 중인 모델이 3배 레시피다.
#   실촬 9,083 + 합성 26,820(3배·판넬가림) + 배경 567 = 36,470장
# 0 이면 합성을 건너뛴다. 환경변수 XR_SYNTH_MULT 로 덮어쓸 수 있다.
# 주의: 배수를 올리면 합성 생성 시간과 디스크가 그만큼 늘어난다(5090 에서 3배가 약 7분).
SYNTH_MULT = 3.0
# 대시보드 학습을 지금 서비스 중인 모델(서버 run 260824_retrain3x/yolo11m, 8001·9412 양쪽에
# 올라가 있는 그 가중치)과 같은 조건으로 맞춘 값들이다. 서버 args.yaml 에서 확인했다.
#   mixup 0.2 + erasing 0.5 = 서버에서 'occl' 이라 부른 레시피. 부품이 기체에 장착돼 반쯤
#   가려진 구간(평가 구간 3·4)이 이 조합에서 0·3% -> 99·80% 로 올라갔다. erasing 0.5 는
#   ultralytics 기본값이라 따로 넘기지 않고, 기본이 0.0 인 mixup 만 넘긴다.
MIXUP = 0.2
# 이미지를 램에 올려 두고 학습한다. 이 값은 x86 서버에서만 쓰인다.
# 토르(aarch64)는 코드에서 none 으로 내린다 - 35,276장을 램에 올리는 동안 프로세스가
# I/O 대기로 멈춰 대시보드 API 가 함께 죽었다(실측: 정적 파일도 20초 타임아웃).
# 학습이 대시보드와 같은 프로세스의 스레드인 구조 때문이고, 그 구조를 고치면 되돌릴 수 있다.
TRAIN_CACHE = "ram"
# 데이터 적재 워커 수(서버 16). 0 은 적재를 학습 스레드가 직접 해서 GPU 가 굶는다.
# 이 값은 x86 서버에서만 쓰인다. 윈도우와 Jetson(토르)은 코드에서 0 으로 내린다.
#   윈도우: spawn 이 pickle truncated 로 죽는다.
#   Jetson: CUDA 를 올린 프로세스가 워커를 포크하면 교착 - 에포크 뒤 학습이 멈춘다(실측).
TRAIN_WORKERS = 16


# --- 서빙 모델 ---
# (구 api_server 용 HOST/PORT 상수는 제거했다. 사용처가 없고 8000 이라 새 포트 9412 와 충돌하는 정보였음.
#  각 서버는 자기 포트를 직접 정한다: 대시보드 7862, YOLO 추론 YOLO_SERVER_PORT=9412)
SERVE_MODEL = NEW_MODEL_PT  # 서빙에 쓸 가중치(.pt 또는 .onnx 로 바꿔도 동작)

# 지원 이미지 확장자(소문자 비교용)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
