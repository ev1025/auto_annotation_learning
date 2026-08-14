# Jetson AGX Thor(aarch64 · JetPack 7 / L4T R38) 용 앱 이미지.
# 대시보드(등록·라벨·학습·평가)와 추론 서버가 같은 의존성을 쓰므로 이미지는 하나만 만들고
# 실행 커맨드로 역할을 나눈다(compose 의 app / infer 서비스).
#
# 베이스 선택 이유(실기 검증, 2026-08-14):
#   NGC 의 l4t-pytorch 는 R38 태그가 없고, pytorch:*-py3-igpu(25.09·26.02) 는 torch 가
#   sm_87(Orin) 로만 빌드돼 Thor(sm_110)에서 PTX JIT 폴백으로 돈다(matmul 오차 0.03).
#   Thor 는 SBSA 계열이라 pytorch.org 의 표준 aarch64 cu130 휠이 맞는다.
#   torch 2.13.0+cu130 arch_list = ['sm_80','sm_90','sm_100','sm_110','sm_120'] 확인,
#   같은 연산 오차 0.0001, conv+backward 정상.
ARG CUDA_BASE=nvidia/cuda:13.0.1-runtime-ubuntu24.04
FROM ${CUDA_BASE}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XR_DB_READS=1 \
    DASH_HOST=0.0.0.0 \
    YOLO_SERVER_HOST=0.0.0.0 \
    PATH=/opt/venv/bin:$PATH

# python + opencv 런타임 의존(헤드리스 환경에 없는 것들)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev git \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 는 시스템 파이썬이 externally-managed 라 venv 를 만들어 쓴다
RUN python3 -m venv /opt/venv && pip install --upgrade pip

WORKDIR /app

# 1) torch 먼저(가장 무겁고 잘 안 바뀌는 레이어). Thor = sm_110 이라 cu130 휠을 쓴다.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 2) 나머지 의존성. torch 계열은 위에서 깐 것을 유지한다(재설치되면 CPU 휠로 덮일 수 있음).
COPY requirements.txt ./
RUN grep -v -E '^\s*(torch|torchvision)\b' requirements.txt > /tmp/req.txt && \
    pip install -r /tmp/req.txt

# 3) 애플리케이션 코드 + 미리 빌드한 프론트(dist).
#    Jetson 에 node 를 깔지 않으려고 개발기에서 빌드한 산출물을 그대로 넣는다.
COPY backend/ ./backend/
COPY scripts/ ./scripts/
COPY frontend/dist/ ./frontend/dist/

# data · results · models 는 이미지에 넣지 않고 볼륨으로 마운트한다(합계 10GB 이상).
# DB 에 저장된 경로가 저장소 루트 기준 상대경로라, 컨테이너 안에서도 /app 기준으로 그대로 맞는다.
VOLUME ["/app/data", "/app/results", "/app/models"]

EXPOSE 7862 9412
CMD ["python3", "backend/autolearning/dashboard_api.py"]
