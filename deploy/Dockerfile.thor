# Jetson Thor(aarch64 · L4T) 용 앱 이미지.
# 대시보드(등록·라벨·학습·평가)와 추론 서버가 같은 의존성을 쓰므로 이미지는 하나만 만들고
# 실행 커맨드로 역할을 나눈다(compose 의 app / infer 서비스).
#
# torch·torchvision 은 pip 기본 휠이 Jetson 용으로 없다. L4T 베이스 이미지에 들어 있는 것을
# 그대로 쓰고, requirements 에서 torch 관련은 설치하지 않는다(--no-deps 로 덮어쓰기 방지).
#
# ponytail: 베이스 태그는 실기의 JetPack 버전에 맞춰야 한다. 확인 방법은 Thor 에서
#   cat /etc/nv_tegra_release   (또는 dpkg -l | grep nvidia-l4t-core)
# 그 뒤 아래 기본값을 바꾸거나 빌드 시 --build-arg L4T_BASE=... 로 넘긴다.
ARG L4T_BASE=nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.2-py3
FROM ${L4T_BASE}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XR_DB_READS=1 \
    DASH_HOST=0.0.0.0 \
    YOLO_SERVER_HOST=0.0.0.0

# opencv 런타임 의존(헤드리스 환경에 없는 것들)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 먼저(레이어 캐시). torch·torchvision 은 베이스 것을 유지한다.
COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip && \
    grep -v -E '^\s*(torch|torchvision)\b' requirements.txt > /tmp/req.txt && \
    python3 -m pip install -r /tmp/req.txt

# 애플리케이션 코드 + 미리 빌드한 프론트(dist).
# Jetson 에서 node 를 깔지 않으려고 개발기에서 빌드한 산출물을 그대로 넣는다.
COPY backend/ ./backend/
COPY scripts/ ./scripts/
COPY frontend/dist/ ./frontend/dist/

# data · results · models 는 이미지에 넣지 않고 볼륨으로 마운트한다(합계 10GB 이상).
# DB 에 저장된 경로가 저장소 루트 기준 상대경로라, 컨테이너 안에서도 /app 기준으로 그대로 맞는다.
VOLUME ["/app/data", "/app/results", "/app/models"]

EXPOSE 7862 9412
CMD ["python3", "backend/autolearning/dashboard_api.py"]
