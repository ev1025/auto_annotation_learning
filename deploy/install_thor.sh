#!/usr/bin/env bash
# 새 Jetson Thor 한 대에 XR 오토러닝 전체(DB · 대시보드 · 추론서버)를 올린다.
#
# 배경: 코드/영상/모델은 개발기(윈도우)에서 tar 로 묶어 보내고, 컨테이너는 Thor 에서 빌드한다.
#       Thor 는 aarch64(SBSA) 라 x86 이미지를 그대로 못 쓰고, JetPack 별 CUDA 도 다르다.
#
# 개발기에서 묶기(Git Bash):
#   cd <repo>
#   tar -czf /tmp/xr_code.tar.gz backend scripts deploy requirements.txt \
#        frontend/dist frontend/src frontend/index.html frontend/package.json frontend/vite.config.js
#   tar -cf  /tmp/xr_data.tar    data/bell412 data/_eval
#   tar -cf  /tmp/xr_models.tar  models/sam2 models/model.pt models/model.onnx
#   tar -cf  /tmp/xr_results.tar results/_served.json results/<서비스런>/model/best.pt \
#        $(find results/autolabels -maxdepth 2 \( -name shots.json -o -name labels -type d \))
#   # results/autolabels 의 images·boxs·_cuts 는 영상에서 다시 만들어지므로 보내지 않는다(8GB 절약)
#
# Thor 에서:
#   mkdir -p ~/xr_transfer && (개발기에서 sftp 로 4개 tar 업로드)
#   bash deploy/install_thor.sh
set -euo pipefail

REPO="${XR_REPO:-$HOME/xr_autolearning}"
TRANSFER="${XR_TRANSFER:-$HOME/xr_transfer}"
COMPOSE="docker compose --env-file deploy/.env -f deploy/docker-compose.thor.yml"

say() { printf '\n=== %s ===\n' "$1"; }

say "0) 사전 점검"
command -v docker >/dev/null || { echo "docker 없음"; exit 1; }
docker info >/dev/null 2>&1 || { echo "docker 권한 없음 - 'sudo usermod -aG docker $USER' 후 재로그인"; exit 1; }
grep -q '"nvidia"' /etc/docker/daemon.json 2>/dev/null || echo "  주의: nvidia 런타임 미등록(학습·SAM2 가 CPU 로 떨어진다)"
for p in 7862 9412 5432; do
  if ss -ltn "( sport = :$p )" 2>/dev/null | grep -q LISTEN; then
    echo "  포트 $p 가 이미 사용 중 - 다른 서비스와 충돌한다. compose 의 ports 를 바꿔라"; exit 1
  fi
done
echo "  docker $(docker --version | awk '{print $3}' | tr -d ,) · 포트 3개 비어 있음"

say "1) 파일 풀기"
mkdir -p "$REPO"
for f in xr_code.tar.gz xr_data.tar xr_models.tar xr_results.tar; do
  [ -f "$TRANSFER/$f" ] && { echo "  $f"; tar -xf "$TRANSFER/$f" -C "$REPO"; }
done
cd "$REPO"
[ -f models/sam2/sam2.1_hiera_base_plus.pt ] || echo "  주의: SAM2 체크포인트가 없다(참조샷 라벨링 불가)"

say "2) .env"
if [ ! -f deploy/.env ]; then
  cp deploy/.env.example deploy/.env
  # DB 는 컨테이너 내부에서만 쓰므로 임의 비밀번호로 충분하다
  PW=$(head -c 18 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 20)
  sed -i "s/^XR_DB_PASSWORD=.*/XR_DB_PASSWORD=$PW/" deploy/.env
  echo "  deploy/.env 생성(비밀번호 자동 생성)"
else
  echo "  기존 deploy/.env 사용"
fi

say "3) 이미지 빌드 (torch 다운로드 때문에 10~20분)"
$COMPOSE build app

say "4) 기동"
$COMPOSE up -d
sleep 8
docker ps --filter name=xr_ --format '  {{.Names}}  {{.Status}}  {{.Ports}}'

say "5) 파일 -> DB 색인 (최초 1회)"
$COMPOSE exec -T app python3 backend/db/migrate_from_files.py || echo "  색인 실패 - 로그 확인: docker logs xr_app"

say "6) 확인"
curl -s -m 10 -o /dev/null -w '  대시보드 7862 -> %{http_code}\n' http://127.0.0.1:7862/
curl -s -m 10 -o /dev/null -w '  추론 9412    -> %{http_code}\n' http://127.0.0.1:9412/health
curl -s -m 10 http://127.0.0.1:7862/api/parts | python3 -c 'import sys,json;print("  부품", len(json.load(sys.stdin)["parts"]), "종")' || true
IP=$(hostname -I | awk '{print $1}')
printf '\n접속: http://%s:7862 (대시보드) · http://%s:9412/docs (추론 API)\n' "$IP" "$IP"
