#!/usr/bin/env bash
# 오토러닝의 '신규 모델 적용' 결과를 XR 연동 서버(8001)로 옮기고 핫 리로드한다.
#
# 왜 필요한가
#   대시보드의 적용은 우리 고정 경로(models/model.pt)와 우리 추론 컨테이너(:9412)만 갱신한다.
#   XR 클라이언트가 실제로 쓰는 서버는 호스트에서 도는 8001(다른 폴더, 다른 파일)이라
#   그대로 두면 최신 모델이 반영되지 않는다.
#
# 왜 컨테이너가 직접 복사하지 않는가
#   8001 폴더는 다른 사람(mrod)의 작업물이다. 우리 컨테이너에 그 폴더 쓰기 권한을 주면
#   결합이 생겨, 그쪽이 폴더를 옮기거나 서비스를 바꿀 때 우리 배포가 깨진다.
#   호스트 스크립트 하나만 두면 그쪽 사정이 바뀌어도 이 파일만 고치면 된다.
#
# 동작
#   1) models/model.pt 의 해시를 8001 폴더의 것과 비교
#   2) 다르면 model.pt(+model.onnx)를 복사
#   3) POST /reload 로 재시작 없이 반영
#   4) 결과를 로그에 남긴다(sync_xr_server.log)
#
# 설치(systemd path unit 으로 자동 실행)
#   bash deploy/sync_xr_server.sh --install
# 수동 실행
#   bash deploy/sync_xr_server.sh
set -uo pipefail

# sudo 로 설치할 때 $HOME 이 /root 로 잡혀 엉뚱한 경로를 감시하는 사고가 있었다.
# 실제 사용자(SUDO_USER)의 홈을 기준으로 잡는다.
REAL_USER="${SUDO_USER:-$USER}"
REAL_HOME="$(getent passwd "$REAL_USER" | cut -d: -f6)"
SRC_DIR="${XR_MODELS_DIR:-$REAL_HOME/xr_autolearning/models}"
DST_DIR="${XR_SERVE_DIR:-$REAL_HOME/Desktop/yolo_test_server/yolo_test_server2/yolo_test_server}"
RELOAD_URL="${XR_RELOAD_URL:-http://127.0.0.1:8001/reload}"
LOG="${XR_SYNC_LOG:-$REAL_HOME/sync_xr_server.log}"
UNIT_DIR=/etc/systemd/system

log() { printf '%s %s\n' "$(date '+%F %T')" "$1" | tee -a "$LOG"; }
sha() { [ -f "$1" ] && sha256sum "$1" | cut -c1-16 || echo none; }

install_unit() {
  # 파일이 바뀔 때만 실행(inotify). 3초 지연 = 복사가 끝나기를 기다림.
  local self; self="$(readlink -f "$0")"
  sudo tee "$UNIT_DIR/xr-model-sync.service" >/dev/null <<UNIT
[Unit]
Description=XR 연동 서버(8001)로 최신 모델 동기화
After=network.target

[Service]
Type=oneshot
User=$REAL_USER
Group=$REAL_USER
ExecStartPre=/bin/sleep 3
ExecStart=/bin/bash $self
UNIT
  sudo tee "$UNIT_DIR/xr-model-sync.path" >/dev/null <<UNIT
[Unit]
Description=models/model.pt 변경 감시 (오토러닝 '신규 모델 적용' 시 갱신됨)

[Path]
PathModified=$SRC_DIR/model.pt
Unit=xr-model-sync.service

[Install]
WantedBy=multi-user.target
UNIT
  sudo systemctl daemon-reload
  sudo systemctl enable --now xr-model-sync.path
  log "[install] 감시 등록: $SRC_DIR/model.pt -> $DST_DIR"
  systemctl status xr-model-sync.path --no-pager | sed -n '1,4p'
  exit 0
}

[ "${1:-}" = "--install" ] && install_unit

[ -d "$DST_DIR" ] || { log "[skip] 대상 폴더 없음: $DST_DIR"; exit 0; }
[ -f "$SRC_DIR/model.pt" ] || { log "[skip] 원본 없음: $SRC_DIR/model.pt"; exit 0; }

if [ "$(sha "$SRC_DIR/model.pt")" = "$(sha "$DST_DIR/model.pt")" ]; then
  log "[skip] 이미 같은 모델 ($(sha "$SRC_DIR/model.pt"))"
  exit 0
fi

cp -f "$SRC_DIR/model.pt" "$DST_DIR/model.pt"
if [ -f "$SRC_DIR/model.onnx" ]; then
  cp -f "$SRC_DIR/model.onnx" "$DST_DIR/model.onnx"
else
  rm -f "$DST_DIR/model.onnx"      # pt 와 짝이 안 맞는 낡은 onnx 는 두지 않는다
fi
log "[copy] model.pt $(sha "$DST_DIR/model.pt") 배포"

# .engine 으로 서빙 중이면 파일 복사만으로는 반영되지 않는다(재변환 필요)
if [ -f "$DST_DIR/model.engine" ]; then
  log "[warn] model.engine 이 있다. TensorRT 로 서빙 중이면 재변환이 필요하다(7분)"
fi

code=$(curl -s -m 60 -o /tmp/xr_reload.json -w '%{http_code}' -X POST "$RELOAD_URL" || echo 000)
if [ "$code" = "200" ]; then
  log "[reload] 8001 반영 완료 · $(head -c 160 /tmp/xr_reload.json)"
else
  log "[reload] 실패(HTTP $code) — 다음 기동 때 새 모델을 읽는다"
fi
