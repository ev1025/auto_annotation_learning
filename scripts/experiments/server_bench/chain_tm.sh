#!/bin/bash
# 4단계(기본 프레임) 학습이 끝나면 5단계(밝기보정 프레임)를 이어 돌린다.
# pgrep -f 를 ssh 원격 명령으로 직접 치면 그 명령 문자열 자체가 패턴에 걸린다.
# 그래서 스크립트 파일 안에서만 쓴다.
cd /workspace/data2/jinwoolee/xr_dash_260901 || exit 1
VENV=/workspace/data2/jinwoolee/xr_autolearning/venv/bin/python

echo "[$(date '+%H:%M:%S')] 4단계 종료 대기"
while pgrep -u "$(whoami)" -f 'train_proc.py' > /dev/null; do
  sleep 60
done
echo "[$(date '+%H:%M:%S')] 4단계 끝. GPU 정리 대기 60초"
sleep 60

echo "[$(date '+%H:%M:%S')] 5단계 시작(밝기보정 autolabels_tm)"
CUDA_VISIBLE_DEVICES=2 \
  XR_AUTOLABELS=autolabels_tm XR_EPOCHS=100 XR_EXPECT=35 \
  XR_TRAIN_BATCH=32 XR_TRAIN_WORKERS=16 XR_TRAIN_CACHE=ram \
  XR_MIXUP=0.2 XR_SYNTH_MULT=3.0 XR_SYNTH_REUSE=0 \
  "$VENV" run_train_parts.py > train_tm_new.log 2>&1
echo "[$(date '+%H:%M:%S')] 5단계 종료(코드 $?)"
