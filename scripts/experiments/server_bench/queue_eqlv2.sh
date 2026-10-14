#!/bin/bash
# Plan C — 분류 손실 EQLv2. 앞 배치(queue_260829.sh)가 끝나면 이어서 GPU2 에서 돈다.
# 데이터·증강·에폭은 기준선(260824_retrain3x/yolo11m)과 완전히 같고 손실 함수만 다르다.
set -u
R=/workspace/data2/jinwoolee/xr_autolearning
W=/workspace/data2/jinwoolee
cd "$R"
export XR_BASE="$R"
export CUDA_VISIBLE_DEVICES=2
LOG="$R/results/bench/queue_260829.log"

until grep -q "===== 배치 종료 =====" "$LOG" 2>/dev/null; do
  if ! pgrep -f "[q]ueue_260829.sh" > /dev/null; then
    echo "[$(date '+%F %T')] 앞 배치가 종료 표시 없이 사라짐 - Plan C 그대로 시작" >> "$LOG"; break
  fi
  sleep 120
done

echo "[$(date '+%F %T')] >>> eqlv2_11m 시작 (Plan C)" >> "$LOG"
"$R/venv/bin/python" "$R/scripts/experiments/run_exp.py" --name eqlv2_11m \
  --data "$W/benchset_3x/data.yaml" --weights yolo11m.pt --epochs 100 --eqlv2 >> "$LOG" 2>&1
echo "[$(date '+%F %T')] <<< eqlv2_11m 종료(exit=$?)" >> "$LOG"
echo "[$(date '+%F %T')] ===== Plan C 종료 =====" >> "$LOG"
