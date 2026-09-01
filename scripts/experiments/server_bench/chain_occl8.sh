#!/bin/bash
# 가림 강도 A/B: benchset_3x 와 OCCL_P(0.5 -> 0.8) 만 다른 데이터로 11m·26s 재학습.
# 앞의 재학습 8종(v8 계열까지)이 끝나면 데이터셋을 만들고 이어서 학습한다. GPU2 고정.
W=/workspace/data2/jinwoolee
until grep -q "v8 계열 종료" "$W/train_3x.log" 2>/dev/null; do sleep 300; done

cd "$W/xr_autolearning"
echo "[$(date '+%F %T')] occl8 데이터셋 생성 시작" >> "$W/train_occl8.log"
CUDA_VISIBLE_DEVICES=2 venv/bin/python "$W/build_occl8.py" >> "$W/train_occl8.log" 2>&1 || {
  echo "[$(date '+%F %T')] 데이터셋 생성 실패 · 중단" >> "$W/train_occl8.log"; exit 1; }

CUDA_VISIBLE_DEVICES=2 venv/bin/python "$W/run_bench.py" \
  "$W/benchset_occl8/data.yaml" \
  "$W/xr_autolearning/results/bench/260825_occl8" \
  yolo11m.pt,yolo26s.pt 100 32 16 ram occl >> "$W/train_occl8.log" 2>&1
echo "[$(date '+%F %T')] occl8 배치 종료" >> "$W/train_occl8.log"
