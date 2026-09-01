#!/bin/bash
# 11m 이 끝나면 나머지 7모델을 이어서 돌린다(같은 데이터셋·같은 레시피, GPU2 고정).
# 무거운 것부터: 26m -> v8m -> 11s -> 26s -> v8s -> 11n -> 26n
W=/workspace/data2/jinwoolee
until grep -q "재학습(yolo11m) 종료" "$W/train_3x.log" 2>/dev/null; do sleep 120; done
cd "$W/xr_autolearning"
CUDA_VISIBLE_DEVICES=2 venv/bin/python "$W/run_bench.py" \
  "$W/benchset_3x/data.yaml" \
  "$W/xr_autolearning/results/bench/260824_retrain3x" \
  yolo26m.pt,yolov8m.pt,yolo11s.pt,yolo26s.pt,yolov8s.pt,yolo11n.pt,yolo26n.pt \
  100 32 16 ram occl >> "$W/train_3x.log" 2>&1
echo "[$(date '+%F %T')] 재학습 나머지 7모델 종료" >> "$W/train_3x.log"
