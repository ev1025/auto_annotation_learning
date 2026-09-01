#!/bin/bash
# 26 계열 재학습(GPU2, 같은 데이터셋·레시피). 앞 단계가 끝나면 시작한다.
W=/workspace/data2/jinwoolee
until grep -q "11 계열 종료" "$W/train_3x.log" 2>/dev/null; do sleep 120; done
cd "$W/xr_autolearning"
CUDA_VISIBLE_DEVICES=2 venv/bin/python "$W/run_bench.py"   "$W/benchset_3x/data.yaml"   "$W/xr_autolearning/results/bench/260824_retrain3x"   yolo26m.pt,yolo26s.pt,yolo26n.pt 100 32 16 ram occl >> "$W/train_3x.log" 2>&1
echo "[$(date '+%F %T')] 26 계열 종료" >> "$W/train_3x.log"
