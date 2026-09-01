#!/bin/bash
# GPU 0 의 현재 배치가 끝나면(로그 종료 표시 확인) yolo26n.pt,yolov8s.pt 를 이어서 돌린다.
L=/workspace/data2/jinwoolee/panel400_gpu0.log
until grep -q "GPU0 종료" "$L" 2>/dev/null; do sleep 120; done
cd /workspace/data2/jinwoolee/xr_autolearning
CUDA_VISIBLE_DEVICES=0 venv/bin/python /workspace/data2/jinwoolee/run_bench.py   /workspace/data2/jinwoolee/benchset_panel400/data.yaml   /workspace/data2/jinwoolee/xr_autolearning/results/bench/260821_panel400   yolo26n.pt,yolov8s.pt 100 32 12 ram occl >> "$L" 2>&1
echo "[$(date '+%F %T')] GPU0 추가분 종료 (yolo26n.pt,yolov8s.pt)" >> "$L"
