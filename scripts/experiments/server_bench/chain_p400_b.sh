#!/bin/bash
# panel400 배치를 GPU 0 에서 yolo26s.pt,yolo11n.pt 순서로 돌린다(두 GPU 병렬로 반씩 나눔).
cd /workspace/data2/jinwoolee/xr_autolearning
CUDA_VISIBLE_DEVICES=0 venv/bin/python /workspace/data2/jinwoolee/run_bench.py   /workspace/data2/jinwoolee/benchset_panel400/data.yaml   /workspace/data2/jinwoolee/xr_autolearning/results/bench/260821_panel400   yolo26s.pt,yolo11n.pt 100 32 12 ram occl   >> /workspace/data2/jinwoolee/panel400_gpu0.log 2>&1
echo "[$(date '+%F %T')] GPU0 종료 (yolo26s.pt,yolo11n.pt)" >> /workspace/data2/jinwoolee/panel400_gpu0.log
