#!/bin/bash
# panel400 배치를 GPU 2 에서 yolo11m.pt,yolo11s.pt 순서로 돌린다(두 GPU 병렬로 반씩 나눔).
cd /workspace/data2/jinwoolee/xr_autolearning
CUDA_VISIBLE_DEVICES=2 venv/bin/python /workspace/data2/jinwoolee/run_bench.py   /workspace/data2/jinwoolee/benchset_panel400/data.yaml   /workspace/data2/jinwoolee/xr_autolearning/results/bench/260821_panel400   yolo11m.pt,yolo11s.pt 100 32 12 ram occl   >> /workspace/data2/jinwoolee/panel400_gpu2.log 2>&1
echo "[$(date '+%F %T')] GPU2 종료 (yolo11m.pt,yolo11s.pt)" >> /workspace/data2/jinwoolee/panel400_gpu2.log
