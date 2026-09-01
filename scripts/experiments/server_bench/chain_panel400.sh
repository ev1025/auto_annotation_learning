#!/bin/bash
# 배경984 + 합성14127(판넬 가림) + mixup·erasing. GPU2 고정. 11m·11s 를 먼저 돌린다.
# venv 는 레포 안에 있으므로 반드시 레포에서 실행한다(이전 실패 원인).
cd /workspace/data2/jinwoolee/xr_autolearning
CUDA_VISIBLE_DEVICES=2 venv/bin/python /workspace/data2/jinwoolee/run_bench.py \
  /workspace/data2/jinwoolee/benchset_panel400/data.yaml \
  /workspace/data2/jinwoolee/xr_autolearning/results/bench/260821_panel400 \
  yolo11m.pt,yolo11s.pt,yolo11n.pt,yolo26s.pt 100 32 16 ram occl \
  >> /workspace/data2/jinwoolee/panel400.log 2>&1
echo "[$(date '+%F %T')] panel400 배치 종료" >> /workspace/data2/jinwoolee/panel400.log
