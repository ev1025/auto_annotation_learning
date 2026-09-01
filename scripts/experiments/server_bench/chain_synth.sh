#!/bin/bash
# 격납고 배치가 끝나면 합성 2121 배치를 이어서 돌린다. GPU2 고정.
# 대괄호 트릭: 이 스크립트 자신이 pgrep 에 걸려 무한 대기하는 것을 막는다.
cd /workspace/data2/jinwoolee
while pgrep -f "[r]un_bench.py .*260821_093658_hangar" > /dev/null; do sleep 120; done
echo "[$(date '+%F %T')] 격납고 배치 종료 확인 -> 합성 배치 시작" >> synth_chain.log
CUDA_VISIBLE_DEVICES=2 venv/bin/python run_bench.py \
  /workspace/data2/jinwoolee/benchset_synth/data.yaml \
  /workspace/data2/jinwoolee/xr_autolearning/results/bench/260821_101947_synth2160 \
  yolo11m.pt,yolo11s.pt,yolo11n.pt,yolo26s.pt 100 32 16 ram >> synth_chain.log 2>&1
echo "[$(date '+%F %T')] 합성 배치 종료" >> synth_chain.log
