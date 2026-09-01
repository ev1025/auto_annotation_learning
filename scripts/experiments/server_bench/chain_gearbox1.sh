#!/bin/bash
# gearbox 1클래스 실험. 36클래스 재학습(benchset_3x)과 레시피는 같고 클래스 수만 다르다.
# 목적: 36종 희석이 test4(기체 장착)를 죽이는지 확인. 7/30 에 1클래스로는 test4 가 잡혔다.
# 앞의 occl8 배치가 끝나면 시작한다. GPU2 고정.
W=/workspace/data2/jinwoolee
until grep -q "occl8 배치 종료" "$W/train_occl8.log" 2>/dev/null; do sleep 300; done

cd "$W/xr_autolearning"
echo "[$(date '+%F %T')] gearbox1 데이터셋 생성 시작" >> "$W/train_gearbox1.log"
CUDA_VISIBLE_DEVICES=2 venv/bin/python "$W/build_gearbox1.py" >> "$W/train_gearbox1.log" 2>&1 || {
  echo "[$(date '+%F %T')] 데이터셋 생성 실패 · 중단" >> "$W/train_gearbox1.log"; exit 1; }

CUDA_VISIBLE_DEVICES=2 venv/bin/python "$W/run_bench.py" \
  "$W/benchset_gearbox1/data.yaml" \
  "$W/xr_autolearning/results/bench/260826_gearbox1" \
  yolo11s.pt 100 32 16 ram occl >> "$W/train_gearbox1.log" 2>&1
echo "[$(date '+%F %T')] gearbox1 배치 종료" >> "$W/train_gearbox1.log"
