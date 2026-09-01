#!/bin/bash
# 데이터셋 빌드(build_3x)가 끝나면 yolo11m 재학습을 GPU2 에서 시작한다.
# 레시피 = occl(mixup 0.2 + erasing 0.5 + degrees 15) — 7/30 검증, panel400 최고 성적 조합.
W=/workspace/data2/jinwoolee
until grep -q "^완료" "$W/build_3x.log" 2>/dev/null; do
  # 빌드가 죽었으면 대기도 끝낸다
  if ! pgrep -f "[b]uild_3x.py" > /dev/null && ! grep -q "^완료" "$W/build_3x.log" 2>/dev/null; then
    echo "[$(date '+%F %T')] 빌드가 완료 없이 종료됨 - 학습 시작 안 함" >> "$W/train_3x.log"; exit 1
  fi
  sleep 60
done
cd "$W/xr_autolearning"
CUDA_VISIBLE_DEVICES=2 venv/bin/python "$W/run_bench.py" \
  "$W/benchset_3x/data.yaml" \
  "$W/xr_autolearning/results/bench/260824_retrain3x" \
  yolo11m.pt 100 32 16 ram occl >> "$W/train_3x.log" 2>&1
echo "[$(date '+%F %T')] 재학습(yolo11m) 종료" >> "$W/train_3x.log"
