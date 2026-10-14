#!/bin/bash
# 도메인갭 대책 실험 배치 (GPU2 고정 · 순차). 하나가 죽어도 다음으로 넘어간다.
#
#   1 fewshot   Plan A  기체 장착샷 10장 + 리플레이로 파인튜닝            ~30분
#   2 real      진단3   합성 74% 를 빼고 실촬만으로 재학습                ~2시간
#   3 merged    Plan B  전 부품을 1클래스로 병합한 1단계 검출기           ~7.5시간
#   (4 eqlv2    Plan C  분류 손실 EQLv2 — 준비되면 queue_eqlv2.sh 로 추가)
#
# 각 단계 끝에 GT 70장(gearbox 40 + a_test 30) 자동 채점까지 한다.
set -u
R=/workspace/data2/jinwoolee/xr_autolearning
W=/workspace/data2/jinwoolee
cd "$R"
export XR_BASE="$R"
export CUDA_VISIBLE_DEVICES=2
LOG="$R/results/bench/queue_260829.log"
mkdir -p "$(dirname "$LOG")"

run () {  # run <이름> <추가인자...>
  local name=$1; shift
  echo "[$(date '+%F %T')] >>> $name 시작" >> "$LOG"
  "$R/venv/bin/python" "$R/scripts/experiments/run_exp.py" --name "$name" "$@" >> "$LOG" 2>&1
  echo "[$(date '+%F %T')] <<< $name 종료(exit=$?)" >> "$LOG"
}

echo "[$(date '+%F %T')] ===== 배치 시작 =====" >> "$LOG"

# 1) Plan A — 장착 도메인 퓨샷. 기준 모델에서 이어 학습하므로 lr 을 낮춘다.
run fewshot_11m \
    --data "$W/benchset_fewshot/data.yaml" \
    --weights "$R/results/bench/260824_retrain3x/yolo11m/weights/best.pt" \
    --epochs 30 --lr0 0.001 --cache ram

# 2) 진단3 — 합성 배제(실촬 9,083 + 배경 567). 기준선과 에폭·증강 동일.
run real_11m \
    --data "$W/benchset_real/data.yaml" \
    --weights yolo11m.pt --epochs 100

# 3) Plan B — 1클래스 병합 검출기. 데이터는 기준선과 완전히 같고 클래스 수만 1.
run merged_11m \
    --data "$W/benchset_merged/data.yaml" \
    --weights yolo11m.pt --epochs 100

echo "[$(date '+%F %T')] ===== 배치 종료 =====" >> "$LOG"
