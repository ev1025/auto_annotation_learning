"""exp_epochs.py - 벤치마크 후속 검증 실험 3종 (일회성).

100ep 벤치마크(results/dashboard/benchmark/benchmark.md)의 결정 보강용:
  ① 300ep 연장 + 조기종료: v8s / 11n (100ep 시점까지 상승 중이던 두 조합의 상한 확인)
  ② 26n 레시피 조건(245ep, lr0 0.0054, MuSGD): 동일 100ep 벤치의 26n 저평가 가설 검증
  ③ 시드 반복(26s / v8s x seed 1,2): 1위-4위 간 1.1%p 차이가 노이즈인지 판별

실행(GPU별 그룹):
  CUDA_VISIBLE_DEVICES=N python scripts/exp_epochs.py --group gpuN
결과: results/dashboard/benchmark/exp_epochs_<group>.json (test split 기준, 벤치마크와 동일 평가)
"""
import argparse
import csv
import json
import time
from pathlib import Path

from ultralytics import YOLO

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ 공용(config 등)
import config
from gpu_utils import free_cuda

BENCH = config.BASE_DIR / "results/dashboard/benchmark"
DATA = BENCH / "bench_data.yaml"  # experiments/benchmark 가 생성해 둔 것을 재사용(동일 분할 보장)

# 부하 균형: 긴 작업을 앞에, 총 소요 30~50분/GPU
GROUPS = {
    "gpu0": [dict(model="yolov8s.pt", name="v8s_300ep", epochs=300, patience=50)],
    "gpu1": [dict(model="yolo11n.pt", name="11n_300ep", epochs=300, patience=50),
             dict(model="yolov8s.pt", name="v8s_seed1", epochs=100, seed=1)],
    "gpu2": [dict(model="yolo26n.pt", name="26n_recipe", epochs=245, lr0=0.0054, optimizer="MuSGD"),
             dict(model="yolov8s.pt", name="v8s_seed2", epochs=100, seed=2)],
    "gpu3": [dict(model="yolo26s.pt", name="26s_seed1", epochs=100, seed=1),
             dict(model="yolo26s.pt", name="26s_seed2", epochs=100, seed=2)],
}


def run_one(c):
    kw = dict(data=str(DATA), imgsz=640, batch=-1, device=0, epochs=c["epochs"],
              project=str(BENCH / "runs"), name=c["name"], exist_ok=True, verbose=False)
    for k in ("patience", "seed", "lr0", "optimizer"):
        if k in c:
            kw[k] = c[k]
    # lr0 를 지정할 땐 optimizer 를 명시해야 함(auto 는 lr0 를 덮어씀)
    print(f"\n===== [{c['name']}] {c['model']} epochs={c['epochs']} =====")
    t0 = time.perf_counter()
    model = YOLO(c["model"])
    try:
        model.train(**kw)
    except Exception as e:
        if kw.get("optimizer") == "MuSGD":  # 이 버전에 MuSGD 가 없으면 SGD 폴백
            print(f"[폴백] MuSGD 실패({e}) -> SGD 로 재시도")
            kw["optimizer"] = "SGD"
            del model
            free_cuda()
            model = YOLO(c["model"])
            model.train(**kw)
        else:
            raise
    best = Path(model.trainer.best)
    n_epochs_run = len(list(csv.DictReader(open(best.parent.parent / "results.csv"))))
    del model
    free_cuda()

    m = YOLO(str(best))
    res = m.val(data=str(DATA), split="test", imgsz=640, verbose=False)
    row = dict(name=c["name"], base=c["model"],
               epochs_set=c["epochs"], epochs_run=n_epochs_run,
               optimizer=kw.get("optimizer", "auto"), seed=c.get("seed", 0),
               map50=round(float(res.box.map50), 4),
               map50_95=round(float(res.box.map), 4),
               train_min=round((time.perf_counter() - t0) / 60, 1))
    del m, res
    free_cuda()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=sorted(GROUPS))
    args = ap.parse_args()

    rows = []
    for c in GROUPS[args.group]:
        try:
            rows.append(run_one(c))
        except Exception as e:  # 한 건 실패해도 나머지 계속
            rows.append(dict(name=c["name"], error=str(e)[:200]))
            print(f"[실패] {c['name']}: {e}")
        (BENCH / f"exp_epochs_{args.group}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[저장] {rows[-1]}")
    print("완료:", args.group)


if __name__ == "__main__":
    main()
