# -*- coding: utf-8 -*-
"""실험 1건 = 학습 + GT 70장 채점. 배치 큐에서 한 줄씩 호출한다.

기준선(results/bench/260824_retrain3x/yolo11m)과 '한 가지만' 다르게 두려고 레시피를 고정한다.
    imgsz 640 · batch 32 · workers 16 · cache ram · seed 0 · deterministic
    증강 occl = degrees 15 + mixup 0.2 + erasing 0.5   (7/30 검증 조합)
바뀌는 것은 --data(데이터셋 구성)와 --weights(시작 가중치)뿐이다.

사용
  XR_BASE=$PWD CUDA_VISIBLE_DEVICES=2 venv/bin/python scripts/experiments/run_exp.py \
      --name real_11m --data /workspace/data2/jinwoolee/benchset_real/data.yaml \
      --weights yolo11m.pt --epochs 100
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BASE = Path(os.environ.get("XR_BASE") or Path(__file__).resolve().parents[2])

AUG = {
    # 7/30 에 장착 도메인(test3·4)을 뚫었던 조합. 기준선 재학습이 쓴 값이라 그대로 맞춘다.
    "occl": {"degrees": 15.0, "mixup": 0.2, "erasing": 0.5},
    "plain": {"degrees": 15.0},
    # 소형 객체 대비: 스케일 지터를 키운다(test4 는 640 기준 115px 로 학습셋 하위 15%)
    "smallobj": {"degrees": 15.0, "mixup": 0.2, "erasing": 0.5, "scale": 0.9},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default="yolo11m.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--cache", default="ram")
    ap.add_argument("--aug", default="occl", choices=list(AUG))
    ap.add_argument("--lr0", type=float, default=None, help="파인튜닝은 0.001 권장")
    ap.add_argument("--out", default=None)
    ap.add_argument("--eqlv2", action="store_true", help="분류 손실을 EQLv2 로 교체(Plan C)")
    ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args()

    out = Path(a.out) if a.out else BASE / "results/bench" / f"260829_{a.name}"
    out.mkdir(parents=True, exist_ok=True)

    if a.eqlv2:                      # 학습 전에 손실을 갈아끼운다(venv 는 건드리지 않는다)
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import eqlv2_loss
        eqlv2_loss.install()
        print("[Plan C] 분류 손실 = EQLv2", flush=True)

    from ultralytics import YOLO
    kw = dict(AUG[a.aug])
    if a.lr0 is not None:
        kw["lr0"] = a.lr0
        kw["warmup_epochs"] = 1.0    # 이어 학습이라 워밍업을 길게 둘 이유가 없다
        kw["cos_lr"] = True
    t0 = time.time()
    print(f"\n===== [{a.name}] 학습 시작  data={a.data}  from={a.weights}  "
          f"epochs={a.epochs} batch={a.batch} aug={a.aug} {kw} =====", flush=True)
    model = YOLO(a.weights)
    model.train(data=a.data, epochs=a.epochs, imgsz=640, batch=a.batch, device=0,
                workers=a.workers, cache=a.cache, project=str(out), name="yolo11m",
                exist_ok=True, verbose=True, plots=True, seed=0, deterministic=True, **kw)
    tr = time.time() - t0
    best = Path(model.trainer.best)
    mv = model.val(data=a.data, imgsz=640, batch=a.batch, device=0, workers=a.workers,
                   project=str(out), name="yolo11m_val", exist_ok=True, verbose=False)
    row = {"name": a.name, "data": a.data, "weights_from": a.weights, "epochs": a.epochs,
           "aug": a.aug, "lr0": a.lr0, "eqlv2": a.eqlv2, "train_sec": round(tr, 1),
           "best": str(best), "insample_map50": round(float(mv.box.map50), 4),
           "insample_map5095": round(float(mv.box.map), 4),
           "finished": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    (out / "bench.json").write_text(json.dumps(row, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"===== [{a.name}] 학습 완료 {tr/60:.1f}분 · 학습셋 mAP50 {row['insample_map50']} =====", flush=True)
    del model

    if not a.skip_eval:              # 학습 로그 mAP 는 포화라 의미 없다. 진짜 점수는 GT 70장.
        cmd = [sys.executable, str(BASE / "scripts/experiments/eval_gt70.py"),
               f"{a.name}={best}", "--out", str(out / "gt70")]
        print("\n[GT 70장 채점]", " ".join(cmd), flush=True)
        subprocess.run(cmd, env={**os.environ, "XR_BASE": str(BASE)}, check=False)


if __name__ == "__main__":
    main()
