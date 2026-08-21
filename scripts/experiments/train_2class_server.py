# -*- coding: utf-8 -*-
"""서버(RTX5090)에서 오토라벨 결과로 다중클래스 YOLO 를 학습하고, 토르에 넣을 모델을 만든다.

왜 별도 스크립트인가
  대시보드의 학습은 토르 안에서 돌지만, 에포크가 많거나 클래스가 늘면 서버가 훨씬 빠르다.
  이 스크립트는 대시보드와 같은 재료(results/autolabels/<부품>/{images,labels})와
  같은 증강(배경 합성)을 써서 결과를 그대로 토르에 이식할 수 있게 한다.

입력 (자동 탐색)
  results/autolabels/<부품>/images/<영상stem>/00000.jpg   프레임
  results/autolabels/<부품>/labels/<영상stem>_00000.txt    라벨(클래스 0 단일)

출력
  results/experiments/train_multi/<시각>/model/best.pt     가중치
  results/experiments/train_multi/<시각>/meta.json          토르 등록용 메타(클래스 순서 포함)

사용(서버, GPU2 고정)
  cd /workspace/data2/jinwoolee/xr_autolearning
  XR_BASE=$PWD CUDA_VISIBLE_DEVICES=2 XR_CLASSES=a_test,medicine \
    nohup ./venv/bin/python scripts/experiments/train_2class_server.py > /tmp/train_multi.log 2>&1 &
  옵션: XR_EPOCHS(기본 100) · XR_NSYN(합성 장수, 기본 475 = 실측 최적) · XR_MODEL(기본 yolo11s)
        XR_SMOKE=1 (3에포크·합성 8장 빠른 점검)
"""
import glob
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

BASE = os.environ.get("XR_BASE") or str(Path(__file__).resolve().parents[2])
sys.path.insert(0, BASE + "/scripts")                    # config
sys.path.insert(0, BASE + "/backend/autolearning")       # sam2_autolabel(_synth_augment)·autolabel

import config                      # noqa: E402
import sam2_autolabel as sa        # noqa: E402

CLASSES = [c.strip() for c in os.environ.get("XR_CLASSES", "a_test,medicine").split(",") if c.strip()]
IDX = {c: i for i, c in enumerate(CLASSES)}
SMOKE = bool(os.environ.get("XR_SMOKE"))
EPOCHS = 3 if SMOKE else int(os.environ.get("XR_EPOCHS", "100"))
N_SYN = 8 if SMOKE else int(os.environ.get("XR_NSYN", "475"))   # 실측 최적(475장 부근에서 최고)
MODEL = os.environ.get("XR_MODEL", "yolo11s")
AUTOLABELS = Path(BASE) / "results" / "autolabels"
OUT = Path(BASE) / "results" / "experiments" / "train_multi"


def log(msg, level="info"):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def collect(cls):
    """부품 하나의 (이미지, 라벨) 짝을 모은다. 라벨 없는 프레임은 버린다."""
    pairs = []
    for ip in sorted(glob.glob(str(AUTOLABELS / cls / "images" / "*" / "*.jpg"))):
        p = Path(ip)
        stem, num = p.parent.name, p.stem                       # <영상stem>/00123.jpg
        lp = AUTOLABELS / cls / "labels" / f"{stem}_{num}.txt"
        if lp.exists() and lp.stat().st_size > 0:
            pairs.append((p, lp))
    return pairs


def build_dataset(dsdir):
    """클래스별 라벨 인덱스를 다시 매겨 하나의 학습셋으로 합친다."""
    di, dl = dsdir / "images", dsdir / "labels"
    di.mkdir(parents=True, exist_ok=True)
    dl.mkdir(parents=True, exist_ok=True)
    counts = {}
    for cls in CLASSES:
        pairs = collect(cls)
        counts[cls] = len(pairs)
        for ip, lp in pairs:
            name = f"{cls}__{ip.parent.name}_{ip.stem}"
            shutil.copy(ip, di / f"{name}.jpg")
            out = []
            for line in lp.read_text(encoding="utf-8").splitlines():
                f = line.split()
                if len(f) == 5:                                  # 단일클래스(0) -> 이 부품의 인덱스로
                    out.append(f"{IDX[cls]} {f[1]} {f[2]} {f[3]} {f[4]}")
            (dl / f"{name}.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    return di, dl, counts


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    run = OUT / ts
    (run / "model").mkdir(parents=True, exist_ok=True)
    log(f"클래스 {CLASSES} · {MODEL} {EPOCHS}에포크 · 합성 {N_SYN}장 · GPU {gpu}")

    di, dl, counts = build_dataset(run / "ds")
    log("원본 프레임: " + " · ".join(f"{c} {n}장" for c, n in counts.items()))
    if min(counts.values(), default=0) < 5:
        log("프레임이 부족한 클래스가 있다 - 중단", "err")
        return 1

    made = sa._synth_augment(di, dl, log, n_syn=N_SYN) or 0      # 배경 합성 증강(대시보드와 같은 함수)
    log(f"배경 합성 {made}장 추가 · 학습셋 총 {sum(counts.values()) + made}장")

    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    yml = run / "data.yaml"
    yml.write_text(f"train: {di.resolve().as_posix()}\nval: {di.resolve().as_posix()}\nnames:\n{names}\n",
                   encoding="utf-8")

    from ultralytics import YOLO
    t0 = time.time()
    m = YOLO(MODEL + ".pt")
    m.train(data=str(yml), epochs=EPOCHS, imgsz=640, batch=8, device=0,
            project=str(run / "runs"), name="model", exist_ok=True, verbose=False, plots=False,
            degrees=15.0)
    took = (time.time() - t0) / 60

    best = Path(m.trainer.best)
    shutil.copy(best, run / "model" / "best.pt")

    # XR 배포용 ONNX 도 같이 만든다(대시보드 학습 경로와 동일하게. 이게 빠지면 적용 시
    # 고정 경로의 낡은 onnx 가 삭제돼 onnx 로 서빙하던 쪽이 모델을 잃는다)
    onnx_out = None
    try:
        from ultralytics import YOLO as _Y
        src = Path(_Y(str(best)).export(format="onnx", imgsz=640))
        onnx_out = run / "model" / "model.onnx"
        shutil.copy(src, onnx_out)
        log(f"ONNX 변환 완료 -> {onnx_out.name}")
    except Exception as e:   # noqa: BLE001 - onnx 실패가 학습 결과를 버리게 하면 안 된다
        log(f"ONNX 변환 실패(건너뜀): {type(e).__name__}: {e}", "err")
    r = m.trainer.metrics or {}
    meta = {
        "run": ts, "model_id": ts, "session": ts,
        "label": f"{datetime.now():%Y-%m-%d %H:%M} · {len(CLASSES)}종 · {', '.join(CLASSES)}",
        "time": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        "classes": CLASSES, "n_classes": len(CLASSES),
        "weights": f"results/{ts}/model/best.pt",                 # 토르에 넣을 때의 상대경로
        "onnx": f"results/{ts}/model/model.onnx" if onnx_out else None,
        "trained_on": f"RTX5090 GPU{gpu}", "epochs": EPOCHS,
        "n_images": sum(counts.values()), "n_augmented": made,
        "per_class_frames": counts,
        "train_metrics": {k: round(float(v), 4) for k, v in r.items() if isinstance(v, (int, float))},
        "note": "train/val 이 같은 폴더이므로 아래 지표는 학습셋 성능이다(일반화 성능이 아니다)",
    }
    (run / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"학습 완료 {took:.1f}분 -> {run / 'model' / 'best.pt'}")
    log("지표(학습셋 기준): " + json.dumps(meta["train_metrics"], ensure_ascii=False))
    shutil.rmtree(run / "ds", ignore_errors=True)                # 수백 MB - 가중치·메타만 남긴다
    print(f"\nDONE {run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
