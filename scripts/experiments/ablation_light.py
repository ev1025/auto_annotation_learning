# -*- coding: utf-8 -*-
"""gearbox 성능 저하 원인 분리 실험 — 조명 정합인가, 학습 데이터 개수인가.

배경: gearbox GT(test1~4, 격납고·장착 도메인) 평균 밝기 104.
      학습 영상 Gearbox_gearbox1 은 밝기 152(밝은 실내), train1 은 107(GT 와 일치).
      최근 ablation 에서 gearbox mAP50 이 0.49~0.53 으로 낮았는데, 7/31 에는 0.997 이었다.
      그때는 train1+train2 493장이었고 지금은 Gearbox_gearbox1 206장이다.
      -> 조명 차이와 데이터 개수가 섞여 있어 분리가 안 된다.

설계(gearbox 단일 클래스, GT 40장 고정):
  A. g1_206      Gearbox_gearbox1 206장   조명 불일치(152) · 개수 206
  B. train1_206  train1 에서 206장 균등   조명 일치(107)   · 개수 206   <- A 와 조명만 다름
  C. train1_292  train1 전량 292장        조명 일치(107)   · 개수 292   <- B 와 개수만 다름
각 조건을 합성 증강 ON/OFF 로 돌려 합성 효과까지 같이 본다(총 6런).

고정: yolo11s · 100 epochs · imgsz 640 · batch 8 · 합성 500장 · 부품당 누끼 10장(직전 실험 최적)

사용(서버, GPU2 고정):
  cd /workspace/data2/jinwoolee/xr_autolearning
  XR_BASE=$PWD CUDA_VISIBLE_DEVICES=2 ./venv/bin/python scripts/experiments/ablation_light.py
  # 스모크: ABL_SMOKE=1
결과: results/experiments/ablation_light/<시각>/{results.json,summary.txt,run.log}
"""
import glob
import json
import os
import shutil
import statistics as st
import sys
import time
from datetime import datetime
from pathlib import Path

BASE = os.environ.get("XR_BASE") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, BASE + "/backend/autolearning")
import sam2_autolabel as sa            # _synth_augment · CUT_PER_CLASS

PART = "gearbox"
STORE = f"{BASE}/results/autolabels/{PART}"
GT_SRC = f"{BASE}/data/bell412/{PART}/gt"
OUT = f"{BASE}/results/experiments/ablation_light"
MODEL = os.environ.get("ABL_MODEL", "yolo11s")
EPOCHS = int(os.environ.get("ABL_EPOCHS", "100"))
N_SYN = int(os.environ.get("ABL_NSYN", "500"))
CUT_PER_CLASS = int(os.environ.get("ABL_CUTS", "10"))
NAMES_BLOCK = f"  0: {PART}"

# (조건명, 영상 stem, 사용할 장수(None=전량))
CONDS = [
    ("g1_206", "Gearbox_gearbox1", None),
    ("train1_206", "train1", 206),
    ("train1_292", "train1", None),
]


def frames_of(stem, cap=None):
    """그 영상의 (이미지, 라벨) 목록. cap 이 있으면 균등 간격으로 줄인다."""
    pairs = []
    for ip in sorted(glob.glob(f"{STORE}/images/{stem}/*.jpg")):
        lp = f"{STORE}/labels/{stem}_{Path(ip).stem}.txt"
        if os.path.exists(lp):
            pairs.append((ip, lp))
    if cap and len(pairs) > cap:
        step = len(pairs) / cap
        pairs = [pairs[int(i * step)] for i in range(cap)]
    return pairs


def brightness(files, n=30):
    import cv2
    fs = files[::max(1, len(files) // n)][:n]
    vals = [float(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY).mean()) for f in fs]
    return round(st.mean(vals), 1) if vals else None


def build_dataset(dsdir, pairs):
    di, dl = Path(dsdir) / "images", Path(dsdir) / "labels"
    di.mkdir(parents=True, exist_ok=True); dl.mkdir(parents=True, exist_ok=True)
    for ip, lp in pairs:
        stem = Path(ip).stem if "__" in Path(ip).stem else f"{Path(lp).stem}"
        shutil.copy(ip, di / f"{stem}.jpg")
        shutil.copy(lp, dl / f"{stem}.txt")
    return di, dl


def build_gt(rundir):
    gi, gl = f"{rundir}/_gt/images", f"{rundir}/_gt/labels"
    os.makedirs(gi, exist_ok=True); os.makedirs(gl, exist_ok=True)
    n = 0
    for ip in sorted(glob.glob(GT_SRC + "/images/*.*")):
        ext = Path(ip).suffix.lower()
        if ext not in (".jpg", ".jpeg", ".png"):
            continue
        base = Path(ip).stem
        dst = f"{gi}/{base}{ext}"
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(ip), dst)
        shutil.copy(f"{GT_SRC}/labels/{base}.txt", f"{gl}/{base}.txt")
        n += 1
    y = f"{rundir}/_gt/gt.yaml"
    Path(y).write_text(f"path: {rundir}/_gt\ntrain:\n  - images\nval:\n  - images\nnames:\n{NAMES_BLOCK}\n",
                       encoding="utf-8")
    return y, n


def train_eval(images_dir, tag, rundir, gt_yaml, epochs):
    from ultralytics import YOLO
    import gc
    import torch
    y = f"{rundir}/data_{tag}.yaml"
    Path(y).write_text(f"train: {os.path.abspath(str(images_dir))}\nval: {os.path.abspath(str(images_dir))}\n"
                       f"names:\n{NAMES_BLOCK}\n", encoding="utf-8")
    t = time.time()
    m = YOLO(MODEL + ".pt")
    m.train(data=y, epochs=epochs, imgsz=640, batch=8, device=0,      # CUDA_VISIBLE_DEVICES 로 GPU 선택
            project=f"{rundir}/runs", name=tag, exist_ok=True, verbose=False, plots=False)
    det = YOLO(str(m.trainer.best))
    r = det.val(data=gt_yaml, imgsz=640, device=0, verbose=False)
    out = {"gt_map50": round(float(r.box.map50), 4), "gt_map5095": round(float(r.box.map), 4),
           "precision": round(float(r.box.mp), 4), "recall": round(float(r.box.mr), 4),
           "min": round((time.time() - t) / 60, 1)}
    del m, det; gc.collect(); torch.cuda.empty_cache()
    return out


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    rundir = f"{OUT}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(rundir, exist_ok=True)

    def log(s):
        print(s, flush=True)
        with open(rundir + "/run.log", "a", encoding="utf-8") as f:
            f.write(s + "\n")

    smoke = bool(os.environ.get("ABL_SMOKE"))
    eps = 3 if smoke else EPOCHS
    n_syn = 8 if smoke else N_SYN

    gt_yaml, n_gt = build_gt(rundir)
    gt_bright = brightness(sorted(glob.glob(GT_SRC + "/images/*.jpg")))
    log(f"gearbox 조명·개수 분리 실험 · GPU {gpu} · {MODEL} {eps}ep · 합성 {n_syn}장 · 누끼 {CUT_PER_CLASS}장/부품")
    log(f"  GT {n_gt}장 (평균 밝기 {gt_bright})")

    results = []

    def save():
        json.dump({"gpu": gpu, "model": MODEL, "epochs": eps, "n_syn": n_syn, "cuts": CUT_PER_CLASS,
                   "gt": {"n": n_gt, "brightness": gt_bright}, "rows": results},
                  open(rundir + "/results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    for name, stem, cap in CONDS:
        pairs = frames_of(stem, cap)
        if len(pairs) < 20:
            log(f"  [{name}] 프레임 부족({len(pairs)}) — 건너뜀"); continue
        bright = brightness([p[0] for p in pairs])
        log(f"  [{name}] {stem} {len(pairs)}장 · 평균 밝기 {bright} (GT 대비 {bright - gt_bright:+.1f})")
        for synth in (False, True):
            tag = f"{name}_{'synth' if synth else 'raw'}"
            dsdir = f"{rundir}/_ds_{tag}"
            try:
                di, dl = build_dataset(dsdir, pairs)
                made = 0
                if synth:
                    sa.CUT_PER_CLASS = CUT_PER_CLASS
                    made = sa._synth_augment(di, dl, lambda s, lvl="info": None, n_syn=n_syn)
                r = train_eval(di, tag, rundir, gt_yaml, eps)
                r.update(cond=name, stem=stem, n_raw=len(pairs), brightness=bright,
                         synth=synth, n_synth=made, n_train=len(pairs) + made, tag=tag)
                results.append(r)
                log(f"    {'합성' if synth else '원본만'}: 학습 {r['n_train']}장 -> mAP50 {r['gt_map50']} "
                    f"mAP50-95 {r['gt_map5095']} P {r['precision']} R {r['recall']} ({r['min']}분)")
            except Exception as e:
                log(f"    {'합성' if synth else '원본만'} 실패: {type(e).__name__}: {e}")
            save()
            shutil.rmtree(dsdir, ignore_errors=True)

    lines = [f"gearbox 조명 vs 개수 (GPU {gpu} · {MODEL} {eps}ep · GT {n_gt}장 밝기 {gt_bright})",
             f"{'조건':14}{'영상':18}{'장수':>5}{'밝기':>6}{'합성':>6}{'학습':>6}{'mAP50':>9}{'50-95':>8}{'P':>7}{'R':>7}"]
    for r in results:
        lines.append(f"{r['cond']:14}{r['stem']:18}{r['n_raw']:>5}{r['brightness']:>6.0f}"
                     f"{r['n_synth']:>6}{r['n_train']:>6}{r['gt_map50']:>9.4f}{r['gt_map5095']:>8.4f}"
                     f"{r['precision']:>7.3f}{r['recall']:>7.3f}")
    lines.append("\n해석 기준")
    lines.append("  g1_206 vs train1_206 : 개수 같고 조명만 다름 -> 차이는 조명 효과")
    lines.append("  train1_206 vs train1_292 : 조명 같고 개수만 다름 -> 차이는 데이터 개수 효과")
    Path(rundir + "/summary.txt").write_text("\n".join(lines), encoding="utf-8")
    log("DONE -> " + rundir + "/summary.txt")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
