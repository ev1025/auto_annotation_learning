# -*- coding: utf-8 -*-
"""누끼(합성용 오려낸 조각) 개수에 따른 성능 ablation — gearbox+a_test 2클래스, 실측 GT mAP.

배경 합성 증강에서 '부품당 누끼 몇 장을 쓰는지'만 바꾸고 나머지는 고정한다.
합성 이미지 장수(N_SYN)는 조건마다 같게 두어, 같은 500장을 몇 종류의 조각으로
만들었는지의 차이만 본다.

축: ABL_CUT_NS (쉼표 구분). 0 = 합성 없음(baseline), 9999 = 전체(무제한)
고정: yolo11s · 100 epochs · imgsz 640 · batch 8 · N_SYN=500 · GT 평가(a_test 30 + gearbox 40)

사용(서버, GPU 하나당 프로세스 하나):
  cd /workspace/data2/jinwoolee/xr_autolearning
  XR_BASE=$PWD CUDA_VISIBLE_DEVICES=1 ABL_CUT_NS=0,10,40 \
    ./venv/bin/python scripts/experiments/ablation_cuts.py
  XR_BASE=$PWD CUDA_VISIBLE_DEVICES=2 ABL_CUT_NS=20,60,9999 \
    ./venv/bin/python scripts/experiments/ablation_cuts.py
  # 스모크: ABL_SMOKE=1 (3에포크, 합성 8장)
결과: results/experiments/ablation_cuts/<시각>_gpu<N>/{results.json,summary.txt,run.log}
"""
import glob
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

BASE = os.environ.get("XR_BASE") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, BASE + "/scripts/experiments")
sys.path.insert(0, BASE + "/backend/autolearning")
import sam2_autolabel as sa            # _synth_augment · CUT_PER_CLASS

CLASSES = ["gearbox", "a_test"]        # 인덱스 = 순서(gearbox=0, a_test=1)
IDX = {c: i for i, c in enumerate(CLASSES)}
GT_SRC = {c: f"{BASE}/data/bell412/{c}/gt" for c in CLASSES}
NAMES_BLOCK = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))

N_SYN = int(os.environ.get("ABL_NSYN", "500"))     # 합성 장수(조건 간 고정)
MODEL = os.environ.get("ABL_MODEL", "yolo11s")
DEF_EP = int(os.environ.get("ABL_EPOCHS", "100"))
CUT_NS = [int(x) for x in os.environ.get("ABL_CUT_NS", "0,10,20,40,60,9999").split(",") if x.strip()]
OUT = BASE + "/results/experiments/ablation_cuts"

_VID2PART = None


def _vid2part():
    """영상 stem -> 부품(폴더명). data/bell412/<부품>/videos/<영상>."""
    global _VID2PART
    if _VID2PART is None:
        m = {}
        for vp in glob.glob(BASE + "/data/bell412/*/videos/*.*"):
            m.setdefault(os.path.splitext(os.path.basename(vp))[0],
                         os.path.basename(os.path.dirname(os.path.dirname(vp))))
        _VID2PART = m
    return _VID2PART


def stem_to_class(stem):
    v = re.sub(r"_\d+$", "", stem).replace("_TEST", "")
    vm = _vid2part()
    if v in vm:
        return re.sub(r"\s+", "_", vm[v].lower())
    part = v.split("_", 1)[1] if "_" in v else v
    return re.sub(r"\s+", "_", re.sub(r"\d+$", "", part).strip().lower())


def collect_frames():
    """클래스별 raw 학습 프레임. {cls: [img_path,...]}"""
    per = {c: [] for c in CLASSES}
    for ip in sorted(glob.glob(BASE + "/results/parts/*/train/images/*.jpg")):
        cls = stem_to_class(Path(ip).stem)
        if cls in per:
            per[cls].append(os.path.abspath(ip))
    return per


def _remap_label(src_txt, idx):
    """단일클래스(0) 라벨 -> 첫 필드를 idx 로 교체."""
    out = []
    if os.path.exists(src_txt):
        for line in open(src_txt, encoding="utf-8"):
            p = line.split()
            if len(p) == 5:
                out.append(f"{idx} {p[1]} {p[2]} {p[3]} {p[4]}")
    return out


def build_dataset(dsdir, frames):
    """조건마다 새 학습셋을 만든다(합성 파일이 섞이지 않게 조건별 폴더)."""
    di, dl = Path(dsdir) / "images", Path(dsdir) / "labels"
    di.mkdir(parents=True, exist_ok=True); dl.mkdir(parents=True, exist_ok=True)
    counts = {c: 0 for c in CLASSES}
    for cls, ips in frames.items():
        for ip in ips:
            stem = f"{cls}__{Path(ip).stem}"
            shutil.copy(ip, di / f"{stem}.jpg")
            lines = _remap_label(ip.replace("/images/", "/labels/")[:-4] + ".txt", IDX[cls])
            (dl / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            counts[cls] += 1
    return di, dl, counts


def build_gt(rundir):
    """2클래스 통합 GT + gt.yaml. 조건 전체가 같은 평가셋을 쓴다."""
    gi, gl = rundir + "/_gt/images", rundir + "/_gt/labels"
    os.makedirs(gi, exist_ok=True); os.makedirs(gl, exist_ok=True)
    n = 0
    for cls in CLASSES:
        src = GT_SRC[cls]
        for ip in sorted(glob.glob(src + "/images/*.*")):
            ext = os.path.splitext(ip)[1].lower()
            if ext not in (".jpg", ".jpeg", ".png"):
                continue
            base = os.path.splitext(os.path.basename(ip))[0]
            stem = f"{cls}__{base}"
            gip = f"{gi}/{stem}{ext}"
            if not os.path.exists(gip):
                os.symlink(os.path.abspath(ip), gip)
            (Path(gl) / f"{stem}.txt").write_text(
                "\n".join(_remap_label(f"{src}/labels/{base}.txt", IDX[cls])) + "\n", encoding="utf-8")
            n += 1
    y = rundir + "/_gt/gt.yaml"
    Path(y).write_text(f"path: {rundir}/_gt\ntrain:\n  - images\nval:\n  - images\nnames:\n{NAMES_BLOCK}\n",
                       encoding="utf-8")
    return y, n


def train_eval(images_dir, tag, rundir, gt_yaml, epochs):
    from ultralytics import YOLO
    import gc
    import torch
    y = str(Path(rundir) / f"data_{tag}.yaml")
    Path(y).write_text(f"train: {os.path.abspath(str(images_dir))}\nval: {os.path.abspath(str(images_dir))}\n"
                       f"names:\n{NAMES_BLOCK}\n", encoding="utf-8")
    t = time.time()
    m = YOLO(MODEL + ".pt")
    # CUDA_VISIBLE_DEVICES 로 GPU 를 고르므로 여기서는 항상 device=0
    m.train(data=y, epochs=epochs, imgsz=640, batch=8, device=0,
            project=rundir + "/runs", name=tag, exist_ok=True, verbose=False, plots=False)
    det = YOLO(str(m.trainer.best))
    r = det.val(data=gt_yaml, imgsz=640, device=0, verbose=False)
    per = {}
    try:
        for i, ci in enumerate(r.box.ap_class_index):
            per[CLASSES[int(ci)]] = round(float(r.box.ap50[i]), 4)
    except Exception:
        pass
    out = {"tag": tag, "gt_map50": round(float(r.box.map50), 4), "gt_map5095": round(float(r.box.map), 4),
           "per_class_map50": per, "precision": round(float(r.box.mp), 4), "recall": round(float(r.box.mr), 4),
           "min": round((time.time() - t) / 60, 1)}
    del m, det; gc.collect(); torch.cuda.empty_cache()
    return out


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rundir = f"{OUT}/{ts}_gpu{gpu}"
    os.makedirs(rundir, exist_ok=True)

    def log(s):
        print(s, flush=True)
        with open(rundir + "/run.log", "a", encoding="utf-8") as f:
            f.write(s + "\n")

    smoke = bool(os.environ.get("ABL_SMOKE"))
    eps = 3 if smoke else DEF_EP
    n_syn = 8 if smoke else N_SYN

    frames = collect_frames()
    gt_yaml, n_gt = build_gt(rundir)
    ng, na = len(frames["gearbox"]), len(frames["a_test"])
    log(f"누끼 개수 ablation · GPU {gpu} · {MODEL} {eps}ep · 합성 {n_syn}장 고정")
    log(f"  raw gearbox {ng} + a_test {na} = {ng + na}장 · GT {n_gt}장 · 조건 {CUT_NS}")
    if ng < 5 or na < 5:
        log("프레임 부족 — 중단"); return

    results = []

    def save():
        json.dump({"gpu": gpu, "model": MODEL, "epochs": eps, "n_syn": n_syn,
                   "raw": {"gearbox": ng, "a_test": na}, "gt": n_gt, "rows": results},
                  open(rundir + "/results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    for n in CUT_NS:
        tag = "nosynth" if n == 0 else ("cut_all" if n >= 9999 else f"cut{n}")
        dsdir = f"{rundir}/_ds_{tag}"
        try:
            di, dl, _ = build_dataset(dsdir, frames)
            made = 0
            if n > 0:
                sa.CUT_PER_CLASS = 10 ** 6 if n >= 9999 else n   # 조건 = 부품당 누끼 상한
                made = sa._synth_augment(di, dl, lambda s, lvl="info": log(f"    [synth] {s}"), n_syn=n_syn)
                if made == 0:
                    log(f"  [{tag}] 합성 0장(배경 없음?) — raw 만으로 학습")
            r = train_eval(di, tag, rundir, gt_yaml, eps)
            r.update(cut_per_class=(0 if n == 0 else n), n_synth=made, n_train=ng + na + made)
            results.append(r)
            pc = r["per_class_map50"]
            log(f"  [{tag}] 합성 {made}장 · 학습 {r['n_train']}장 -> mAP50 {r['gt_map50']} "
                f"(gearbox {pc.get('gearbox','-')} / a_test {pc.get('a_test','-')}) "
                f"mAP50-95 {r['gt_map5095']} P {r['precision']} R {r['recall']} ({r['min']}분)")
        except Exception as e:
            log(f"  [{tag}] 실패: {type(e).__name__}: {e}")
        save()
        shutil.rmtree(dsdir, ignore_errors=True)   # 조건마다 수백 MB — 지표만 남기고 정리

    lines = [f"누끼 개수 ablation (GPU {gpu} · {MODEL} {eps}ep · 합성 {n_syn}장 고정 · GT {n_gt}장)",
             f"{'조건':10} {'누끼/부품':>9} {'합성':>6} {'학습장수':>8} {'mAP50':>8} {'mAP50-95':>9} {'gearbox':>8} {'a_test':>8} {'분':>6}"]
    for r in results:
        pc = r["per_class_map50"]
        lines.append(f"{r['tag']:10} {r['cut_per_class']:>9} {r['n_synth']:>6} {r['n_train']:>8} "
                     f"{r['gt_map50']:>8.4f} {r['gt_map5095']:>9.4f} "
                     f"{pc.get('gearbox', 0):>8.4f} {pc.get('a_test', 0):>8.4f} {r['min']:>6.1f}")
    Path(rundir + "/summary.txt").write_text("\n".join(lines), encoding="utf-8")
    log("DONE -> " + rundir + "/summary.txt")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
