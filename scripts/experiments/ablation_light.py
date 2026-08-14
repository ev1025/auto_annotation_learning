# -*- coding: utf-8 -*-
"""gearbox 성능 저하 원인 분리 실험 — 조명 정합인가, 학습 데이터 개수인가.

배경: gearbox GT(test1~4, 격납고·장착 도메인) 평균 밝기 101.
      학습 영상 Gearbox_gearbox1 은 151(밝은 실내), train1 은 108(GT 와 근접).
      최근 2클래스 ablation 에서 gearbox mAP50 이 0.43~0.53 이었는데 7/31 에는 0.997 이었다.
      그때는 train1+train2 493장(단독 학습)이었고 지금은 Gearbox_gearbox1 206장이다.
      -> 조명 차이와 데이터 개수가 섞여 있어 분리가 안 된다.

조건(GT 고정, 나머지 전부 고정):
  A. g1_206      Gearbox_gearbox1 206장   조명 불일치(151) · 개수 206
  B. train1_206  train1 에서 206장 균등   조명 일치(108)   · 개수 206   <- A 와 조명만 다름
  C. train1_292  train1 전량 292장        조명 일치(108)   · 개수 292   <- B 와 개수만 다름
각 조건을 합성 증강 ON/OFF 로 돌린다(총 6런).

모드 두 가지 — 둘 다 돌려 대조한다.
  기본(단독)      : gearbox 1클래스, GT 40장. 7/31(0.997) 과 같은 축
  ABL_2CLASS=1   : a_test 를 클래스 1 로 함께 학습, GT 70장. 앞선 누끼 실험과 같은 축

고정: yolo11s · 100 epochs · imgsz 640 · batch 8 · 합성 500장 · 부품당 누끼 10장(직전 실험 최적)

사용(서버, GPU2 고정):
  cd /workspace/data2/jinwoolee/xr_autolearning
  XR_BASE=$PWD CUDA_VISIBLE_DEVICES=2 ./venv/bin/python scripts/experiments/ablation_light.py
  XR_BASE=$PWD CUDA_VISIBLE_DEVICES=2 ABL_2CLASS=1 ./venv/bin/python scripts/experiments/ablation_light.py
결과: results/experiments/ablation_light/<시각>_{single|2class}/{results.json,summary.txt,run.log}
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

# ABL_2CLASS=1 이면 a_test 를 함께 학습한다(gearbox=0, a_test=1). GT 도 두 부품을 합친다.
TWO = bool(os.environ.get("ABL_2CLASS"))
CLASSES = ["gearbox", "a_test"] if TWO else ["gearbox"]
NAMES_BLOCK = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))

STORE = f"{BASE}/results/autolabels/gearbox"
ATEST_STORE = f"{BASE}/results/autolabels/a_test"
OUT = f"{BASE}/results/experiments/ablation_light"
MODEL = os.environ.get("ABL_MODEL", "yolo11s")
EPOCHS = int(os.environ.get("ABL_EPOCHS", "100"))
N_SYN = int(os.environ.get("ABL_NSYN", "500"))
CUT_PER_CLASS = int(os.environ.get("ABL_CUTS", "10"))

# (조건명, 영상 stem, 사용할 장수(None=전량))
CONDS = [
    ("g1_206", "Gearbox_gearbox1", None),
    ("train1_206", "train1", 206),
    ("train1_292", "train1", None),
]


def pairs_of(store, stem, cap=None):
    """그 영상의 (이미지, 라벨) 목록. cap 이 있으면 균등 간격으로 줄인다."""
    out = []
    for ip in sorted(glob.glob(f"{store}/images/{stem}/*.jpg")):
        lp = f"{store}/labels/{stem}_{Path(ip).stem}.txt"
        if os.path.exists(lp):
            out.append((ip, lp))
    if cap and len(out) > cap:
        step = len(out) / cap
        out = [out[int(i * step)] for i in range(cap)]
    return out


def atest_pairs():
    """a_test 학습 프레임 전량(2클래스 모드에서 고정 추가)."""
    out = []
    for ip in sorted(glob.glob(f"{ATEST_STORE}/images/*/*.jpg")):
        lp = f"{ATEST_STORE}/labels/{Path(ip).parent.name}_{Path(ip).stem}.txt"
        if os.path.exists(lp):
            out.append((ip, lp))
    return out


def brightness(files, n=30):
    import cv2
    fs = files[::max(1, len(files) // n)][:n]
    vals = [float(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY).mean()) for f in fs]
    return round(st.mean(vals), 1) if vals else 0.0


def remap(lp, idx):
    """단일클래스(0) 라벨 -> 첫 필드를 idx 로 바꾼 줄 목록."""
    out = []
    for line in open(lp, encoding="utf-8"):
        f = line.split()
        if len(f) == 5:
            out.append(f"{idx} {f[1]} {f[2]} {f[3]} {f[4]}")
    return out


def build_dataset(dsdir, gear_pairs):
    di, dl = Path(dsdir) / "images", Path(dsdir) / "labels"
    di.mkdir(parents=True, exist_ok=True); dl.mkdir(parents=True, exist_ok=True)
    for ip, lp in gear_pairs:
        stem = f"gearbox__{Path(lp).stem}"
        shutil.copy(ip, di / f"{stem}.jpg")
        (dl / f"{stem}.txt").write_text("\n".join(remap(lp, 0)) + "\n", encoding="utf-8")
    n_atest = 0
    if TWO:
        for ip, lp in atest_pairs():
            stem = f"a_test__{Path(lp).stem}"
            shutil.copy(ip, di / f"{stem}.jpg")
            (dl / f"{stem}.txt").write_text("\n".join(remap(lp, 1)) + "\n", encoding="utf-8")
            n_atest += 1
    return di, dl, n_atest


def build_gt(rundir):
    gi, gl = f"{rundir}/_gt/images", f"{rundir}/_gt/labels"
    os.makedirs(gi, exist_ok=True); os.makedirs(gl, exist_ok=True)
    n = 0
    for ci, cls in enumerate(CLASSES):
        src = f"{BASE}/data/bell412/{cls}/gt"
        for ip in sorted(glob.glob(src + "/images/*.*")):
            ext = Path(ip).suffix.lower()
            if ext not in (".jpg", ".jpeg", ".png"):
                continue
            base = Path(ip).stem
            stem = f"{cls}__{base}"
            dst = f"{gi}/{stem}{ext}"
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(ip), dst)
            (Path(gl) / f"{stem}.txt").write_text("\n".join(remap(f"{src}/labels/{base}.txt", ci)) + "\n",
                                                 encoding="utf-8")
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
    per = {}
    try:
        for i, ci in enumerate(r.box.ap_class_index):
            per[CLASSES[int(ci)]] = round(float(r.box.ap50[i]), 4)
    except Exception:
        pass
    out = {"gt_map50": round(float(r.box.map50), 4), "gt_map5095": round(float(r.box.map), 4),
           "per_class_map50": per,
           "precision": round(float(r.box.mp), 4), "recall": round(float(r.box.mr), 4),
           "min": round((time.time() - t) / 60, 1)}
    del m, det; gc.collect(); torch.cuda.empty_cache()
    return out


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    mode = "2class" if TWO else "single"
    rundir = f"{OUT}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{mode}"
    os.makedirs(rundir, exist_ok=True)

    def log(s):
        print(s, flush=True)
        with open(rundir + "/run.log", "a", encoding="utf-8") as f:
            f.write(s + "\n")

    smoke = bool(os.environ.get("ABL_SMOKE"))
    eps = 3 if smoke else EPOCHS
    n_syn = 8 if smoke else N_SYN

    gt_yaml, n_gt = build_gt(rundir)
    gt_bright = brightness(sorted(glob.glob(f"{BASE}/data/bell412/gearbox/gt/images/*.jpg")))
    n_atest_all = len(atest_pairs()) if TWO else 0
    log(f"gearbox 조명·개수 분리 실험 · {'2클래스(+a_test)' if TWO else 'gearbox 단독'} · GPU {gpu} · "
        f"{MODEL} {eps}ep · 합성 {n_syn}장 · 누끼 {CUT_PER_CLASS}장/부품")
    log(f"  GT {n_gt}장 (gearbox GT 평균 밝기 {gt_bright})" + (f" · a_test 학습 {n_atest_all}장 고정" if TWO else ""))

    results = []

    def save():
        json.dump({"mode": mode, "gpu": gpu, "model": MODEL, "epochs": eps, "n_syn": n_syn,
                   "cuts": CUT_PER_CLASS, "gt": {"n": n_gt, "gearbox_brightness": gt_bright},
                   "rows": results},
                  open(rundir + "/results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    for name, stem, cap in CONDS:
        gp = pairs_of(STORE, stem, cap)
        if len(gp) < 20:
            log(f"  [{name}] 프레임 부족({len(gp)}) — 건너뜀"); continue
        bright = brightness([p[0] for p in gp])
        log(f"  [{name}] {stem} {len(gp)}장 · 평균 밝기 {bright} (GT 대비 {bright - gt_bright:+.1f})")
        for synth in (False, True):
            tag = f"{name}_{'synth' if synth else 'raw'}"
            dsdir = f"{rundir}/_ds_{tag}"
            try:
                di, dl, n_at = build_dataset(dsdir, gp)
                made = 0
                if synth:
                    sa.CUT_PER_CLASS = CUT_PER_CLASS
                    made = sa._synth_augment(di, dl, lambda s, lvl="info": None, n_syn=n_syn)
                r = train_eval(di, tag, rundir, gt_yaml, eps)
                r.update(cond=name, stem=stem, n_gear=len(gp), n_atest=n_at, brightness=bright,
                         synth=synth, n_synth=made, n_train=len(gp) + n_at + made, tag=tag)
                results.append(r)
                pc = r["per_class_map50"]
                log(f"    {'합성' if synth else '원본만'}: 학습 {r['n_train']}장 -> mAP50 {r['gt_map50']} "
                    + (f"(gearbox {pc.get('gearbox', '-')} / a_test {pc.get('a_test', '-')}) " if TWO else "")
                    + f"mAP50-95 {r['gt_map5095']} P {r['precision']} R {r['recall']} ({r['min']}분)")
            except Exception as e:
                log(f"    {'합성' if synth else '원본만'} 실패: {type(e).__name__}: {e}")
            save()
            shutil.rmtree(dsdir, ignore_errors=True)   # 조건마다 수백 MB — 지표만 남긴다

    head = (f"{'조건':14}{'영상':18}{'gearbox':>8}{'밝기':>6}{'합성':>6}{'학습':>6}"
            f"{'mAP50':>9}{'50-95':>8}{'P':>7}{'R':>7}")
    if TWO:
        head += f"{'g_mAP50':>9}{'a_mAP50':>9}"
    lines = [f"gearbox 조명 vs 개수 ({'2클래스(+a_test)' if TWO else 'gearbox 단독'} · GPU {gpu} · "
             f"{MODEL} {eps}ep · GT {n_gt}장)", head]
    for r in results:
        pc = r.get("per_class_map50", {})
        row = (f"{r['cond']:14}{r['stem']:18}{r['n_gear']:>8}{r['brightness']:>6.0f}"
               f"{r['n_synth']:>6}{r['n_train']:>6}{r['gt_map50']:>9.4f}{r['gt_map5095']:>8.4f}"
               f"{r['precision']:>7.3f}{r['recall']:>7.3f}")
        if TWO:
            row += f"{pc.get('gearbox', 0):>9.4f}{pc.get('a_test', 0):>9.4f}"
        lines.append(row)
    lines += ["", "해석 기준",
              "  g1_206 vs train1_206     : 개수 같고 조명만 다름 -> 차이는 조명 효과",
              "  train1_206 vs train1_292 : 조명 같고 개수만 다름 -> 차이는 데이터 개수 효과",
              "  각 조건 원본만 vs 합성    : 배경 합성 증강 효과"]
    Path(rundir + "/summary.txt").write_text("\n".join(lines), encoding="utf-8")
    log("DONE -> " + rundir + "/summary.txt")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
