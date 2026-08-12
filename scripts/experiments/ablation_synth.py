# -*- coding: utf-8 -*-
"""합성(copy-paste 배경합성) 적용 단일클래스 ablation — 도메인갭 부품(gearbox 등) 공정 재측정.

ablation_atest.py 와 동일 구조지만, 학습 전 sam2_autolabel._synth_augment 로 실배경 합성 N장을
추가한 뒤 학습 → 실측 GT mAP 평가. 축은 모델 9종(전체 라벨·100ep 고정)으로 '합성 효과'만 격리한다.
(원본 ablation 은 합성 미적용 = raw SAM2 프레임만 학습이라, 도메인갭 부품이 과소평가됨. 그 보정 실험.)

사용:
  XR_BASE=$HOME/xr_autolearning CUDA_VISIBLE_DEVICES=2 ABL_PART=gearbox \
    ABL_GT=$HOME/xr_autolearning/data/bell412/gearbox/gt/gt.yaml \
    python scripts/experiments/ablation_synth.py
  # 스모크(1모델·3ep·합성 8장): 위에 ABL_SMOKE=1 ABL_NSYN=8 추가
결과: results/experiments/ablation_synth/<part>/<시각>/results.json · summary.txt · run.log
"""
import os, sys, glob, json, time, shutil
from pathlib import Path
from datetime import datetime

BASE = os.environ.get("XR_BASE") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, BASE + "/scripts/experiments")
sys.path.insert(0, BASE + "/backend/autolearning")
import build_multiclass as bm          # stem_to_class
import sam2_autolabel as sa            # _synth_augment (SAM2 누끼 + 배경합성)

PART = os.environ.get("ABL_PART", "gearbox")
GT_YAML = os.environ.get("ABL_GT", BASE + "/data/bell412/" + PART + "/gt/gt.yaml")
N_SYN = int(os.environ.get("ABL_NSYN", "400"))
OUT = BASE + "/results/experiments/ablation_synth/" + PART
MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolo11n", "yolo11s", "yolo11m", "yolo26n", "yolo26s", "yolo26m"]
DEF_EP = 100


def collect_frames():
    """PART 에 해당하는 raw SAM2 학습 프레임 경로(원본 ablation 과 동일 소스)."""
    fr = []
    for ip in sorted(glob.glob(BASE + "/results/parts/*/train/images/*.jpg")):
        if bm.stem_to_class(os.path.splitext(os.path.basename(ip))[0]) == PART:
            fr.append(os.path.abspath(ip))
    return fr


def build_train_dir(frames, rundir, logln):
    """raw 프레임+라벨을 작업 dir 로 복사한 뒤 _synth_augment 로 합성 N장 추가.
    반환: (images_dir(Path), n_raw, n_syn)."""
    oi = Path(rundir) / "train_src" / "images"
    ol = Path(rundir) / "train_src" / "labels"
    oi.mkdir(parents=True, exist_ok=True)
    ol.mkdir(parents=True, exist_ok=True)
    n_raw = 0
    for ip in frames:
        stem = Path(ip).stem
        lp = Path(ip).parent.parent / "labels" / f"{stem}.txt"   # train/images/x.jpg -> train/labels/x.txt
        if not lp.exists():
            continue
        shutil.copy(ip, oi / Path(ip).name)
        shutil.copy(lp, ol / f"{stem}.txt")
        n_raw += 1
    made = sa._synth_augment(oi, ol, logln, n_syn=N_SYN)   # oi/ol 에 syn_*.jpg/.txt 추가(클래스 idx 유지)
    return oi, n_raw, made


def make_yaml(images_dir, rundir):
    y = str(Path(rundir) / "data.yaml")
    Path(y).write_text(
        f"train: {os.path.abspath(str(images_dir))}\nval: {os.path.abspath(str(images_dir))}\nnames:\n  0: {PART}\n",
        encoding="utf-8")
    return y


def train_eval(images_dir, model, epochs, tag, rundir):
    from ultralytics import YOLO
    import torch, gc
    y = make_yaml(images_dir, rundir)
    t = time.time()
    m = YOLO(model if model.endswith(".pt") else model + ".pt")
    m.train(data=y, epochs=epochs, imgsz=640, batch=8, device=0,   # CUDA_VISIBLE_DEVICES=2 → 물리 GPU2
            project=rundir + "/runs", name=tag, exist_ok=True, verbose=False, plots=False)
    det = YOLO(str(m.trainer.best))
    r = det.val(data=GT_YAML, imgsz=640, device=0, verbose=False)
    res = {"tag": tag, "model": model, "epochs": epochs,
           "gt_map50": round(float(r.box.map50), 4), "gt_map5095": round(float(r.box.map), 4),
           "min": round((time.time() - t) / 60, 1)}
    del m, det; gc.collect(); torch.cuda.empty_cache()
    return res


def main():
    frames = collect_frames()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rundir = OUT + f"/{ts}"; os.makedirs(rundir, exist_ok=True)

    def log(s):
        print(s, flush=True)
        with open(rundir + "/run.log", "a", encoding="utf-8") as f:
            f.write(s + "\n")

    def logln(s, lvl="info"):
        log("  [synth] " + s)

    log(f"{PART} 합성적용 ablation · raw 프레임 {len(frames)}장 · GT {GT_YAML}")
    if len(frames) < 5:
        log("프레임 부족 — 중단"); return

    images_dir, n_raw, n_syn = build_train_dir(frames, rundir, logln)
    log(f"학습셋: raw {n_raw} + 합성 {n_syn} = {n_raw + n_syn}장")
    if n_syn == 0:
        log("경고: 합성 0장(배경 없음?) — raw 만으로 학습됨")

    smoke = bool(os.environ.get("ABL_SMOKE"))
    models = ["yolo26n"] if smoke else MODELS
    eps = 3 if smoke else DEF_EP

    results = {"model": []}

    def save():
        json.dump(results, open(rundir + "/results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    for md in models:                         # 모델 축(전체 라벨+합성·100ep)
        try:
            r = train_eval(images_dir, md, eps, f"model_{md}", rundir); results["model"].append(r)
            log(f"[모델] {md}: GT mAP50 {r['gt_map50']} / mAP50-95 {r['gt_map5095']} ({r['min']}분)")
        except Exception as e:
            log(f"[모델] {md} 실패: {type(e).__name__}: {e}")
        save()

    lines = [f"{PART} 단독 + 배경합성 ablation — 실측 GT mAP (raw {n_raw} + 합성 {n_syn} = {n_raw + n_syn}장)"]
    lines.append("\n== model ==")
    for r in results["model"]:
        lines.append(f"  {r['tag']:16} mAP50 {r['gt_map50']:.4f}  mAP50-95 {r['gt_map5095']:.4f}  ({r['min']}분)")
    open(rundir + "/summary.txt", "w", encoding="utf-8").write("\n".join(lines))
    log("DONE → " + rundir + "/summary.txt")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
