# -*- coding: utf-8 -*-
"""a_test ablation: 모델·라벨수·에포크 축을 각각 독립으로(서로 안 곱함) 학습 → 실측 GT mAP 측정.
기본값 고정: yolo11s · 100ep · 전체 라벨. 각 축은 한 변수만 변경.
사용: XR_BASE=$HOME/xr_autolearning CUDA_VISIBLE_DEVICES=3 python scripts/experiments/ablation_atest.py
결과: results/ablation/a_test/<시각>/results.json · summary.txt · run.log
"""
import os, sys, glob, json, time
from datetime import datetime

BASE = os.environ.get("XR_BASE") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, BASE + "/scripts/experiments")
import build_multiclass as bm

# PART/GT는 환경변수로 바꿔 다른 부품에도 재사용(기본=a_test). OUT은 부품명으로 분리.
PART = os.environ.get("ABL_PART", "a_test")
GT_YAML = os.environ.get("ABL_GT", BASE + "/data/bell412/" + PART + "/gt/gt.yaml")
OUT = BASE + "/results/ablation/" + PART
MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolo11n", "yolo11s", "yolo11m", "yolo26n", "yolo26s", "yolo26m"]
DEF_MODEL, DEF_EP = "yolo11s", 100


def collect_frames():
    fr = []
    for ip in sorted(glob.glob(BASE + "/results/parts/*/train/images/*.jpg")):
        if bm.stem_to_class(os.path.splitext(os.path.basename(ip))[0]) == PART:
            fr.append(os.path.abspath(ip))
    return fr


def subsample(frames, m):
    n = len(frames)
    if m >= n:
        return frames
    return [frames[int(i * (n - 1) / (m - 1))] for i in range(m)]   # 균등 간격


def make_yaml(frames, tag, rundir):
    d = rundir + f"/data_{tag}"; os.makedirs(d, exist_ok=True)
    lst = d + "/train.txt"; open(lst, "w").write("\n".join(frames) + "\n")
    y = d + "/data.yaml"
    open(y, "w", encoding="utf-8").write(f"train: {os.path.abspath(lst)}\nval: {os.path.abspath(lst)}\nnames:\n  0: {PART}\n")
    return y


def train_eval(frames, model, epochs, tag, rundir):
    from ultralytics import YOLO
    import torch, gc
    y = make_yaml(frames, tag, rundir)
    t = time.time()
    m = YOLO(model if model.endswith(".pt") else model + ".pt")
    m.train(data=y, epochs=epochs, imgsz=640, batch=8, device=0,
            project=rundir + "/runs", name=tag, exist_ok=True, verbose=False, plots=False)
    best = m.trainer.best
    det = YOLO(str(best))
    r = det.val(data=GT_YAML, imgsz=640, device=0, verbose=False)
    res = {"tag": tag, "model": model, "epochs": epochs, "n_labels": len(frames),
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

    log(f"{PART} train 프레임 {len(frames)}장 · GT {GT_YAML}")
    if len(frames) < 5:
        log("프레임 부족 — 중단"); return
    results = {"model": [], "labels": [], "epochs": []}
    DEF_N = len(frames)

    def save():
        json.dump(results, open(rundir + "/results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    for md in MODELS:                                   # ① 모델 축(전체 라벨·100ep) = 실측 GT 모델 재선정
        try:
            r = train_eval(frames, md, DEF_EP, f"model_{md}", rundir); results["model"].append(r)
            log(f"[모델] {md}: GT mAP50 {r['gt_map50']} / mAP50-95 {r['gt_map5095']} ({r['min']}분)")
        except Exception as e:
            log(f"[모델] {md} 실패: {type(e).__name__}: {e}")
        save()
    for m in [20, 50, 100, 200, DEF_N]:                 # ② 라벨수 축(yolo11s·100ep)
        try:
            r = train_eval(subsample(frames, m), DEF_MODEL, DEF_EP, f"labels_{m}", rundir); results["labels"].append(r)
            log(f"[라벨수] {m}: GT mAP50 {r['gt_map50']} / mAP50-95 {r['gt_map5095']}")
        except Exception as e:
            log(f"[라벨수] {m} 실패: {type(e).__name__}: {e}")
        save()
    for ep in [30, 60, 100, 200]:                       # ③ 에포크 축(yolo11s·전체 라벨)
        try:
            r = train_eval(frames, DEF_MODEL, ep, f"epoch_{ep}", rundir); results["epochs"].append(r)
            log(f"[에포크] {ep}: GT mAP50 {r['gt_map50']} / mAP50-95 {r['gt_map5095']}")
        except Exception as e:
            log(f"[에포크] {ep} 실패: {type(e).__name__}: {e}")
        save()

    lines = [f"{PART} 단독(단일클래스) ablation — 실측 GT mAP"]
    for axis, rows in results.items():
        lines.append(f"\n== {axis} ==")
        for r in rows:
            lines.append(f"  {r['tag']:16} mAP50 {r['gt_map50']:.4f}  mAP50-95 {r['gt_map5095']:.4f}  (n={r['n_labels']}, {r['epochs']}ep, {r['min']}분)")
    open(rundir + "/summary.txt", "w", encoding="utf-8").write("\n".join(lines))
    log("DONE → " + rundir + "/summary.txt")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
