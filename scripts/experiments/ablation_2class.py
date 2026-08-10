# -*- coding: utf-8 -*-
"""gearbox+a_test 2클래스 '함께 학습' ablation — 실측 GT mAP(전체 + 클래스별).

단일클래스 실험(ablation_atest.py)과 축(모델/라벨수/에포크)은 동일하되, 두 부품을 한 모델에
같이 넣어 학습한다. 클래스 혼동을 포함한 배포-현실값을 재는 것이 목적.
per-part 라벨은 단일클래스(0)로 저장돼 있어 gearbox→0, a_test→1 로 재매핑해 통합한다.
GT 도 두 부품 GT(각 class0)를 같은 규칙으로 재매핑해 2클래스 통합 평가셋을 만든다.

사용: XR_BASE=$HOME/xr_autolearning CUDA_VISIBLE_DEVICES=2 python scripts/experiments/ablation_2class.py
결과: results/ablation/gearbox_atest/<시각>/{results.json,summary.txt,run.log}
"""
import os, sys, glob, json, time
from datetime import datetime

BASE = os.environ.get("XR_BASE") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, BASE + "/scripts/experiments")
import build_multiclass as bm

CLASSES = ["gearbox", "a_test"]            # 인덱스 = 리스트 순서(gearbox=0, a_test=1)
IDX = {c: i for i, c in enumerate(CLASSES)}
GT_SRC = {                                  # 각 부품 수동 GT(단일 class0로 저장됨)
    "gearbox": BASE + "/data/bell412/_gearbox_domaingap/gt",
    "a_test":  BASE + "/data/bell412/a_test/gt",
}
OUT = BASE + "/results/ablation/gearbox_atest"
MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolo11n", "yolo11s", "yolo11m", "yolo26n", "yolo26s", "yolo26m"]
DEF_MODEL, DEF_EP = "yolo11s", 100
NAMES_BLOCK = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))


def collect_frames():
    """클래스별 학습 프레임 경로. {cls: [img_path,...]}"""
    per = {c: [] for c in CLASSES}
    for ip in sorted(glob.glob(BASE + "/results/parts/*/train/images/*.jpg")):
        cls = bm.stem_to_class(os.path.splitext(os.path.basename(ip))[0])
        if cls in per:
            per[cls].append(os.path.abspath(ip))
    return per


def _remap_label(src_txt, idx):
    """단일클래스(0) 라벨 파일을 읽어 첫 필드를 idx 로 바꾼 줄 리스트 반환."""
    out = []
    if os.path.exists(src_txt):
        for l in open(src_txt, encoding="utf-8"):
            p = l.split()
            if len(p) == 5:
                out.append(f"{idx} {p[1]} {p[2]} {p[3]} {p[4]}")
    return out


def build_dataset(rundir):
    """2클래스 통합 학습셋(이미지 심볼릭 + 라벨 재매핑)을 한 번 만든다.
    반환: {cls: [ds이미지경로,...]}"""
    di, dl = rundir + "/_ds/images", rundir + "/_ds/labels"
    os.makedirs(di, exist_ok=True); os.makedirs(dl, exist_ok=True)
    out = {c: [] for c in CLASSES}
    for cls, ips in collect_frames().items():
        for ip in ips:
            stem = cls + "__" + os.path.splitext(os.path.basename(ip))[0]
            dip = di + "/" + stem + ".jpg"
            if not os.path.exists(dip):
                os.symlink(ip, dip)
            lines = _remap_label(ip.replace("/images/", "/labels/")[:-4] + ".txt", IDX[cls])
            open(dl + "/" + stem + ".txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
            out[cls].append(dip)
    return out


def build_gt(rundir):
    """2클래스 통합 GT(이미지 심볼릭 + 라벨 재매핑) + gt.yaml 경로 반환."""
    gi, gl = rundir + "/_gt/images", rundir + "/_gt/labels"
    os.makedirs(gi, exist_ok=True); os.makedirs(gl, exist_ok=True)
    for cls in CLASSES:
        src = GT_SRC[cls]
        for ip in sorted(glob.glob(src + "/images/*.*")):
            ext = os.path.splitext(ip)[1].lower()
            if ext not in (".jpg", ".jpeg", ".png"):
                continue
            base = os.path.splitext(os.path.basename(ip))[0]
            stem = cls + "__" + base
            gip = gi + "/" + stem + ext
            if not os.path.exists(gip):
                os.symlink(os.path.abspath(ip), gip)
            lines = _remap_label(src + "/labels/" + base + ".txt", IDX[cls])
            open(gl + "/" + stem + ".txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    y = rundir + "/_gt/gt.yaml"
    open(y, "w", encoding="utf-8").write(f"path: {rundir}/_gt\ntrain:\n  - images\nval:\n  - images\nnames:\n{NAMES_BLOCK}\n")
    return y


def subsample(frames, m):
    n = len(frames)
    if m >= n:
        return list(frames)
    return [frames[int(i * (n - 1) / (m - 1))] for i in range(m)]   # 균등 간격


def make_yaml(img_paths, tag, rundir):
    d = rundir + f"/data_{tag}"; os.makedirs(d, exist_ok=True)
    lst = d + "/train.txt"; open(lst, "w", encoding="utf-8").write("\n".join(img_paths) + "\n")
    y = d + "/data.yaml"
    open(y, "w", encoding="utf-8").write(f"train: {os.path.abspath(lst)}\nval: {os.path.abspath(lst)}\nnames:\n{NAMES_BLOCK}\n")
    return y


def train_eval(img_paths, model, epochs, tag, rundir, gt_yaml):
    from ultralytics import YOLO
    import torch, gc
    y = make_yaml(img_paths, tag, rundir)
    t = time.time()
    m = YOLO(model if model.endswith(".pt") else model + ".pt")
    m.train(data=y, epochs=epochs, imgsz=640, batch=8, device=0,
            project=rundir + "/runs", name=tag, exist_ok=True, verbose=False, plots=False)
    best = m.trainer.best
    det = YOLO(str(best))
    r = det.val(data=gt_yaml, imgsz=640, device=0, verbose=False)
    per = {}
    try:                                        # 클래스별 AP@0.5
        for i, ci in enumerate(r.box.ap_class_index):
            per[CLASSES[int(ci)]] = round(float(r.box.ap50[i]), 4)
    except Exception:
        pass
    res = {"tag": tag, "model": model, "epochs": epochs, "n_imgs": len(img_paths),
           "gt_map50": round(float(r.box.map50), 4), "gt_map5095": round(float(r.box.map), 4),
           "per_class_map50": per, "min": round((time.time() - t) / 60, 1)}
    del m, det; gc.collect(); torch.cuda.empty_cache()
    return res


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rundir = OUT + f"/{ts}"; os.makedirs(rundir, exist_ok=True)

    def log(s):
        print(s, flush=True)
        with open(rundir + "/run.log", "a", encoding="utf-8") as f:
            f.write(s + "\n")

    per = build_dataset(rundir)
    gt_yaml = build_gt(rundir)
    ng, na = len(per["gearbox"]), len(per["a_test"])
    log(f"2클래스 함께 학습 — gearbox {ng} + a_test {na} 프레임 · GT {gt_yaml}")
    if ng < 5 or na < 5:
        log("프레임 부족 — 중단"); return
    full = per["gearbox"] + per["a_test"]
    results = {"model": [], "labels": [], "epochs": []}

    def save():
        json.dump(results, open(rundir + "/results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    for md in MODELS:                                   # ① 모델 축(전체·100ep)
        try:
            r = train_eval(full, md, DEF_EP, f"model_{md}", rundir, gt_yaml); results["model"].append(r)
            log(f"[모델] {md}: GT mAP50 {r['gt_map50']} 클래스별 {r['per_class_map50']} / mAP50-95 {r['gt_map5095']} ({r['min']}분)")
        except Exception as e:
            log(f"[모델] {md} 실패: {type(e).__name__}: {e}")
        save()
    for mc in [20, 50, 100, 200, max(ng, na)]:          # ② 라벨수 축(yolo11s·100ep) — 클래스당 mc장
        sub = subsample(per["gearbox"], mc) + subsample(per["a_test"], mc)
        try:
            r = train_eval(sub, DEF_MODEL, DEF_EP, f"labels_{mc}", rundir, gt_yaml); results["labels"].append(r)
            log(f"[라벨수] 클래스당~{mc}(총{len(sub)}): GT mAP50 {r['gt_map50']} 클래스별 {r['per_class_map50']}")
        except Exception as e:
            log(f"[라벨수] {mc} 실패: {type(e).__name__}: {e}")
        save()
    for ep in [30, 60, 100, 200]:                       # ③ 에포크 축(yolo11s·전체)
        try:
            r = train_eval(full, DEF_MODEL, ep, f"epoch_{ep}", rundir, gt_yaml); results["epochs"].append(r)
            log(f"[에포크] {ep}: GT mAP50 {r['gt_map50']} 클래스별 {r['per_class_map50']}")
        except Exception as e:
            log(f"[에포크] {ep} 실패: {type(e).__name__}: {e}")
        save()

    lines = ["gearbox+a_test 2클래스 함께 학습 — 실측 GT mAP (전체 + 클래스별 AP50)"]
    for axis, rows in results.items():
        lines.append(f"\n== {axis} ==")
        for r in rows:
            pc = r.get("per_class_map50", {})
            lines.append(f"  {r['tag']:16} mAP50 {r['gt_map50']:.4f}  gearbox {pc.get('gearbox','-')} a_test {pc.get('a_test','-')}  mAP50-95 {r['gt_map5095']:.4f}  ({r['n_imgs']}장,{r['epochs']}ep,{r['min']}분)")
    open(rundir + "/summary.txt", "w", encoding="utf-8").write("\n".join(lines))
    log("DONE → " + rundir + "/summary.txt")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
