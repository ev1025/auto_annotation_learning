# -*- coding: utf-8 -*-
"""gearbox GT 40장 채점: 배포 모델 vs 새로 학습한 모델. gt40.py 와 같은 규칙(IoU>=0.5)."""
import glob, os
from ultralytics import YOLO

GT = "/workspace/data2/jinwoolee/xr_autolearning/data/bell412/gearbox/gt"
IMGS = sorted(glob.glob(GT + "/images/*.jpg"))
MODELS = {
    "배포 260824_retrain3x 11m":
        "/workspace/data2/jinwoolee/xr_autolearning/results/bench/260824_retrain3x/yolo11m/weights/best.pt",
    "신규 260901_181828 11m":
        "/workspace/data2/jinwoolee/xr_dash_260901/results/260901_181828/model/best.pt",
}
THRS = [0.25, 0.7]


def gt_boxes(name):
    p = GT + "/labels/" + os.path.splitext(os.path.basename(name))[0] + ".txt"
    out = []
    if os.path.exists(p):
        for ln in open(p):
            f = ln.split()
            if len(f) >= 5:
                out.append(tuple(float(x) for x in f[1:5]))
    return out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


print("GT %d장 (test1~4 각 10장)" % len(IMGS))
for label, w in MODELS.items():
    m = YOLO(w)
    gid = next((k for k, v in m.names.items() if v == "gearbox"), None)
    if gid is None:
        print("%s: gearbox 클래스 없음" % label); continue
    hit = {t: {} for t in THRS}
    for ip in IMGS:
        grp = os.path.basename(ip).split("_")[0]
        r = m.predict(ip, conf=0.10, imgsz=960, verbose=False, device=0)[0]
        H, W = r.orig_shape
        gts = [((cx - bw/2)*W, (cy - bh/2)*H, (cx + bw/2)*W, (cy + bh/2)*H)
               for cx, cy, bw, bh in gt_boxes(ip)]
        best_conf = 0.0
        for b, cf, cl in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                             r.boxes.cls.cpu().numpy()):
            if int(cl) != gid:
                continue
            if any(iou(tuple(map(float, b)), g) >= 0.5 for g in gts):
                best_conf = max(best_conf, float(cf))
        for t in THRS:
            d = hit[t].setdefault(grp, [0, 0])
            d[1] += 1
            if best_conf >= t:
                d[0] += 1
    print("=== %s (클래스 %d종)" % (label, len(m.names)))
    for t in THRS:
        tot = [sum(v[0] for v in hit[t].values()), sum(v[1] for v in hit[t].values())]
        per = "  ".join("%s %d/%d" % (k, v[0], v[1]) for k, v in sorted(hit[t].items()))
        print("   conf %-5s 전체 %2d/%2d   %s" % (t, tot[0], tot[1], per))
