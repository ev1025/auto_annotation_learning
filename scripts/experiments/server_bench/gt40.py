# -*- coding: utf-8 -*-
"""gearbox GT 40장 채점. '맞춤' = 그 사진에서 gearbox 를 GT 박스와 IoU>=0.5 로 잡은 경우.
test1~4 로 나눠 센다(각 10장). 36클래스 모델은 gearbox 클래스만 본다."""
import glob, os
from ultralytics import YOLO

GT = "/workspace/data2/jinwoolee/xr_autolearning/data/bell412/gearbox/gt"
IMGS = sorted(glob.glob(f"{GT}/images/*.jpg"))
MODELS = {
    "1클래스  gearbox1  (11s)":
        "/workspace/data2/jinwoolee/xr_autolearning/results/bench/260826_gearbox1/yolo11s/weights/best.pt",
    "36클래스 retrain3x (11s)":
        "/workspace/data2/jinwoolee/xr_autolearning/results/bench/260824_retrain3x/yolo11s/weights/best.pt",
    "36클래스 retrain3x (11m)":
        "/workspace/data2/jinwoolee/xr_autolearning/results/bench/260824_retrain3x/yolo11m/weights/best.pt",
}
THRS = [0.25, 0.7]


def gt_boxes(name):
    """YOLO 라벨 -> [(cx,cy,w,h)] 비율 좌표."""
    p = f"{GT}/labels/{os.path.splitext(os.path.basename(name))[0]}.txt"
    out = []
    if os.path.exists(p):
        for ln in open(p):
            f = ln.split()
            if len(f) >= 5:
                out.append(tuple(float(x) for x in f[1:5]))
    return out


def iou(a, b):
    """a=(x1,y1,x2,y2) 픽셀, b=(x1,y1,x2,y2) 픽셀."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


print(f"GT {len(IMGS)}장  (test1~4 각 10장)\n")
for label, w in MODELS.items():
    m = YOLO(w)
    gid = next((k for k, v in m.names.items() if v == "gearbox"), None)
    if gid is None:
        print(f"{label}: gearbox 클래스 없음"); continue
    # 프레임별 (최고 신뢰도, 그 박스의 GT 대비 최대 IoU)
    rec = {}
    for i in range(0, len(IMGS), 40):
        batch = IMGS[i:i+40]
        for path, r in zip(batch, m.predict(batch, conf=0.10, verbose=False, device=0)):
            h, wd = r.orig_shape
            gts = []
            for cx, cy, bw, bh in gt_boxes(path):
                gts.append((( cx-bw/2)*wd, (cy-bh/2)*h, (cx+bw/2)*wd, (cy+bh/2)*h))
            hits = []
            if r.boxes is not None:
                for cl, cf, xy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(),
                                      r.boxes.xyxy.tolist()):
                    if int(cl) != gid:
                        continue
                    best = max((iou(tuple(xy), g) for g in gts), default=0.0)
                    hits.append((cf, best))
            rec[os.path.basename(path)] = hits
    print(f"=== {label}   (클래스 {len(m.names)}종)")
    for t in THRS:
        per = {}
        for name, hits in rec.items():
            grp = name.split("_")[0]
            ok = any(cf >= t and io >= 0.5 for cf, io in hits)
            per.setdefault(grp, [0, 0])
            per[grp][1] += 1
            per[grp][0] += 1 if ok else 0
        tot_ok = sum(v[0] for v in per.values()); tot_n = sum(v[1] for v in per.values())
        detail = "  ".join(f"{g} {per[g][0]}/{per[g][1]}" for g in sorted(per))
        print(f"   conf {t}   맞춤 {tot_ok}/{tot_n}  ({round(100*tot_ok/tot_n,1)}%)   [{detail}]")
    print()
