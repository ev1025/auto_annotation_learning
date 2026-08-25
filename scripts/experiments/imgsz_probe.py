# -*- coding: utf-8 -*-
"""학습은 그대로 두고 추론 해상도만 올려 GT 70장에 어떤 영향이 있는지 먼저 본다.
60시간짜리 1280 재학습을 걸기 전에 2분으로 방향을 확인하는 용도."""
import sys
from pathlib import Path
BASE = Path(r"C:/Users/eg287/OneDrive/바탕 화면/project/수리온/xr_autolearning")
sys.path.insert(0, str(BASE / "scripts" / "experiments"))
import gt_viewer as gv
from ultralytics import YOLO

MODELS = {
    "11m(3x)":  BASE / "results/bench/260824_retrain3x/yolo11m/weights/best.pt",
    "11s(3x)":  BASE / "results/bench/260824_retrain3x/yolo11s/weights/best.pt",
    "11m(배포)": BASE / "backend/yolo_server/model.pt",
}
images = gv.gt_images()
paths = [im["abs"] for im in images]
print(f"GT {len(images)}장 · 적중 = 정답클래스 IoU>=0.5\n")
print(f"{'모델':10s} {'해상도':>6s} {'conf0.25':>9s} {'conf0.7':>8s}")
for name, w in MODELS.items():
    if not w.exists():
        print(" 없음", w); continue
    mod = YOLO(str(w))
    for sz in (640, 960, 1280, 1600):
        res = []
        for k in range(0, len(paths), 8):
            res += list(mod.predict(source=paths[k:k+8], conf=0.25, imgsz=sz, device=0, verbose=False))
        h25 = h70 = 0
        for im, r in zip(images, res):
            b25 = b70 = 0.0
            for b, cf, cl in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                 r.boxes.cls.cpu().numpy()):
                if mod.names[int(cl)] != im["part"]:
                    continue
                v = max((gv.iou([float(x) for x in b], g) for g in im["gt"]), default=0.0)
                b25 = max(b25, v)
                if float(cf) >= 0.7:
                    b70 = max(b70, v)
            h25 += b25 >= 0.5
            h70 += b70 >= 0.5
        print(f"{name:10s} {sz:6d} {h25:9d} {h70:8d}")
    del mod
