# -*- coding: utf-8 -*-
"""방법 8(포인트참조+self-training) 증거 이미지 생성 -> docs/method_previews/selftrain/."""
import sys, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'verify'))
import cv2
import config
import dashboard_core as core
from ultralytics import YOLO

WORK = config.BASE_DIR / "exp_selftrain"
OUT = config.BASE_DIR / "docs" / "method_previews" / "selftrain"
OUT.mkdir(parents=True, exist_ok=True)
VAL = config.DATA_DIR / "gearbox_register_trial1" / "gearbox"


def draw(im, r, color=(0, 150, 255)):
    dets = []
    for b, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = map(int, b)
        dets.append((x1, y1, x2, y2, f"gearbox {float(cf):.2f}", color))
    core.render_detections(im, dets)
    return im


# 01: 참조점 -> 기어박스 분할 (스모크에서 만든 ref_check 재사용)
if (WORK / "ref_check.jpg").exists():
    shutil.copy(WORK / "ref_check.jpg", OUT / "01_지정한_점1개로_기어박스_참조확보.jpg")

# 02: self-training 이 주운 pool 프레임 (round1 이 conf>=0.6 로 pseudo 라벨한 것)
r1 = YOLO(str(WORK / "runs" / "round1" / "weights" / "best.pt"))
pool = sorted((WORK / "pool" / "images").glob("*.jpg"))
for p in pool:
    im = cv2.imread(str(p))
    res = r1.predict(source=str(p), conf=0.6, imgsz=640, verbose=False)[0]
    if len(res.boxes) > 0:
        cv2.imwrite(str(OUT / "02_방법7이_놓친_프레임을_selftraining이_주움.jpg"), draw(im, res))
        break

# 03: 검증 영상에서 round1 놓침 vs round2 검출 (나란히)
r2 = YOLO(str(WORK / "runs" / "round2" / "weights" / "best.pt"))
frames = sorted(VAL.glob("*.jpg"))
made = False
for p in frames:
    a = r1.predict(source=str(p), conf=0.4, imgsz=640, verbose=False)[0]
    b = r2.predict(source=str(p), conf=0.4, imgsz=640, verbose=False)[0]
    if len(a.boxes) == 0 and len(b.boxes) > 0:      # round1 놓치고 round2 잡은 프레임
        im = cv2.imread(str(p)); h, w = im.shape[:2]
        left = draw(im.copy(), a, (0, 0, 255))
        right = draw(im.copy(), b, (0, 150, 255))
        cv2.putText(left, "round1 (seed only): MISS", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(right, "round2 (+self-training): DETECT", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 150, 255), 3)
        combo = cv2.hconcat([cv2.resize(left, (760, int(h*760/w))), cv2.resize(right, (760, int(h*760/w)))])
        cv2.imwrite(str(OUT / "03_round1_놓침_vs_round2_검출.jpg"), combo)
        made = True
        break
print("증거 이미지 생성:", [p.name for p in sorted(OUT.glob("*.jpg"))], "| 03 생성" if made else "| 03 해당 프레임 없음")
