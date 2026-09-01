"""밝기(톤매핑) 비교: 같은 프레임을 배포 모델로 두 번 돌려 결과를 견준다."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/app/scripts"); sys.path.insert(0, "/app/backend/autolearning")
import cv2  # noqa: E402
from ultralytics import YOLO  # noqa: E402

CONF, IMGSZ = 0.70, 960
W = Path("/app/results")
SETS = {"지금(변환없음)": W / "gt_frames", "톤매핑 npl=200": W / "gt_frames_tm"}
model = YOLO("/app/models/model.pt")
names = model.names

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

out = {}
common = None
for tag, root in SETS.items():
    keys = set()
    for d in sorted(root.glob("*")):
        for f in d.glob("*.jpg"):
            keys.add((d.name, f.stem))
    common = keys if common is None else (common & keys)
print("공통 프레임", len(common), flush=True)

for tag, root in SETS.items():
    cls_cnt, conf_sum, dup, frames_with = {}, {}, 0, 0
    for stem, idx in sorted(common):
        im = cv2.imread(str(root / stem / (idx + ".jpg")))
        if im is None:
            continue
        r = model.predict(source=im, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
        best = {}
        for b, cf, cl in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                             r.boxes.cls.cpu().numpy()):
            nm = names[int(cl)]
            cls_cnt[nm] = cls_cnt.get(nm, 0) + 1
            conf_sum[nm] = conf_sum.get(nm, 0.0) + float(cf)
            if nm not in best or float(cf) > best[nm][4]:
                best[nm] = [*map(float, b), float(cf)]
        if best:
            frames_with += 1
        if "r-ring" in best and "shim" in best and iou(best["r-ring"][:4], best["shim"][:4]) >= 0.5:
            dup += 1
    out[tag] = {"검출수": sum(cls_cnt.values()), "탐지된프레임": frames_with,
                "클래스수": len(cls_cnt), "rring_shim_겹침": dup,
                "부품별": {k: [v, round(conf_sum[k] / v, 3)] for k, v in
                          sorted(cls_cnt.items(), key=lambda kv: -kv[1])}}
    print(tag, json.dumps(out[tag], ensure_ascii=False)[:400], flush=True)

Path("/app/results/bright_cmp.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                encoding="utf-8")
print("저장 /app/results/bright_cmp.json", flush=True)
