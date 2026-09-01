# -*- coding: utf-8 -*-
"""평가영상에서 부품별 검출 빈도·신뢰도를 센다. GT 가 없어도 '무엇이 몇 프레임에 잡히나' 는 알 수 있다.
0.5초(15프레임) 간격으로 표본을 뽑는다. 연속 프레임은 거의 같은 그림이라 세는 의미가 없다."""
import glob, json, os
from collections import defaultdict
import cv2
from ultralytics import YOLO

VID = "/app/data/bell412/_gt_videos"
STEP = 15
CONFS = (0.25, 0.5, 0.7)
m = YOLO("/app/models/model.pt")
names = m.names

out = {}
for vp in sorted(glob.glob(f"{VID}/*.mp4")):
    base = os.path.basename(vp)
    cap = cv2.VideoCapture(vp)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames, idx = [], 0
    while True:
        ok, im = cap.read()
        if not ok: break
        if idx % STEP == 0: frames.append((idx, im))
        idx += 1
    cap.release()
    # 부품별: 임계값별 '검출된 표본 수' 와 최고 신뢰도
    per = defaultdict(lambda: {"n": {c: 0 for c in CONFS}, "best": 0.0})
    for i in range(0, len(frames), 32):
        chunk = frames[i:i+32]
        for (fi, _), r in zip(chunk, m.predict([f[1] for f in chunk], conf=0.10, verbose=False, device=0)):
            seen = defaultdict(float)
            if r.boxes is not None:
                for cl, cf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                    nm = names[int(cl)]
                    seen[nm] = max(seen[nm], cf)
            for nm, cf in seen.items():
                for c in CONFS:
                    if cf >= c: per[nm]["n"][c] += 1
                per[nm]["best"] = max(per[nm]["best"], cf)
    out[base] = {"total_frames": total, "sampled": len(frames),
                 "parts": {k: {"n": v["n"], "best": round(v["best"], 3)} for k, v in per.items()}}
    print(f"[{base}] 전체 {total}f · 표본 {len(frames)}장")
    for nm, v in sorted(per.items(), key=lambda x: -x[1]["n"][0.25])[:12]:
        print(f"    {nm:<28} 0.25:{v['n'][0.25]:>3}  0.5:{v['n'][0.5]:>3}  0.7:{v['n'][0.7]:>3}  최고 {v['best']:.2f}")
    print()
json.dump(out, open("/app/results/gt_scan.json", "w"), ensure_ascii=False, indent=1)

print("===== 전체 합계 (표본 프레임 기준) =====")
agg = defaultdict(lambda: {c: 0 for c in CONFS})
tot = sum(v["sampled"] for v in out.values())
for v in out.values():
    for nm, d in v["parts"].items():
        for c in CONFS: agg[nm][c] += d["n"][c]
print(f"표본 총 {tot}장")
print(f"  {'부품':<28} {'0.25':>6} {'0.5':>6} {'0.7':>6}")
for nm, d in sorted(agg.items(), key=lambda x: -x[1][0.25]):
    print(f"  {nm:<28} {d[0.25]:>6} {d[0.5]:>6} {d[0.7]:>6}")
