# -*- coding: utf-8 -*-
"""test4(기체 장착 gearbox) 프레임에서 gearbox 검출률 비교.
검출률 = 200 프레임 중 gearbox 를 1개 이상 잡은 프레임의 비율. GT 박스가 없어도 재는 값이다
(그 자리에 gearbox 가 있다는 사실이 정답이므로)."""
import glob, json
from ultralytics import YOLO

FRAMES = sorted(glob.glob("/workspace/data2/jinwoolee/xr_autolearning/data/_frame_cache/test4/*.jpg"))
MODELS = {
    "1클래스 gearbox1 (yolo11s)":
        "/workspace/data2/jinwoolee/xr_autolearning/results/bench/260826_gearbox1/yolo11s/weights/best.pt",
    "36클래스 retrain3x (yolo11s)":
        "/workspace/data2/jinwoolee/xr_autolearning/results/bench/260824_retrain3x/yolo11s/weights/best.pt",
}
THRS = [0.25, 0.5, 0.7]
print(f"test4 프레임 {len(FRAMES)}장\n")
out = {}
for label, w in MODELS.items():
    m = YOLO(w)
    names = m.names
    gid = [k for k, v in names.items() if v == "gearbox"]
    if not gid:
        print(f"{label}: 클래스에 gearbox 없음 -> {list(names.values())[:8]}"); continue
    gid = gid[0]
    # conf 를 가장 낮은 값으로 한 번만 추론하고, 임계는 결과에서 걸러 센다
    best = []          # 프레임별 gearbox 최고 신뢰도(없으면 0)
    for i in range(0, len(FRAMES), 50):
        for r in m.predict(FRAMES[i:i+50], conf=0.10, verbose=False, device=0):
            c = r.boxes.cls.tolist() if r.boxes is not None else []
            f = r.boxes.conf.tolist() if r.boxes is not None else []
            g = [cf for cl, cf in zip(c, f) if int(cl) == gid]
            best.append(max(g) if g else 0.0)
    row = {}
    for t in THRS:
        n = sum(1 for b in best if b >= t)
        row[t] = (n, round(100 * n / len(best), 1))
    mean_conf = round(sum(b for b in best if b > 0) / max(1, sum(1 for b in best if b > 0)), 3)
    out[label] = {"rows": row, "n_classes": len(names), "mean_conf": mean_conf}
    print(f"=== {label}   (클래스 {len(names)}종)")
    for t in THRS:
        n, p = row[t]
        print(f"   conf {t:<5} 검출 {n:>3}/{len(best)}  = {p:>5}%")
    print(f"   잡힌 프레임 평균 신뢰도 {mean_conf}\n")
json.dump(out, open("/workspace/data2/jinwoolee/test4_result.json", "w"), ensure_ascii=False, indent=1)
