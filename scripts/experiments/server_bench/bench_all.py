"""5090 서버에 남아 있는 벤치 배치 전부의 설정·결과를 한 줄씩 요약한다."""
import json, os
R = "/workspace/data2/jinwoolee/xr_autolearning/results/bench"
for b in sorted(os.listdir(R)):
    d = os.path.join(R, b)
    if not os.path.isdir(d):
        continue
    meta = {}
    mp = os.path.join(d, "meta.json")
    if os.path.exists(mp):
        meta = json.load(open(mp, encoding="utf-8"))
    print(f"[{b}] {meta.get('실험', '(설명 없음)')}")
    bp = os.path.join(d, "bench.json")
    if os.path.exists(bp):
        for r in json.load(open(bp, encoding="utf-8")):
            print("   %-9s mAP50 %-7s mAP50-95 %-7s %d분" % (
                r.get("model"), r.get("map50"), r.get("map5095"), round(r.get("train_sec", 0) / 60)))
    else:
        done = []
        for m in ("yolo11n", "yolo11s", "yolo11m", "yolo26s"):
            csv = os.path.join(d, m, "results.csv")
            if os.path.exists(csv):
                ep = open(csv, encoding="utf-8").read().strip().split("\n")[-1].split(",")[0].strip()
                done.append(f"{m} {ep}/100")
        print("   진행:", ", ".join(done) if done else "시작 전")
