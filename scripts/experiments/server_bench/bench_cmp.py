import json, os
R = "/workspace/data2/jinwoolee/xr_autolearning/results/bench"
for b in ["260820_113527", "260820_191442_bg", "260821_091951_aug"]:
    p = os.path.join(R, b, "bench.json")
    if not os.path.exists(p):
        print("[%s] bench.json 없음" % b); continue
    rows = json.load(open(p))
    print("[%s]" % b)
    for r in rows:
        keys = [k for k in r if "map" in k.lower()]
        vals = " · ".join("%s %s" % (k, r[k]) for k in keys if not isinstance(r[k], dict))
        print("   %-9s %s · %d분" % (r.get("model"), vals, round(r.get("train_sec", 0) / 60)))
