import hashlib, json, sys
import torch
p = sys.argv[1]
ck = torch.load(p, map_location="cpu", weights_only=False)
m = ck.get("model"); names = getattr(m, "names", None) or ck.get("names")
print("클래스 수:", len(names) if names else "?")
print("클래스:", list(names.values()) if isinstance(names, dict) else names)
print("epoch:", ck.get("epoch"), "· date:", ck.get("date"), "· ultralytics:", ck.get("version"))
ta = ck.get("train_args") or {}
keys = ("model", "data", "epochs", "imgsz", "batch", "seed", "degrees", "scale", "fliplr", "mosaic")
print("train_args:", json.dumps({k: ta.get(k) for k in keys if k in ta}, ensure_ascii=False))
print("md5:", hashlib.md5(open(p, "rb").read()).hexdigest())
