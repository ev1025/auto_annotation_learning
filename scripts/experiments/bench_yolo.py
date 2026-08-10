# -*- coding: utf-8 -*-
"""YOLO 버전 벤치마크 - 런처. 같은 부품 데이터(85:15 분할)로 버전별 학습→val mAP 비교.
GPU 2·3 병렬(2개 동시), 순위표 저장. 라벨이 SAM2 자동생성이라 mAP는 버전 간 상대비교용.
사용: XR_BASE=$HOME/xr_autolearning python scripts/experiments/bench_yolo.py
결과: results/bench/<시각>/ (bench.yaml·train/val.txt·runs/·<model>.json·summary.json·summary.txt)
"""
import os, sys, json, random, subprocess, time
from datetime import datetime
from pathlib import Path
import yaml

BASE = Path(os.environ.get("XR_BASE") or Path(__file__).resolve().parents[2])
MULTI = BASE / "results" / "parts" / "auto_baseline" / "multiclass"
IMG_DIR = MULTI / "images"
SRC_YAML = MULTI / "data.yaml"
MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolo11n", "yolo11s", "yolo11m", "yolo26n", "yolo26s", "yolo26m"]
DEVICES = ["2", "3"]
EPOCHS = 100

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out = BASE / "results" / "bench" / ts
out.mkdir(parents=True, exist_ok=True)


def log(m):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(out / "bench.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


# 1) 85:15 분할(고정 시드) — 모든 모델 공통
imgs = sorted(IMG_DIR.glob("*.jpg"))
random.seed(42); random.shuffle(imgs)
k = max(1, int(len(imgs) * 0.15))
val, train = imgs[:k], imgs[k:]
(out / "train.txt").write_text("\n".join(str(p) for p in train), encoding="utf-8")
(out / "val.txt").write_text("\n".join(str(p) for p in val), encoding="utf-8")
names = yaml.safe_load(SRC_YAML.read_text(encoding="utf-8"))["names"]
nb = "\n".join(f"  {i}: {n}" for i, n in sorted(names.items()))
(out / "bench.yaml").write_text(
    f"path: {BASE.as_posix()}\ntrain: {(out/'train.txt').as_posix()}\nval: {(out/'val.txt').as_posix()}\nnames:\n{nb}\n",
    encoding="utf-8")
log(f"분할: train {len(train)} / val {len(val)} · {len(names)}클래스 · 모델 {len(MODELS)}개 · {EPOCHS}ep")

# 2) 사전학습 가중치 미리 확보(다운로드 경쟁 방지, CPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from ultralytics import YOLO
for m in MODELS:
    if not (BASE / f"{m}.pt").exists():
        log(f"다운로드: {m}.pt")
        try:
            YOLO(f"{m}.pt")
        except Exception as e:
            log(f"  실패 {m}: {e}")

# 3) GPU 2·3 병렬 학습(2개 동시)
def launch(model, dev):
    log(f"학습 시작: {model} @GPU{dev}")
    return subprocess.Popen(
        [str(BASE / "venv/bin/python"), str(BASE / "scripts/experiments/bench_worker.py"),
         model, dev, str(out), str(out / "bench.yaml"), str(EPOCHS)],
        cwd=str(BASE))

queue = list(MODELS)
running = {}
for dev in DEVICES:
    if queue:
        running[dev] = launch(queue.pop(0), dev)
while running:
    time.sleep(15)
    for dev in list(running):
        if running[dev].poll() is not None:
            done = running.pop(dev)
            log(f"완료 슬롯 GPU{dev} (남은 {len(queue)})")
            if queue:
                running[dev] = launch(queue.pop(0), dev)

# 4) 집계 → 순위표(mAP@0.5 기준)
res = []
for m in MODELS:
    f = out / f"{m}.json"
    if f.exists():
        res.append(json.loads(f.read_text(encoding="utf-8")))
ok = [r for r in res if "map50" in r]
ok.sort(key=lambda x: -x["map50"])
(out / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
hdr = f"{'model':10} {'mAP50':>7} {'mAP50-95':>9} {'P':>6} {'R':>6} {'params(M)':>10} {'size(MB)':>9} {'속도(ms)':>9} {'학습(min)':>9}"
lines = ["YOLO 버전 벤치마크 (부품 34클래스, 85:15, imgsz640, " + str(EPOCHS) + "ep) — mAP는 상대비교용(자동라벨)", hdr, "-" * len(hdr)]
for r in ok:
    lines.append(f"{r['model']:10} {r['map50']:>7} {r['map5095']:>9} {r['precision']:>6} {r['recall']:>6} "
                 f"{r['params_M']:>10} {r['size_MB']:>9} {r['speed_ms']:>9} {r['train_min']:>9}")
for r in res:
    if "error" in r:
        lines.append(f"{r['model']:10}  ERROR: {r['error']}")
(out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
log("벤치마크 완료 → " + str(out / "summary.txt"))
print("\n".join(lines))
