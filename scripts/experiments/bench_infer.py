# -*- coding: utf-8 -*-
"""추론 성능 벤치마크: 학습된 각 모델을 PyTorch-GPU/PyTorch-CPU/ONNX-CPU로 재고
'사진(단건 지연 ms)'과 '실시간(연속 처리 FPS)' 둘 다 측정.
ONNX-GPU는 onnxruntime-gpu(CUDAExecutionProvider) 있을 때만.
사용: XR_BASE=$HOME/xr_autolearning python scripts/experiments/bench_infer.py <bench_ts> [gpu번호]
결과: results/bench/<ts>/infer.json · infer.txt
"""
import os, sys, time, json, glob
from pathlib import Path

BASE = Path(os.environ.get("XR_BASE") or Path(__file__).resolve().parents[2])
TS = sys.argv[1]
GPU = sys.argv[2] if len(sys.argv) > 2 else "3"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU          # torch-gpu → device 0 = 물리 GPU<GPU>
out = BASE / "results" / "bench" / TS

from ultralytics import YOLO
import onnxruntime as ort
GPU_ONNX = "CUDAExecutionProvider" in ort.get_available_providers()

val = [l for l in (out / "val.txt").read_text(encoding="utf-8").splitlines() if l][:80]


def measure(model_path, device):
    m = YOLO(model_path)
    for i in range(6):                              # 워밍업
        m.predict(val[i % len(val)], imgsz=640, device=device, verbose=False)
    t = time.time()                                 # 사진: 단건 20회 평균 지연
    for i in range(20):
        m.predict(val[i % len(val)], imgsz=640, device=device, verbose=False)
    single_ms = (time.time() - t) / 20 * 1000
    t = time.time()                                 # 실시간: 연속 80프레임 처리량
    for i in range(80):
        m.predict(val[i % len(val)], imgsz=640, device=device, verbose=False)
    fps = 80 / (time.time() - t)
    return {"ms": round(single_ms, 1), "fps": round(fps, 1)}


rows = []
for pt in sorted(glob.glob(str(out / "runs" / "*" / "weights" / "best.pt"))):
    name = Path(pt).parents[1].name
    onnx = Path(pt).with_suffix(".onnx")
    if not onnx.exists():
        try:
            YOLO(pt).export(format="onnx", imgsz=640, device="cpu")
        except Exception:
            pass
    row = {"model": name}
    for key, mp, dev in [("torch_gpu", pt, 0), ("torch_cpu", pt, "cpu"),
                         ("onnx_cpu", str(onnx) if onnx.exists() else None, "cpu"),
                         ("onnx_gpu", str(onnx) if (onnx.exists() and GPU_ONNX) else None, 0)]:
        if mp is None:
            row[key] = None; continue
        try:
            row[key] = measure(mp, dev)
        except Exception as e:
            row[key] = {"error": type(e).__name__}
    rows.append(row)
    print("DONE", name, row.get("torch_gpu"), flush=True)

data = {"gpu_onnx_available": GPU_ONNX, "rows": rows}
(out / "infer.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def cell(v):
    if not v: return "     -    "
    if "error" in v: return "   ERR   "
    return f"{v['ms']:>5}ms/{v['fps']:>4}fps"
lines = ["추론 성능 (사진=단건 지연 · 실시간=연속 FPS, imgsz640)",
         "ONNX-GPU: " + ("가능" if GPU_ONNX else "불가(onnxruntime-gpu 미설치)"),
         f"{'model':10}{'PyTorch-GPU':>16}{'PyTorch-CPU':>16}{'ONNX-CPU':>16}{'ONNX-GPU':>16}"]
lines.append("-" * len(lines[-1]))
for r in rows:
    lines.append(f"{r['model']:10}{cell(r.get('torch_gpu')):>16}{cell(r.get('torch_cpu')):>16}{cell(r.get('onnx_cpu')):>16}{cell(r.get('onnx_gpu')):>16}")
(out / "infer.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
