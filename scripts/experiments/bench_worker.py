# -*- coding: utf-8 -*-
"""YOLO 버전 벤치마크 - 워커(모델 1개 학습+검증). 런처 bench_yolo.py가 GPU별로 호출.
사용: python bench_worker.py <model> <device> <outdir> <data_yaml> <epochs>
  <model>  : yolov8s 등(사전학습 <model>.pt 사용, 없으면 다운로드)
  <device> : 물리 GPU 번호(예: 2). CUDA_VISIBLE_DEVICES로 고정 후 내부 device=0.
결과: <outdir>/<model>.json (mAP·정밀도·재현율·파라미터·크기·속도·학습시간)
"""
import sys, os, json, time
from pathlib import Path

model_name, device, outdir, data_yaml, epochs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
os.environ["CUDA_VISIBLE_DEVICES"] = device            # torch import 전에 GPU 고정
out = Path(outdir)
res = {"model": model_name, "device": device}
try:
    from ultralytics import YOLO
    t0 = time.time()
    m = YOLO(f"{model_name}.pt")
    m.train(data=data_yaml, epochs=epochs, imgsz=640, batch=16, device=0,
            project=str(out / "runs"), name=model_name, exist_ok=True, verbose=False, plots=False)
    best = Path(m.trainer.best)
    res["train_min"] = round((time.time() - t0) / 60, 1)

    det = YOLO(str(best))
    r = det.val(data=data_yaml, imgsz=640, batch=16, device=0, verbose=False)
    box = r.box
    res.update({
        "map50": round(float(box.map50), 4),
        "map5095": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "params_M": round(sum(p.numel() for p in det.model.parameters()) / 1e6, 2),
        "size_MB": round(best.stat().st_size / 1e6, 1),
        "speed_ms": round(float(r.speed.get("inference", 0)), 2),
        "weights": str(best),
    })
except Exception as e:
    res["error"] = f"{type(e).__name__}: {e}"
(out / f"{model_name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print("DONE", model_name, res.get("map50", res.get("error")))
