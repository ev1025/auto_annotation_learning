# -*- coding: utf-8 -*-
"""벤치 결과에서 모델 하나를 골라 배포용 run 폴더로 묶는다(대시보드 '적용'이 읽는 형식)."""
import json, shutil, tarfile
from pathlib import Path

BENCH = Path("/workspace/data2/jinwoolee/xr_autolearning/results/bench/260821_panel400")
MODEL = "yolo11m"
RUN_ID = "260823_120000"
OUT = Path("/workspace/data2/jinwoolee/xr_autolearning/results") / RUN_ID

# bench.json 은 실행 묶음마다 덮여 써진다(11m 은 앞 묶음이라 안 남았다).
# 홀드아웃 지표는 그 모델의 results.csv 마지막 줄에서 읽는다.
csv = (BENCH / MODEL / "results.csv").read_text(encoding="utf-8").strip().splitlines()
head = [h.strip() for h in csv[0].split(",")]
last = [v.strip() for v in csv[-1].split(",")]
row = dict(zip(head, last))
map50 = float(next(v for k, v in row.items() if "mAP50(B)" in k))
data_yaml = (Path("/workspace/data2/jinwoolee/benchset_panel400/data.yaml")).read_text(encoding="utf-8")
classes = [l.split(": ", 1)[1].strip() for l in data_yaml.splitlines() if l.strip() and l.startswith("  ") and ": " in l]

shutil.rmtree(OUT, ignore_errors=True)
(OUT / "model").mkdir(parents=True)
shutil.copyfile(BENCH / MODEL / "weights" / "best.pt", OUT / "model" / "best.pt")
meta = {
    "session": RUN_ID,
    "label": f"2026-08-23 12:00 · {len(classes)}종 · {MODEL} + 합성14127(판넬가림) + 배경984",
    "time": "2026-08-23 12:00:00",
    "classes": classes,
    "n_classes": len(classes),
    "weights": f"results/{RUN_ID}/model/best.pt",
    "trained_on": "RTX5090 GPU2",
    "epochs": 100,
    "base_model": f"{MODEL}.pt",
    "n_images": 22751,
    "augment": {"mixup": 0.2, "erasing": 0.5, "degrees": 15, "occluder": "금속판넬 2종"},
    "holdout_map50": round(map50, 4),
    "gt70_hit": 55,
    "note": "사람 GT 70장 기준 55장(IoU>=0.5). test3 8/10 · test4 1/10 로 장착 도메인이 처음 올라온 모델",
}
(OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

tar = Path("/workspace/data2/jinwoolee/run_deploy.tar.gz")
with tarfile.open(tar, "w:gz") as t:
    t.add(OUT, arcname=RUN_ID)
print(f"  run {RUN_ID} · {len(classes)}클래스 · best.pt {(OUT/'model'/'best.pt').stat().st_size:,}B")
print(f"  묶음 {tar} {tar.stat().st_size:,}B")
