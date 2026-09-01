# -*- coding: utf-8 -*-
"""모델별 벤치마크: 같은 데이터·같은 설정으로 여러 사전학습 모델을 100 epoch 학습한다.

같은 것: 데이터셋(9354장/36클래스), imgsz 640, batch, epochs, seed, 증강 설정
다른 것: 사전학습 가중치(모델 크기·계열)뿐
결과: results/<run>/<model>/ 에 가중치·results.csv, bench.json 에 지표 요약
"""
import json
import sys
import time
from pathlib import Path

from ultralytics import YOLO

DATA = sys.argv[1]                       # data.yaml
OUT = Path(sys.argv[2])                  # 결과 폴더
MODELS = sys.argv[3].split(",")          # 예: yolo11n.pt,yolo11s.pt,yolo11m.pt
EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 100
BATCH = int(sys.argv[5]) if len(sys.argv) > 5 else 32
WORKERS = int(sys.argv[6]) if len(sys.argv) > 6 else 16
# 원본이 1080x1920 이라 JPEG 디코딩이 병목이었다(GPU 27%). 첫 에폭에 리샘플 결과를 RAM 에
# 올려두면 이후 에폭은 디코딩이 사라진다. 서버 여유 메모리 331GB, 필요량 약 21GB.
CACHE = sys.argv[7] if len(sys.argv) > 7 else "ram"
# 증강 세기. strong = 각도·시점 변화를 크게 준다(기본은 회전 15도 · 좌우반전만).
# 미검출 대부분이 '작게 찍힘 / 다른 각도 / 뒤집힘' 이라 기하 증강을 키워 본다.
AUG = {"degrees": 15.0}
if len(sys.argv) > 8 and sys.argv[8] == "strong":
    AUG = {"degrees": 45.0, "flipud": 0.5, "perspective": 0.0005, "shear": 5.0, "scale": 0.7}
# occl = 7/30 에 장착 도메인(test3·4)을 뚫은 레시피. 가림 시뮬레이션(erasing) + 배경 교란(mixup).
# 그때 test3·4 가 0·3% -> 99·80% 로 올라간 조합이라 그대로 되살린다.
elif len(sys.argv) > 8 and sys.argv[8] == "occl":
    AUG = {"degrees": 15.0, "mixup": 0.2, "erasing": 0.5}
OUT.mkdir(parents=True, exist_ok=True)

rows = []
for m in MODELS:
    name = Path(m).stem
    t0 = time.time()
    print(f"\n===== {name} 학습 시작 (epochs={EPOCHS}, batch={BATCH}) =====", flush=True)
    model = YOLO(m)
    model.train(data=DATA, epochs=EPOCHS, imgsz=640, batch=BATCH, device=0, workers=WORKERS,
                cache=CACHE, project=str(OUT), name=name, exist_ok=True, verbose=True, plots=True,
                seed=0, deterministic=True, **AUG)
    tr = time.time() - t0
    mv = model.val(data=DATA, imgsz=640, batch=BATCH, device=0, workers=WORKERS,
                   project=str(OUT), name=f"{name}_val", exist_ok=True, verbose=False)
    b = mv.box
    row = {"model": name, "train_sec": round(tr, 1),
           "map50": round(float(b.map50), 4), "map5095": round(float(b.map), 4),
           "precision": round(float(b.mp), 4), "recall": round(float(b.mr), 4),
           "weights": str(Path(model.trainer.best)),
           "params_M": round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2)}
    row["f1"] = round(2 * row["precision"] * row["recall"] / (row["precision"] + row["recall"]), 4) \
        if (row["precision"] + row["recall"]) else 0.0
    # 부품별 mAP50 (가장 약한 클래스 확인용)
    try:
        row["per_class_map50"] = {mv.names[int(c)]: round(float(v), 4)
                                  for c, v in zip(b.ap_class_index, b.ap50)}
    except Exception as e:
        row["per_class_map50"] = {"error": type(e).__name__}
    rows.append(row)
    print(f"===== {name} 완료: mAP50 {row['map50']:.4f} · {tr / 60:.1f}분 =====", flush=True)
    (OUT / "bench.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    del model

print("\n== 요약 ==")
print(f"{'모델':10} {'파라미터M':>9} {'mAP50':>8} {'mAP50-95':>9} {'P':>7} {'R':>7} {'F1':>7} {'학습분':>7}")
for r in rows:
    print(f"{r['model']:10} {r['params_M']:9.2f} {r['map50']:8.4f} {r['map5095']:9.4f} "
          f"{r['precision']:7.4f} {r['recall']:7.4f} {r['f1']:7.4f} {r['train_sec'] / 60:7.1f}")
