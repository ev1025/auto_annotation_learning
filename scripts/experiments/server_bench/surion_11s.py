"""Surion 오토러닝의 기본 사전학습 모델을 yolo26s -> yolo11s 로 바꾼다.

근거(2026-08-21 벤치, 36부품 100epoch 동일 조건):
  사람 GT mAP50  11s 0.789 / 26s 0.743,  학습시간 62분 / 91분
"""
import shutil
from pathlib import Path

edits = [
    (Path("/workspace/data1/Surion/autolearn/app/config.py"), "yolo26s.pt", "yolo11s.pt"),
    (Path("/workspace/data1/Surion/docker-compose.yml"), "AUTOLEARN_PRETRAINED:-yolo26s.pt",
     "AUTOLEARN_PRETRAINED:-yolo11s.pt"),
]
bak = Path("/workspace/data2/jinwoolee/surion_backup_260821")
bak.mkdir(parents=True, exist_ok=True)
for p, old, new in edits:
    s = p.read_text(encoding="utf-8")
    if new in s and old not in s:
        print(f"  이미 반영됨: {p.name}"); continue
    assert s.count(old) == 1, f"{p.name}: {old} {s.count(old)}건"
    shutil.copyfile(p, bak / (p.name + ".pre11s"))
    p.write_text(s.replace(old, new), encoding="utf-8")
    print(f"  바꿈: {p} · {old} -> {new}")

# 오프라인에서도 학습이 시작되게 가중치를 함께 둔다(컨테이너 CWD=/app 에 COPY 되는 위치)
src = Path("/workspace/data2/jinwoolee/xr_autolearning/yolo11s.pt")
dst = Path("/workspace/data1/Surion/autolearn/app/yolo11s.pt")
if src.exists() and not dst.exists():
    shutil.copyfile(src, dst)
    print(f"  가중치 배치: {dst} ({round(dst.stat().st_size/1e6,1)} MB)")
