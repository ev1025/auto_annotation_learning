"""Surion 스택(지성현 영역)의 데이터 루트에 배경 984장을 넣는다. 학습 시 자동 산입된다."""
import shutil
from pathlib import Path
SRC = Path("/workspace/data2/jinwoolee/xr_autolearning/data/bell412/_negatives/images")
DST = Path("/workspace/data1/Surion/video/_negatives/images")
DST.mkdir(parents=True, exist_ok=True)
n = 0
for p in sorted(SRC.glob("*.jpg")):
    d = DST / p.name
    if not d.exists():
        shutil.copyfile(p, d); n += 1
print(f"Surion: 새로 {n}장 · 총 {len(list(DST.glob('*.jpg')))}장 -> {DST}")
