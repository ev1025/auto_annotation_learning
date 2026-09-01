"""배경 984장을 우리 작업폴더 정본으로 모으고, 배포용 묶음을 만든다(우리 영역만 건드린다)."""
import shutil, tarfile
from pathlib import Path

SRC = Path("/workspace/data2/jinwoolee/benchset_hangar/images/train")
DST = Path("/workspace/data2/jinwoolee/xr_autolearning/data/bell412/_negatives/images")
bgs = sorted(SRC.glob("bg_*.jpg"))
DST.mkdir(parents=True, exist_ok=True)
new = 0
for p in bgs:
    d = DST / p.name[3:]            # 앞의 bg_ 는 학습 조립 때 다시 붙으므로 떼어 둔다
    if not d.exists():
        shutil.copyfile(p, d); new += 1
print(f"5090 우리폴더: 새로 {new}장 · 총 {len(list(DST.glob('*.jpg')))}장")

tar = Path("/workspace/data2/jinwoolee/negatives_984.tar.gz")
with tarfile.open(tar, "w:gz") as t:
    for p in bgs:
        t.add(p, arcname=p.name[3:])
print("묶음:", tar.name, round(tar.stat().st_size / 1e6, 1), "MB")
