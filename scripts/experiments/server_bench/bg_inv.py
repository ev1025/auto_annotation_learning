"""배경 984장이 어떤 촬영에서 왔는지 접두사로 센다."""
import collections
from pathlib import Path
SRC = Path("/workspace/data2/jinwoolee/benchset_hangar/images/train")
c = collections.Counter()
for p in sorted(SRC.glob("bg_*.jpg")):
    stem = p.stem[3:]                       # bg_ 제거
    key = stem.split("_")[0] if "_" in stem else stem[:12]
    c[key] += 1
print("배경 합계:", sum(c.values()), "장")
for k, n in c.most_common():
    print(f"  {k:<22}{n:>5}장")
