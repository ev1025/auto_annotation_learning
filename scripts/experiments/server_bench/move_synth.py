"""합성 자원(배경·가림)을 data/bell412/_synth 한 폴더로 모은다."""
import shutil
from pathlib import Path
import sys
root = Path(sys.argv[1])                       # .../data/bell412
dst = root / "_synth"
dst.mkdir(exist_ok=True)
for name in ("backgrounds", "occluders"):
    src = root / name
    if src.is_dir() and not (dst / name).exists():
        shutil.move(str(src), str(dst / name))
        print(f"  이동: {name} -> _synth/{name}")
    elif (dst / name).exists():
        print(f"  이미 있음: _synth/{name}")
    else:
        print(f"  원본 없음: {name}")
print("  배경", len(list((dst / 'backgrounds').rglob('*.jpg'))) if (dst / 'backgrounds').exists() else 0,
      "· 가림", len(list((dst / 'occluders').glob('*'))) if (dst / 'occluders').exists() else 0)
