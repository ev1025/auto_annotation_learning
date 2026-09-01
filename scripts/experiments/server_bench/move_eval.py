"""평가 영상을 data/_eval/<부품> -> data/bell412/<부품>/eval 로 옮긴다."""
import shutil, sys
from pathlib import Path
root = Path(sys.argv[1])                     # .../data
src_root = root / "_eval"
if not src_root.is_dir():
    print("  _eval 없음(이미 옮겼거나 대상 아님)"); raise SystemExit(0)
for pdir in sorted(src_root.iterdir()):
    if not pdir.is_dir():
        continue
    dst = root / "bell412" / pdir.name / "eval"
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in pdir.iterdir():
        if not (dst / f.name).exists():
            shutil.move(str(f), str(dst / f.name)); n += 1
    print(f"  {pdir.name}: {n}개 -> {dst}")
    try:
        pdir.rmdir()
    except OSError:
        pass
try:
    src_root.rmdir(); print("  _eval 폴더 제거")
except OSError as e:
    print("  _eval 남음:", e)
