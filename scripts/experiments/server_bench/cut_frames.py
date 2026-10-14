"""라벨이 가리키는 프레임을 영상에서 다시 만든다(로컬에서 8GB 를 옮기지 않기 위해).
같은 코드·같은 영상이므로 서브샘플 결과(파일명 = 순번)도 같다."""
import sys, time
from pathlib import Path
W = Path("/workspace/data2/jinwoolee/xr_dash_260901")
sys.path.insert(0, str(W / "scripts"))
sys.path.insert(0, str(W / "backend" / "autolearning"))
import autolabel  # noqa: E402

vids = sorted(p for p in (W / "data" / "bell412").rglob("*")
              if p.suffix.lower() in (".mp4", ".mov", ".avi") and "_synth" not in str(p))
print(f"영상 {len(vids)}개", flush=True)
t0 = time.time()
for i, vp in enumerate(vids, 1):
    dest = autolabel.cache_dir_of(vp)
    autolabel._extract(vp, dest)
    n = len(list(dest.glob("*.jpg")))
    print(f"  [{i}/{len(vids)}] {vp.parent.parent.name}/{vp.stem}  {n}장  {time.time()-t0:.0f}s", flush=True)
print("완료", flush=True)
