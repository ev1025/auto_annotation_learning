# -*- coding: utf-8 -*-
"""학습셋을 '영상별 뒤 15% 블록'으로 나눈다.

왜 이렇게 나누나
  부품마다 라벨된 영상이 한 개뿐이라 영상 단위 홀드아웃이 불가능하다(빼면 클래스가 사라진다).
  10장당 1장 같은 교차 추출은 인접 프레임이 학습에 남아 사실상 같은 사진을 채점하게 된다.
  영상의 마지막 연속 구간을 통째로 빼면 같은 영상이라도 자세·구도가 다른 부분이 검증에 남는다.
  정답은 SAM2 오토라벨이다(사람 GT 아님) -> '미학습 구간에서 오토라벨과의 일치도'를 재는 지표.
합성(syn_*)은 배경 합성이라 검증에 넣지 않는다(학습에만).
"""
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
TAIL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
ti, tl = ROOT / "images" / "train", ROOT / "labels" / "train"
vi, vl = ROOT / "images" / "val", ROOT / "labels" / "val"
for d in (vi, vl):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)

by_vid = {}
for p in sorted(ti.glob("*.jpg")):
    if p.stem.startswith("syn_"):
        continue
    vid = re.sub(r"_\d+$", "", p.stem)
    by_vid.setdefault(vid, []).append(p)

moved = 0
per = {}
for vid, ps in sorted(by_vid.items()):
    ps.sort(key=lambda x: int(re.search(r"_(\d+)$", x.stem).group(1)))
    k = max(1, int(len(ps) * TAIL))
    for p in ps[-k:]:
        shutil.move(str(p), vi / p.name)
        shutil.move(str(tl / f"{p.stem}.txt"), vl / f"{p.stem}.txt")
        moved += 1
    per[vid] = {"frames": len(ps), "val": k}

n_tr = len(list(ti.glob("*.jpg")))
n_syn = len(list(ti.glob("syn_*.jpg")))
print(f"학습 {n_tr}장(실촬 {n_tr - n_syn} + 합성 {n_syn}) · 검증 {moved}장 · 영상 {len(by_vid)}개")
(ROOT / "split.json").write_text(json.dumps(
    {"tail": TAIL, "train": n_tr, "synth": n_syn, "val": moved, "per_video": per},
    ensure_ascii=False, indent=1), encoding="utf-8")
