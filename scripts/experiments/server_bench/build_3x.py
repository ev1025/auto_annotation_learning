# -*- coding: utf-8 -*-
"""재학습 데이터셋: 실촬 전체 + 부품별 합성 3배(가중 추첨) + 배경 567(격납고·조종실).

클래스명은 부품 폴더명(개명 반영: seal_spring·cap_spring). cap_spring 은 라벨이 없어 제외.
이미지는 기존 benchset 풀 + 이번에 올린 누락분에서 하드링크(디스크 절약).
"""
import os, shutil, subprocess, sys, tarfile, time
from collections import Counter
from pathlib import Path

ROOT = Path("/workspace/data2/jinwoolee/xr_autolearning")
sys.path.insert(0, str(ROOT / "backend" / "autolearning"))
sys.path.insert(0, str(ROOT / "scripts"))

WORK = Path("/workspace/data2/jinwoolee")
OUT = WORK / "benchset_3x"
POOL_OLD = WORK / "benchset_hangar" / "images" / "train"
POOL_NEW = WORK / "pool_new"
VAL_EVERY = 10
MULT = 3                      # 합성 = 실촬의 3배(부품별)

t0 = time.time()
# 0) 라벨·누락 이미지 전개
lab = WORK / "labels_stage"
shutil.rmtree(lab, ignore_errors=True); lab.mkdir()
with tarfile.open(WORK / "labels_full.tar.gz") as t:
    t.extractall(lab)
POOL_NEW.mkdir(exist_ok=True)
if (WORK / "missing_imgs.tar").exists() and not any(POOL_NEW.iterdir()):
    with tarfile.open(WORK / "missing_imgs.tar") as t:
        t.extractall(POOL_NEW)
print(f"라벨 전개 {sum(1 for _ in lab.glob('*/labels/*.txt'))} · 신규 이미지 {sum(1 for _ in POOL_NEW.glob('*.jpg'))}", flush=True)

# 1) 부품별 라벨 수집(빈 라벨 제외)
staged, per = [], Counter()
miss_img = 0
for lp in sorted(lab.glob("*/labels/*.txt")):
    part = lp.parent.parent.name
    boxes = [f[1:5] for f in (l.split() for l in lp.read_text(encoding="utf-8").splitlines()) if len(f) == 5]
    if not boxes:
        continue
    ip = POOL_OLD / f"{lp.stem}.jpg"
    if not ip.exists():
        ip = POOL_NEW / f"{lp.stem}.jpg"
    if not ip.exists():
        miss_img += 1; continue
    staged.append((ip, lp.stem, part, boxes)); per[part] += 1
names = sorted(per)
cidx = {c: i for i, c in enumerate(names)}
print(f"산입 {len(staged)}장 · {len(names)}종 · 사진없음 {miss_img}", flush=True)

# 2) 출력 폴더
ti, tl = OUT / "images" / "train", OUT / "labels" / "train"
vi, vl = OUT / "images" / "val", OUT / "labels" / "val"
for d in (ti, tl, vi, vl):
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
for ip, stem, part, boxes in staged:
    os.link(ip, ti / f"{stem}.jpg")
    (tl / f"{stem}.txt").write_text(
        "\n".join(f"{cidx[part]} {b[0]} {b[1]} {b[2]} {b[3]}" for b in boxes) + "\n", encoding="utf-8")

# 3) 배경(격납고 543 + 조종실 24)
n_bg = 0
for bp in sorted((ROOT / "data" / "bell412" / "_negatives" / "images").glob("*.jpg")):
    os.link(bp, ti / f"bg_{bp.stem}.jpg"); n_bg += 1
print(f"배경 {n_bg}장", flush=True)

# 4) 합성 3배(부품별 실촬 수 비례 추첨 + 판넬 가림)
import sam2_autolabel as sa
weights = {cidx[c]: per[c] for c in names}
n_syn = MULT * len(staged)
made = sa._synth_augment(ti, tl, lambda m, lv="info": print("  " + m, flush=True),
                         n_syn=n_syn, class_weights=weights)
# 부품별 합성 배수 확인
syn_cnt = Counter()
for p in ti.glob("syn_*.jpg"):
    for line in (tl / f"{p.stem}.txt").read_text(encoding="utf-8").splitlines():
        syn_cnt[names[int(line.split()[0])]] += 1
worst = sorted(((syn_cnt[c] / per[c], c) for c in names))[:3]
print(f"합성 {made}장 · 목표 {n_syn} · 배수 최저 3종: " +
      ", ".join(f"{c} {r:.1f}x" for r, c in worst), flush=True)

# 5) 검증(실촬 10장당 1장, in-sample 모니터용)
reals = sorted(p for p in ti.glob("*.jpg") if not p.stem.startswith(("syn_", "bg_")))
for p in reals[::VAL_EVERY]:
    os.link(p, vi / p.name)
    shutil.copy(tl / f"{p.stem}.txt", vl / f"{p.stem}.txt")

nb = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
(OUT / "data.yaml").write_text(
    f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{nb}\n", encoding="utf-8")
print(f"완료 {round(time.time()-t0)}초 · 학습 {len(list(ti.glob('*.jpg')))}장"
      f"(실촬 {len(staged)} + 합성 {made} + 배경 {n_bg}) · 검증 {len(list(vi.glob('*.jpg')))}장", flush=True)
