# -*- coding: utf-8 -*-
"""gearbox 1클래스 데이터셋. 36클래스 재학습(benchset_3x)과 레시피를 똑같이 맞춘다.

    실촬   gearbox 라벨 전부
    합성   실촬의 3배 · 판넬 가림(OCCL_P 0.5)
    배경   567장(사무실 제외 = 격납고 543 + 조종실 24)
    검증   실촬 10장당 1장(in-sample, 학습 진행 모니터용)

바꾸는 것은 '클래스 수' 하나뿐이다. 36클래스 희석이 test4(기체 장착)를 죽이는지 보려는 것이라,
증강·배경·합성 배수를 3배 배치와 동일하게 둬야 비교가 성립한다.
"""
import os, shutil, sys, time
from pathlib import Path

ROOT = Path("/workspace/data2/jinwoolee/xr_autolearning")
sys.path.insert(0, str(ROOT / "backend" / "autolearning"))
sys.path.insert(0, str(ROOT / "scripts"))

WORK = Path("/workspace/data2/jinwoolee")
OUT = WORK / "benchset_gearbox1"
LAB = WORK / "labels_stage" / "gearbox" / "labels"
POOL_OLD = WORK / "benchset_hangar" / "images" / "train"
POOL_NEW = WORK / "pool_new"
VAL_EVERY = 10
MULT = 3
PART = "gearbox"

t0 = time.time()
shutil.rmtree(OUT, ignore_errors=True)
ti, tl = OUT / "images" / "train", OUT / "labels" / "train"
vi, vl = OUT / "images" / "val", OUT / "labels" / "val"
for d in (ti, tl, vi, vl):
    d.mkdir(parents=True)

# 1) 실촬: gearbox 라벨 + 대응 이미지. 클래스 인덱스는 0 하나로 고정
n_real = miss = 0
for lp in sorted(LAB.glob("*.txt")):
    boxes = [f[1:5] for f in (l.split() for l in lp.read_text(encoding="utf-8").splitlines()) if len(f) == 5]
    if not boxes:
        continue
    ip = POOL_OLD / f"{lp.stem}.jpg"
    if not ip.exists():
        ip = POOL_NEW / f"{lp.stem}.jpg"
    if not ip.exists():
        miss += 1
        continue
    os.link(ip, ti / f"{lp.stem}.jpg")
    (tl / f"{lp.stem}.txt").write_text(
        "\n".join(f"0 {b[0]} {b[1]} {b[2]} {b[3]}" for b in boxes) + "\n", encoding="utf-8")
    n_real += 1
print(f"실촬 {n_real}장 산입 · 사진없음 {miss}", flush=True)
if n_real < 50:
    raise SystemExit(f"실촬이 너무 적다({n_real}). 중단")

# 2) 배경(네거티브): 라벨 파일을 쓰지 않는다 = '여기엔 아무것도 없다'
n_bg = 0
for bp in sorted((ROOT / "data" / "bell412" / "_negatives" / "images").glob("*.jpg")):
    os.link(bp, ti / f"bg_{bp.stem}.jpg")
    n_bg += 1
print(f"배경 {n_bg}장", flush=True)

# 3) 합성: 실촬의 3배. 판넬 가림은 sam2_autolabel 의 기본값(OCCL_P 0.5)을 그대로 쓴다
import sam2_autolabel as sa
made = sa._synth_augment(ti, tl, lambda m, lv="info": print("  " + m, flush=True), n_syn=MULT * n_real)

# 4) 검증(in-sample 모니터)
reals = sorted(p for p in ti.glob("*.jpg") if not p.stem.startswith(("syn_", "bg_")))
for p in reals[::VAL_EVERY]:
    os.link(p, vi / p.name)
    shutil.copy(tl / f"{p.stem}.txt", vl / f"{p.stem}.txt")

(OUT / "data.yaml").write_text(
    f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: {PART}\n",
    encoding="utf-8")
n_tr = len(list(ti.glob("*.jpg")))
print(f"완료 {round(time.time()-t0)}초 · 학습 {n_tr}장(실촬 {n_real} + 합성 {made} + 배경 {n_bg})"
      f" · 검증 {len(list(vi.glob('*.jpg')))}장 · 1클래스({PART})", flush=True)
