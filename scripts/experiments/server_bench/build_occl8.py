# -*- coding: utf-8 -*-
"""가림 강도만 올린 A/B 데이터셋: benchset_3x 와 전부 같고 OCCL_P 만 0.5 -> 0.8.

benchset_3x 를 하드링크로 복제한 뒤 합성(syn_*)만 지우고 다시 만든다.
실촬 9083 · 배경 567 · 검증 909 · 합성 장수 · 부품별 가중치가 전부 동일하므로
차이는 "합성 객체 중 몇 %를 판넬로 가렸나" 하나뿐이다.
"""
import os, shutil, subprocess, sys, time
from collections import Counter
from pathlib import Path

ROOT = Path("/workspace/data2/jinwoolee/xr_autolearning")
sys.path.insert(0, str(ROOT / "backend" / "autolearning"))
sys.path.insert(0, str(ROOT / "scripts"))

WORK = Path("/workspace/data2/jinwoolee")
SRC = WORK / "benchset_3x"
OUT = WORK / "benchset_occl8"
MULT = 3
OCCL_P_NEW = 0.8

t0 = time.time()
if OUT.exists():
    shutil.rmtree(OUT)
subprocess.run(["cp", "-al", str(SRC), str(OUT)], check=True)   # 같은 사진을 두 번 저장하지 않는다
ti, tl = OUT / "images" / "train", OUT / "labels" / "train"

# 1) 기존 합성만 제거
n_old = 0
for p in list(ti.glob("syn_*.jpg")):
    p.unlink()
    lp = tl / f"{p.stem}.txt"
    if lp.exists():
        lp.unlink()
    n_old += 1

# 2) 남은 실촬 라벨에서 부품별 장수를 세어 가중치를 복원(benchset_3x 와 같은 추첨을 쓰기 위해)
per = Counter()
n_real = 0
for lp in tl.glob("*.txt"):
    if lp.stem.startswith(("syn_", "bg_")):
        continue
    lines = [l for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        continue
    n_real += 1
    for l in lines:
        per[int(l.split()[0])] += 1
n_bg = len(list(ti.glob("bg_*.jpg")))
print(f"복제 완료 · 기존 합성 {n_old}장 제거 · 실촬 {n_real} · 배경 {n_bg} · 부품 {len(per)}종", flush=True)

# 3) 가림 확률만 올려 합성 재생성
import sam2_autolabel as sa
print(f"OCCL_P {sa.OCCL_P} -> {OCCL_P_NEW}", flush=True)
sa.OCCL_P = OCCL_P_NEW
made = sa._synth_augment(ti, tl, lambda m, lv="info": print("  " + m, flush=True),
                         n_syn=MULT * n_real, class_weights=dict(per))

y = (SRC / "data.yaml").read_text(encoding="utf-8").replace(str(SRC), str(OUT))
(OUT / "data.yaml").write_text(y, encoding="utf-8")
n_tr = len(list(ti.glob("*.jpg")))
print(f"완료 {round(time.time()-t0)}초 · 학습 {n_tr}장(실촬 {n_real} + 합성 {made} + 배경 {n_bg})"
      f" · 검증 {len(list((OUT/'images'/'val').glob('*.jpg')))}장", flush=True)
