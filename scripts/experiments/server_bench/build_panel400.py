# -*- coding: utf-8 -*-
"""배경984 + 합성 부품당 400장(판넬 가림) 데이터셋을 만든다.

기존 benchset_hangar(실촬 7640 + 배경 984 + 검증 1324)를 하드링크로 복제해
합성만 새로 만든다. 실촬·배경·검증 분할이 앞선 배치와 완전히 같아 합성 효과만 분리된다.
"""
import os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path("/workspace/data2/jinwoolee/xr_autolearning")
sys.path.insert(0, str(ROOT / "backend" / "autolearning"))
sys.path.insert(0, str(ROOT / "scripts"))

SRC = Path("/workspace/data2/jinwoolee/benchset_hangar")
OUT = Path("/workspace/data2/jinwoolee/benchset_panel400")
N_SYN = 14400            # 36부품 x 400장

t0 = time.time()
if OUT.exists():
    shutil.rmtree(OUT)
# 하드링크 복제: 같은 사진을 두 번 저장하지 않는다
subprocess.run(["cp", "-al", str(SRC), str(OUT)], check=True)
ti, tl = OUT / "images" / "train", OUT / "labels" / "train"
old = [p for p in ti.glob("syn_*.jpg")]
for p in old:
    p.unlink()
    lp = tl / f"{p.stem}.txt"
    if lp.exists():
        lp.unlink()
print(f"복제 완료 · 기존 합성 {len(old)}장 제거 · 남은 학습 {len(list(ti.glob('*.jpg')))}장", flush=True)

import sam2_autolabel as sa
made = sa._synth_augment(ti, tl, lambda m, lv="info": print("  " + m, flush=True), n_syn=N_SYN)
n_tr = len(list(ti.glob("*.jpg")))
n_bg = len(list(ti.glob("bg_*.jpg")))
n_syn = len(list(ti.glob("syn_*.jpg")))
# data.yaml 의 path 만 새 폴더로
y = (SRC / "data.yaml").read_text(encoding="utf-8").replace(str(SRC), str(OUT))
(OUT / "data.yaml").write_text(y, encoding="utf-8")
print(f"완료 {round(time.time()-t0)}초 · 학습 {n_tr}장 (실촬 {n_tr-n_bg-n_syn} + 합성 {n_syn} + 배경 {n_bg})"
      f" · 검증 {len(list((OUT/'images'/'val').glob('*.jpg')))}장", flush=True)
