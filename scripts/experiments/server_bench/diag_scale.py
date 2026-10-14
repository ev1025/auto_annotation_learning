# -*- coding: utf-8 -*-
"""학습셋 vs 평가구간의 '객체 크기 분포' 진단.

왜: 합성(copy-paste)은 부품을 화면 폭의 18~42% 로만 붙인다(sam2_autolabel._synth_augment).
    실촬도 전부 근접 촬영이다. 반면 기체 장착 구간(test4)의 gearbox 는 화면 폭의 10% 남짓이다.
    학습에서 한 번도 본 적 없는 크기라면, 클래스 간섭과 별개로 '소형 객체 미검출'이 겹친다.
사용: ./venv/bin/python scripts/experiments/diag_scale.py [--data <benchset>/data.yaml] [--class gearbox]
"""
import argparse, glob, json, os
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get("XR_BASE") or Path(__file__).resolve().parents[2])


def label_sizes(label_dir, want_idx=None):
    """YOLO 라벨 폴더 -> {클래스idx: [상대 변길이 sqrt(w*h)]} (kind 별로 분리)"""
    out = defaultdict(lambda: defaultdict(list))
    for lp in glob.glob(str(Path(label_dir) / "*.txt")):
        stem = Path(lp).stem
        kind = "synth" if stem.startswith("syn_") else ("bg" if stem.startswith("bg_") else "real")
        for ln in open(lp, encoding="utf-8"):
            f = ln.split()
            if len(f) < 5:
                continue
            ci = int(f[0])
            if want_idx is not None and ci != want_idx:
                continue
            w, h = float(f[3]), float(f[4])
            out[kind][ci].append(float(np.sqrt(w * h)))
    return out


def q(a):
    a = np.asarray(a, np.float32)
    if not len(a):
        return {}
    return {"n": int(len(a)), "p05": round(float(np.percentile(a, 5)), 4),
            "p50": round(float(np.percentile(a, 50)), 4), "p95": round(float(np.percentile(a, 95)), 4),
            "min": round(float(a.min()), 4), "max": round(float(a.max()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/data2/jinwoolee/benchset_3x/labels/train")
    ap.add_argument("--names", default="/workspace/data2/jinwoolee/benchset_3x/data.yaml")
    ap.add_argument("--cls", default="gearbox")
    a = ap.parse_args()
    names = {}
    for ln in open(a.names, encoding="utf-8"):
        s = ln.strip()
        if s and s[0].isdigit() and ":" in s:
            i, n = s.split(":", 1)
            names[int(i)] = n.strip()
    want = next((i for i, n in names.items() if n == a.cls), None)

    print(f"학습 라벨: {a.data}")
    allk = label_sizes(a.data, None)
    for kind in ("real", "synth"):
        vals = [v for d in allk[kind].values() for v in d]
        print(f"  [{kind:5}] 전체 객체 {q(vals)}")
    print(f"\n  '{a.cls}'(idx {want}) 만:")
    for kind in ("real", "synth"):
        print(f"    [{kind:5}] {q(allk[kind].get(want, []))}")

    # 평가 구간 GT
    GT = BASE / "data/bell412/gearbox/gt"
    print(f"\n평가 GT: {GT}")
    zs = defaultdict(list)
    for lp in sorted(glob.glob(str(GT / "labels/*.txt"))):
        z = Path(lp).stem.split("_")[0]
        for ln in open(lp, encoding="utf-8"):
            f = ln.split()
            if len(f) >= 5:
                zs[z].append(float(np.sqrt(float(f[3]) * float(f[4]))))
    for z in sorted(zs):
        s = q(zs[z])
        print(f"  {z}: {s}   (640 입력 기준 {round(s['p50']*640,1)}px)")

    # 학습 분포에서 test4 크기가 차지하는 위치
    tr = [v for kind in ("real", "synth") for d in allk[kind].values() for v in d]
    if tr and zs.get("test4"):
        t4 = float(np.median(zs["test4"]))
        pct = float((np.asarray(tr) < t4).mean() * 100)
        print(f"\n  ▶ test4 중앙 크기 {t4:.3f} 는 학습셋 객체 중 하위 {pct:.1f}% 지점")
        for kind in ("real", "synth"):
            v = [x for d in allk[kind].values() for x in d]
            if v:
                print(f"      {kind}: {float((np.asarray(v) < t4).mean()*100):.1f}% 가 그보다 작음"
                      f" · 최소 {min(v):.3f}")


if __name__ == "__main__":
    main()
