# -*- coding: utf-8 -*-
"""진단·대책 실험용 데이터셋 변형본을 benchset_3x 에서 파생시킨다(하드링크라 디스크를 안 먹는다).

기준셋 benchset_3x = 실촬 9,083 + 합성 26,820(74%) + 배경 567 · 36클래스.
여기서 '한 가지만' 바꾼 셋을 만들어야 원인이 갈린다.

  real     합성 전부 제거(실촬 + 배경만).            <- 진단 3: 합성 경계 아티팩트가 숏컷인가
  merged   이미지·박스는 그대로, 클래스만 1종으로 병합. <- Plan B 1단계 검출기 / 클래스 간섭 검증
  fewshot  base 모델 파인튜닝용. 기체 장착 사진(test3 GT 10장)을 소량 + 원 데이터 리플레이.
                                                    <- Plan A 도메인 적응

사용: ./venv/bin/python scripts/experiments/build_variants.py real merged fewshot
"""
import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

WORK = Path("/workspace/data2/jinwoolee")
SRC = WORK / "benchset_3x"
ROOT = Path(os.environ.get("XR_BASE") or Path(__file__).resolve().parents[2])


def read_names(yml):
    names = {}
    for ln in Path(yml).read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if s and s[0].isdigit() and ":" in s:
            i, n = s.split(":", 1)
            names[int(i)] = n.strip()
    return names


def link(src, dst):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy(src, dst)


def fresh(out):
    for sub in ("images/train", "labels/train", "images/val", "labels/val"):
        d = out / sub
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)


def write_yaml(out, names):
    nb = "\n".join(f"  {i}: {n}" for i, n in sorted(names.items()))
    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{nb}\n", encoding="utf-8")


# ---------------- real: 합성 제거 ----------------
def build_real(names):
    out = WORK / "benchset_real"
    fresh(out)
    n_real = n_bg = 0
    for ip in sorted((SRC / "images/train").glob("*.jpg")):
        if ip.stem.startswith("syn_"):
            continue                                   # 합성만 뺀다
        link(ip, out / "images/train" / ip.name)
        lp = SRC / "labels/train" / f"{ip.stem}.txt"
        if lp.exists():
            link(lp, out / "labels/train" / lp.name); n_real += 1
        else:
            n_bg += 1                                  # 라벨 없는 이미지 = 배경(네거티브)
    for ip in sorted((SRC / "images/val").glob("*.jpg")):
        link(ip, out / "images/val" / ip.name)
        link(SRC / "labels/val" / f"{ip.stem}.txt", out / "labels/val" / f"{ip.stem}.txt")
    write_yaml(out, names)
    print(f"[real] 학습 {n_real + n_bg}장(실촬 {n_real} + 배경 {n_bg}) · "
          f"검증 {len(list((out/'images/val').glob('*.jpg')))}장 · {len(names)}클래스 -> {out}")


# ---------------- merged: 클래스 1종 병합 ----------------
def build_merged(names):
    out = WORK / "benchset_merged"
    fresh(out)
    n = 0
    for split in ("train", "val"):
        for ip in sorted((SRC / f"images/{split}").glob("*.jpg")):
            link(ip, out / f"images/{split}" / ip.name)
            lp = SRC / f"labels/{split}" / f"{ip.stem}.txt"
            if not lp.exists():
                continue                               # 배경은 라벨 없이 그대로
            lines = []
            for ln in lp.read_text(encoding="utf-8").splitlines():
                f = ln.split()
                if len(f) == 5:
                    lines.append("0 " + " ".join(f[1:]))   # 클래스만 0 으로
            if lines:
                (out / f"labels/{split}" / f"{ip.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
                n += 1
    write_yaml(out, {0: "part"})
    print(f"[merged] 라벨 {n}장을 1클래스(part)로 병합 · "
          f"학습 {len(list((out/'images/train').glob('*.jpg')))}장 -> {out}")


# ---------------- fewshot: 장착 도메인 소량 + 리플레이 ----------------
def build_fewshot(names, shots_zone="test3", n_replay=4000, oversample=20, seed=0):
    """test3(기체 장착) 사람 GT 10장을 소량 학습 자료로 쓴다. 평가(test4·a_test)와 겹치지 않는다.

    소량만 그대로 파인튜닝하면 나머지 35종을 잊는다(catastrophic forgetting). 그래서
    원 학습셋에서 n_replay 장을 무작위로 섞어 넣는다(리플레이).
    """
    import cv2
    out = WORK / "benchset_fewshot"
    fresh(out)
    rnd = random.Random(seed)
    idx = {v: k for k, v in names.items()}
    gt = ROOT / "data/bell412/gearbox/gt"
    shots = 0
    for ip in sorted(gt.glob(f"images/{shots_zone}_*.jpg")):
        lp = gt / "labels" / f"{ip.stem}.txt"
        if not lp.exists():
            continue
        body = []
        for ln in lp.read_text(encoding="utf-8").splitlines():
            f = ln.split()
            if len(f) >= 5:
                body.append(f"{idx['gearbox']} " + " ".join(f[1:5]))
        if not body:
            continue
        for k in range(oversample):                    # 10장뿐이라 배치에 자주 들어오게 복제
            nm = f"shot_{ip.stem}_{k:02d}"
            link(ip, out / "images/train" / f"{nm}.jpg")
            (out / "labels/train" / f"{nm}.txt").write_text("\n".join(body) + "\n", encoding="utf-8")
        shots += 1
    pool = sorted((SRC / "images/train").glob("*.jpg"))
    rnd.shuffle(pool)
    for ip in pool[:n_replay]:
        link(ip, out / "images/train" / ip.name)
        lp = SRC / "labels/train" / f"{ip.stem}.txt"
        if lp.exists():
            link(lp, out / "labels/train" / lp.name)
    for ip in sorted((SRC / "images/val").glob("*.jpg"))[::4]:
        link(ip, out / "images/val" / ip.name)
        link(SRC / "labels/val" / f"{ip.stem}.txt", out / "labels/val" / f"{ip.stem}.txt")
    write_yaml(out, names)
    print(f"[fewshot] 장착샷 {shots}장 x{oversample} + 리플레이 {min(n_replay, len(pool))}장 = "
          f"{len(list((out/'images/train').glob('*.jpg')))}장 -> {out}")
    print(f"          (평가 test4·a_test 와 겹치지 않음 · {shots_zone} 는 이 실험에서 평가 제외)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="+", choices=["real", "merged", "fewshot"])
    a = ap.parse_args()
    names = read_names(SRC / "data.yaml")
    assert len(names) == 36, f"기준셋 클래스 수가 이상하다: {len(names)}"
    for w in a.which:
        {"real": build_real, "merged": build_merged, "fewshot": build_fewshot}[w](names)


if __name__ == "__main__":
    main()
