# -*- coding: utf-8 -*-
"""여러 가중치를 같은 GT(사람이 그린 정답지)로 재서 나란히 비교한다.

왜 필요한가
  학습 로그의 mAP 는 train 과 val 이 같은 폴더라 0.99 로 포화된다(배운 사진으로 채점).
  모델 크기(11s vs 11m)나 설정 비교는 그 값으로 판단할 수 없다. GT 로 재야 차이가 보인다.

GT 위치 (부품별, 사람이 그린 단일클래스 라벨)
  data/bell412/<부품>/gt/images/*.jpg
  data/bell412/<부품>/gt/labels/*.txt        (첫 필드 0)

평가 방법
  - 모델의 클래스 순서에 맞춰 GT 라벨의 클래스 인덱스를 바꿔 붙인다(부품 -> 그 모델의 인덱스).
  - GT 가 있는 부품만 평가한다. GT 없는 부품(예: medicine)은 '미측정'으로 남긴다.
  - conf 임계값을 따로 주지 않는다(ultralytics 기본 0.001) -> PR 커브 전 구간으로 AP 산출.
    운영 임계값(추론서버 0.7)과 다른 값이며, 지표 규격(IoU 0.5)은 mAP50 이 그 값이다.

사용(서버)
  cd /workspace/data2/jinwoolee/xr_autolearning
  XR_BASE=$PWD CUDA_VISIBLE_DEVICES=2 ./venv/bin/python scripts/experiments/compare_models_gt.py \
      results/experiments/train_multi/260818_181913/model/best.pt \
      results/experiments/train_multi/260818_191737/model/best.pt
결과: results/experiments/compare_gt/<시각>/{summary.txt,results.json}
"""
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(os.environ.get("XR_BASE") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(BASE / "scripts"))
OUT = BASE / "results" / "experiments" / "compare_gt"


def gt_parts():
    """GT 를 가진 부품 목록."""
    return sorted(Path(p).parents[1].name for p in glob.glob(str(BASE / "data/bell412/*/gt/images")))


def build_gt(rundir, names):
    """모델의 클래스 순서(names)에 맞춘 통합 GT 폴더 + yaml 을 만든다."""
    gi, gl = rundir / "_gt/images", rundir / "_gt/labels"
    gi.mkdir(parents=True, exist_ok=True)
    gl.mkdir(parents=True, exist_ok=True)
    idx = {c: i for i, c in enumerate(names)}
    used, n = [], 0
    for part in gt_parts():
        if part not in idx:                       # 그 모델이 모르는 부품은 평가 대상이 아니다
            continue
        src = BASE / "data/bell412" / part / "gt"
        for ip in sorted(glob.glob(str(src / "images/*"))):
            ip = Path(ip)
            if ip.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            dst = gi / f"{part}__{ip.stem}{ip.suffix}"
            if not dst.exists():
                os.symlink(ip.resolve(), dst)
            lines = []
            lp = src / "labels" / f"{ip.stem}.txt"
            if lp.exists():
                for line in lp.read_text(encoding="utf-8").splitlines():
                    f = line.split()
                    if len(f) == 5:
                        lines.append(f"{idx[part]} {f[1]} {f[2]} {f[3]} {f[4]}")
            (gl / f"{part}__{ip.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            n += 1
        used.append(part)
    nb = "\n".join(f"  {i}: {c}" for i, c in enumerate(names))
    y = rundir / "gt.yaml"
    y.write_text(f"path: {(rundir / '_gt').resolve().as_posix()}\ntrain:\n  - images\nval:\n  - images\nnames:\n{nb}\n",
                 encoding="utf-8")
    return y, n, used


def main(paths):
    from ultralytics import YOLO
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    rundir = OUT / ts
    rundir.mkdir(parents=True, exist_ok=True)
    rows = []
    for wp in paths:
        w = Path(wp)
        if not w.exists():
            print(f"  건너뜀(파일 없음): {w}")
            continue
        m = YOLO(str(w))
        names = [m.names[i] for i in sorted(m.names)]
        gy, n_gt, used = build_gt(rundir / w.parent.parent.name, names)
        r = m.val(data=str(gy), imgsz=640, device=0, verbose=False,
                  project=str(rundir / "val"), name=w.parent.parent.name, exist_ok=True)
        per = {}
        try:
            for i, ci in enumerate(r.box.ap_class_index):
                per[names[int(ci)]] = round(float(r.box.ap50[i]), 4)
        except Exception:
            pass
        rows.append({"weights": str(w), "run": w.parent.parent.name, "classes": names,
                     "gt_images": n_gt, "gt_parts": used,
                     "map50": round(float(r.box.map50), 4), "map5095": round(float(r.box.map), 4),
                     "precision": round(float(r.box.mp), 4), "recall": round(float(r.box.mr), 4),
                     "per_class_ap50": per,
                     "params_M": round(sum(p.numel() for p in m.model.parameters()) / 1e6, 1)})
        print(f"  {w.parent.parent.name}: mAP50 {rows[-1]['map50']} · mAP50-95 {rows[-1]['map5095']} "
              f"· 클래스별 {per}")

    (rundir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"모델 비교 (GT 실측 · IoU 0.5 · conf 미적용) {ts}",
             f"{'run':18}{'파라미터(M)':>11}{'GT장수':>8}{'mAP50':>9}{'mAP50-95':>10}{'P':>8}{'R':>8}  클래스별 AP50"]
    for r in rows:
        lines.append(f"{r['run']:18}{r['params_M']:>11}{r['gt_images']:>8}{r['map50']:>9.4f}"
                     f"{r['map5095']:>10.4f}{r['precision']:>8.3f}{r['recall']:>8.3f}  "
                     + ", ".join(f"{k} {v}" for k, v in r["per_class_ap50"].items()))
    if rows:
        miss = [c for c in rows[0]["classes"] if c not in rows[0]["gt_parts"]]
        if miss:
            lines.append(f"미측정(GT 없음): {', '.join(miss)}")
    (rundir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nDONE {rundir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
