# -*- coding: utf-8 -*-
"""GT 라벨로 mAP 측정 — 지금 서비스 중인 모델(또는 지정 가중치)을 사람이 그린 정답과 비교한다.

대시보드의 '인식률'은 정답이 없어서 '프레임당 검출 여부'만 센 값이다. 이 스크립트는
사람이 라벨한 GT(data/bell412/<부품>/gt)로 진짜 mAP@0.5, mAP@0.5:0.95 를 낸다.

사용:
  venv/Scripts/python.exe scripts/eval_gt.py                    # 서비스 모델, GT 있는 부품 전부
  venv/Scripts/python.exe scripts/eval_gt.py --weights <best.pt>
  venv/Scripts/python.exe scripts/eval_gt.py --parts gearbox

GT 형식: gt/images/<이름>.jpg + gt/labels/<이름>.txt (YOLO 형식, class 0 = 그 부품)
결과:    results/gt_eval/<실행시각>/{data.yaml, summary.json}
"""
import argparse, json, shutil, sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "backend" / "autolearning"))
sys.path.insert(0, str(BASE / "scripts"))

PARTS_ROOT = BASE / "data" / "bell412"
IMG_EXT = {".jpg", ".jpeg", ".png"}


def gt_parts():
    """GT 폴더(images·labels 둘 다 있는)를 가진 부품 이름."""
    return sorted(d.name for d in PARTS_ROOT.iterdir()
                  if (d / "gt" / "images").is_dir() and (d / "gt" / "labels").is_dir())


def served_weights():
    """현재 서비스 중인 모델 가중치. 없으면 None."""
    import sam2_autolabel as sa
    m = sa.served_model()
    return m.get("weights") if m else None


def build_dataset(parts, names, out):
    """부품별 GT 를 한 평가셋으로 합친다. GT 라벨의 class 0 을 모델의 클래스 번호로 바꾼다.

    모델이 모르는 부품(클래스에 없음)은 건너뛴다 — 평가 대상이 아니라 미학습 부품이다.
    """
    idx = {n: i for i, n in names.items()}
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    used, skipped, n_img, n_box = [], [], 0, 0
    for part in parts:
        if part not in idx:
            skipped.append(part)
            continue
        cid = idx[part]
        gt = PARTS_ROOT / part / "gt"
        for ip in sorted(p for p in (gt / "images").iterdir() if p.suffix.lower() in IMG_EXT):
            lp = gt / "labels" / f"{ip.stem}.txt"
            if not lp.exists():                       # 라벨 없는 이미지는 제외(배경으로 오해되면 수치가 왜곡된다)
                continue
            stem = f"{part}__{ip.stem}"               # 부품이 달라도 이름이 안 겹치게
            shutil.copy(ip, out / "images" / f"{stem}{ip.suffix}")
            lines = []
            for ln in lp.read_text(encoding="utf-8").splitlines():
                f = ln.split()
                if len(f) >= 5:
                    lines.append(" ".join([str(cid)] + f[1:]))
            (out / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            n_img += 1
            n_box += len(lines)
        used.append(part)
    yaml = out / "data.yaml"
    yaml.write_text("path: {}\ntrain: images\nval: images\nnames:\n{}\n".format(
        out.as_posix(), "\n".join(f"  {i}: {n}" for i, n in sorted(names.items()))), encoding="utf-8")
    return yaml, {"parts": used, "skipped": skipped, "images": n_img, "boxes": n_box}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", help="측정할 가중치(.pt). 기본 = 현재 서비스 모델")
    ap.add_argument("--parts", nargs="*", help="측정할 부품. 기본 = GT 있는 부품 전부")
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()

    w = a.weights or served_weights()
    if not w or not Path(w).exists():
        print("가중치를 찾지 못했습니다. --weights 로 지정하세요."); return 1
    parts = a.parts or gt_parts()
    if not parts:
        print(f"GT 폴더가 없습니다: {PARTS_ROOT}/<부품>/gt/{{images,labels}}"); return 1

    from ultralytics import YOLO
    model = YOLO(str(w))
    out = BASE / "results" / "gt_eval" / datetime.now().strftime("%y%m%d_%H%M%S")
    yaml, info = build_dataset(parts, model.names, out)
    if not info["parts"]:
        print(f"이 모델의 클래스({list(model.names.values())})에 해당하는 GT 가 없습니다."); return 1
    if info["skipped"]:
        print(f"건너뜀(모델이 학습하지 않은 부품): {', '.join(info['skipped'])}")

    r = model.val(data=str(yaml), imgsz=a.imgsz, split="val", verbose=False, plots=False, project=str(out), name="val")
    per = {}
    for i, c in enumerate(r.box.ap_class_index):
        per[model.names[int(c)]] = {"map50": round(float(r.box.ap50[i]), 4),
                                    "map50_95": round(float(r.box.ap[i].mean()), 4),
                                    "precision": round(float(r.box.p[i]), 4),
                                    "recall": round(float(r.box.r[i]), 4)}
    summary = {"weights": str(w), "imgsz": a.imgsz, **info,
               "map50": round(float(r.box.map50), 4), "map50_95": round(float(r.box.map), 4),
               "per_class": per, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n가중치 : {w}")
    print(f"평가셋 : {', '.join(info['parts'])} · 이미지 {info['images']}장 · 박스 {info['boxes']}개")
    print(f"{'부품':<16}{'mAP50':>9}{'mAP50-95':>11}{'정밀도':>9}{'재현율':>9}")
    for n, v in per.items():
        print(f"{n:<16}{v['map50']:>9.3f}{v['map50_95']:>11.3f}{v['precision']:>9.3f}{v['recall']:>9.3f}")
    print(f"{'전체':<16}{summary['map50']:>9.3f}{summary['map50_95']:>11.3f}")
    print(f"\n결과: {out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
