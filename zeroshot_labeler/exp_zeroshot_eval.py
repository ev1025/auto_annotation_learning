"""exp_zeroshot_eval.py - 제로샷(Grounding DINO) 라벨 품질을 정답(GT)과 대조 평가.

목적: 라벨이 이미 있는 mechanical-parts test 분할(225장)을 "라벨이 없는 척"
제로샷으로 라벨링한 뒤, 숨겨둔 정답과 IoU 0.5 기준으로 채점한다.
-> 콜드스타트(모델 없음) 상황에서 Grounding DINO 부트스트랩이 쓸만한지 판단.
비교 기준: 학습된 모델의 오토라벨 정밀도 0.87~0.90 (exp_results/report_*.json)

실행(전용 venv):
  CUDA_VISIBLE_DEVICES=0 ./venv_zs/bin/python zeroshot_labeler/exp_zeroshot_eval.py
결과: zeroshot_labeler/eval_out/zeroshot_eval.json + 미리보기 8장(초록=예측, 빨강=정답)
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM

BASE = Path(__file__).resolve().parent.parent
SRC = BASE / "mechanical-parts-yolo" / "test"
OUT = Path(__file__).resolve().parent / "eval_out"

# 프롬프트 -> 클래스명. 순서는 data.yaml(bearing, bolt, gear, nut)과 동일해야 채점이 맞는다.
PROMPTS = {
    "metal ball bearing ring": "bearing",
    "metal hex bolt screw":    "bolt",
    "metal gear cog wheel":    "gear",
    "metal hex nut":           "nut",
}
CLASSES = list(PROMPTS.values())
BOX_THR = 0.35
IOU_THR = 0.5
# 후처리(v2): Grounding DINO 원시 출력의 두 가지 고질 문제 교정
TIGHT = "--no-tight" not in sys.argv  # False = DINO 원본 박스로 채점(타이트닝 효과 비교용)
NMS_IOU = 0.5        # 같은 클래스 중복 박스 제거 (DINO 출력엔 NMS 가 없음)
MAX_AREA_FRAC = 0.7  # 화면의 70% 이상을 덮는 거대 박스 제거 (배경/접시 오탐)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def postprocess(preds, w, h):
    """클래스별 NMS + 거대 박스 제거. preds = [(cls, conf, xyxy)] conf 내림차순."""
    out = []
    for c, cf, box in preds:
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area > MAX_AREA_FRAC * w * h:
            continue  # 이미지 대부분을 덮는 박스는 배경 오탐
        if any(oc == c and iou(box, ob) >= NMS_IOU for oc, _, ob in out):
            continue  # 같은 클래스 고신뢰 박스와 겹치면 중복
        out.append((c, cf, box))
    return out


def load_gt(img_path, w, h):
    """YOLO txt 정답 -> [(cls, xyxy)] (픽셀 좌표)."""
    lbl = SRC / "labels" / f"{img_path.stem}.txt"
    out = []
    if lbl.exists():
        for line in lbl.read_text().splitlines():
            f = line.split()
            if len(f) < 5:
                continue
            c, cx, cy, bw, bh = int(f[0]), *map(float, f[1:5])
            out.append((c, (round((cx - bw / 2) * w), round((cy - bh / 2) * h),
                            round((cx + bw / 2) * w), round((cy + bh / 2) * h))))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    imgs = sorted((SRC / "images").glob("*.jpg"))
    print(f"[준비] test {len(imgs)}장 / 클래스 {CLASSES} / box_thr {BOX_THR}")

    model = GroundedSAM(ontology=CaptionOntology(PROMPTS),
                        box_threshold=BOX_THR, text_threshold=0.25)

    stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASSES}
    tp_iou_sum, tp_iou_n = 0.0, 0
    t0 = time.perf_counter()
    n_preview = 0
    for i, img_path in enumerate(imgs):
        det = model.predict(str(img_path))
        im = cv2.imread(str(img_path))
        h, w = im.shape[:2]
        gt = load_gt(img_path, w, h)

        # SAM 마스크가 있으면 마스크 경계로 타이트 박스 재계산 (--no-tight 로 끄면 DINO 원본 박스)
        raw = []
        for i in range(len(det.xyxy)):
            box = tuple(map(float, det.xyxy[i]))
            if TIGHT and det.mask is not None:
                ys, xs = np.where(det.mask[i])
                if len(xs):
                    box = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
            raw.append((int(det.class_id[i]), float(det.confidence[i]), box))
        preds = sorted(raw, key=lambda p: -p[1])
        preds = postprocess(preds, w, h)

        # 신뢰도 내림차순 그리디 매칭: 같은 클래스 + IoU>=0.5 미매칭 GT 가 있으면 TP
        used = [False] * len(gt)
        for c, cf, box in preds:
            hit = -1
            best = IOU_THR
            for j, (gc, gbox) in enumerate(gt):
                if used[j] or gc != c:
                    continue
                v = iou(box, gbox)
                if v >= best:
                    best, hit = v, j
            if hit >= 0:
                used[hit] = True
                stats[CLASSES[c]]["tp"] += 1
                tp_iou_sum += best
                tp_iou_n += 1
            else:
                stats[CLASSES[c]]["fp"] += 1
        for j, (gc, _) in enumerate(gt):
            if not used[j]:
                stats[CLASSES[gc]]["fn"] += 1

        if n_preview < 8:  # 미리보기: 초록=예측, 빨강=정답
            for c, cf, (x1, y1, x2, y2) in preds:
                cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(im, f"{CLASSES[c]} {cf:.2f}", (int(x1), int(y1) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            for gc, (x1, y1, x2, y2) in gt:
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.imwrite(str(OUT / f"preview_{img_path.name}"), im)
            n_preview += 1
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(imgs)} ({(time.perf_counter() - t0) / (i + 1):.2f}s/장)")

    # 집계
    report = {"n_images": len(imgs), "box_thr": BOX_THR, "iou_thr": IOU_THR,
              "post": {"nms_iou": NMS_IOU, "max_area_frac": MAX_AREA_FRAC},
              "engine": f"grounded-sam({'tight' if TIGHT else 'dino-box'})",
              "mean_tp_iou": None,  # 아래에서 채움
              "prompts": PROMPTS, "sec_per_image": round((time.perf_counter() - t0) / len(imgs), 2),
              "per_class": {}}
    T = {"tp": 0, "fp": 0, "fn": 0}
    for c, s in stats.items():
        p = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 0.0
        r = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else 0.0
        report["per_class"][c] = {**s, "precision": round(p, 4), "recall": round(r, 4)}
        for k in T:
            T[k] += s[k]
    report["mean_tp_iou"] = round(tp_iou_sum / tp_iou_n, 4) if tp_iou_n else None
    report["micro"] = {
        "precision": round(T["tp"] / (T["tp"] + T["fp"]), 4) if T["tp"] + T["fp"] else 0.0,
        "recall": round(T["tp"] / (T["tp"] + T["fn"]), 4) if T["tp"] + T["fn"] else 0.0, **T}
    (OUT / ("zeroshot_eval.json" if TIGHT else "zeroshot_eval_dinobox.json")).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n완료: {OUT / 'zeroshot_eval.json'}")


if __name__ == "__main__":
    main()
