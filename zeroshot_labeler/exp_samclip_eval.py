"""exp_samclip_eval.py - SAM(전체 분할) + CLIP 갤러리 분류 콜드스타트 평가.

아이디어: 텍스트 프롬프트(제로샷 실패 원인 = 세밀 부품을 말로 구분 불가) 대신,
  1단: SAM 이 클래스 무관하게 '물체' 후보를 전부 분할 (objectness 재현율 높음)
  2단: 각 후보 크롭을 CLIP 임베딩 -> 부품별 참조 갤러리(클래스당 K장)와
       코사인 유사도 최근접 매칭으로 분류. 유사도 임계값 미달은 배경으로 버림.

갤러리 = train 분할 정답 크롭 K=10/클래스 (운영에서는 3D 렌더 크롭으로 대체
= 수작업 0 유지). 같은 도메인 크롭이라 렌더->실사 대비 낙관적 조건임을 명시.

평가: test 분할, IoU 0.5 채점 (기존 하네스와 동일 기준).
  비교 기준: Grounded-SAM P 0.239 / R 0.418, 학습모델 오토라벨 P 0.87~0.90

실행: CUDA_VISIBLE_DEVICES=0 ./venv_zs/bin/python zeroshot_labeler/exp_samclip_eval.py
결과: eval_out/samclip_eval.json (+ 임계값 스윕 표, 미리보기)
"""
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

BASE = Path(__file__).resolve().parent.parent
TRAIN = BASE / "mechanical-parts-yolo" / "train"
TEST = BASE / "mechanical-parts-yolo" / "test"
OUT = Path(__file__).resolve().parent / "eval_out"
CLASSES = ["bearing", "bolt", "gear", "nut"]

K_GALLERY = 10          # 클래스당 참조 크롭 수
PAD = 0.08              # 크롭 시 여유 패딩 비율
MIN_AREA_FRAC = 5e-4    # 후보 최소 크기(이미지 대비)
MAX_AREA_FRAC = 0.35    # 후보 최대 크기(접시/배경 컷)
NMS_IOU = 0.5
IOU_THR = 0.5
TAUS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]  # CLIP 유사도 임계값 스윕
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def yolo_to_xyxy(line, w, h):
    f = line.split()
    c, cx, cy, bw, bh = int(f[0]), *map(float, f[1:5])
    return c, (round((cx - bw / 2) * w), round((cy - bh / 2) * h),
               round((cx + bw / 2) * w), round((cy + bh / 2) * h))


def crop(im, box, w, h):
    x1, y1, x2, y2 = box
    px, py = int((x2 - x1) * PAD), int((y2 - y1) * PAD)
    return im[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]


def load_models():
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    from transformers import CLIPModel, CLIPProcessor

    # autodistill 이 받아둔 SAM ViT-H 체크포인트 재사용
    ckpts = list(Path(os.path.expanduser("~")).rglob("sam_vit_h*.pth"))
    if not ckpts:
        raise SystemExit("[오류] SAM 체크포인트를 찾지 못했습니다 (grounded-sam 먼저 1회 실행)")
    print(f"[로드] SAM: {ckpts[0]}")
    sam = sam_model_registry["vit_h"](checkpoint=str(ckpts[0])).to(DEV)
    gen = SamAutomaticMaskGenerator(sam, points_per_side=32, min_mask_region_area=64)

    print("[로드] CLIP ViT-L/14 (첫 실행 시 ~1.7GB 다운로드)")
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEV).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return gen, clip, proc


@torch.no_grad()
def embed(clip, proc, crops):
    """BGR crop 리스트 -> 정규화 CLIP 임베딩 (배치 64)."""
    out = []
    for i in range(0, len(crops), 64):
        pil = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops[i:i + 64]]
        x = proc(images=pil, return_tensors="pt").to(DEV)
        e = clip.get_image_features(**x)
        out.append(e / e.norm(dim=-1, keepdim=True))
    return torch.cat(out)


def build_gallery(clip, proc):
    """train 정답에서 클래스당 K개 크롭 -> 갤러리 임베딩 (운영에선 렌더 크롭으로 대체)."""
    rng = np.random.RandomState(0)
    per_class = {c: [] for c in range(len(CLASSES))}
    lbls = sorted((TRAIN / "labels").glob("*.txt"))
    rng.shuffle(lbls)
    for lf in lbls:
        img_p = TRAIN / "images" / f"{lf.stem}.jpg"
        if not img_p.exists():
            continue
        im = cv2.imread(str(img_p))
        h, w = im.shape[:2]
        for line in lf.read_text().splitlines():
            c, box = yolo_to_xyxy(line, w, h)
            if len(per_class[c]) < K_GALLERY and (box[2] - box[0]) > 12 and (box[3] - box[1]) > 12:
                per_class[c].append(crop(im, box, w, h))
        if all(len(v) >= K_GALLERY for v in per_class.values()):
            break
    crops, owner = [], []
    for c, items in per_class.items():
        crops += items
        owner += [c] * len(items)
    emb = embed(clip, proc, crops)
    print(f"[갤러리] {' / '.join(f'{CLASSES[c]} {sum(1 for o in owner if o==c)}장' for c in range(4))}")
    return emb, torch.tensor(owner, device=DEV)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gen, clip, proc = load_models()
    gal_emb, gal_cls = build_gallery(clip, proc)

    imgs = sorted((TEST / "images").glob("*.jpg"))
    print(f"[평가] test {len(imgs)}장, τ 스윕 {TAUS}")

    all_preds, all_gts = [], []   # 이미지별 (preds, gt) 보관 후 τ 그리드로 일괄 채점
    t0 = time.perf_counter()
    for i, img_path in enumerate(imgs):
        im = cv2.imread(str(img_path))
        h, w = im.shape[:2]
        gt = []
        lbl = TEST / "labels" / f"{img_path.stem}.txt"
        if lbl.exists():
            gt = [yolo_to_xyxy(l, w, h) for l in lbl.read_text().splitlines() if len(l.split()) >= 5]

        masks = gen.generate(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        cands, crops_ = [], []
        for m in masks:
            x, y, bw, bh = map(int, m["bbox"])  # SAM bbox 는 float 로 옴
            frac = (bw * bh) / (w * h)
            if frac < MIN_AREA_FRAC or frac > MAX_AREA_FRAC or bw < 8 or bh < 8:
                continue
            box = (x, y, x + bw, y + bh)
            cands.append(box)
            crops_.append(crop(im, box, w, h))
        preds = []
        if cands:
            emb = embed(clip, proc, crops_)          # (N, D)
            sims = emb @ gal_emb.T                    # (N, K*4)
            for j, box in enumerate(cands):
                # 클래스별 최고 유사도 -> 최근접 클래스 + 신뢰도
                cls_sims = [sims[j][gal_cls == c].max().item() for c in range(len(CLASSES))]
                c = int(np.argmax(cls_sims))
                preds.append((c, float(cls_sims[c]), box))
        all_preds.append(sorted(preds, key=lambda p: -p[1]))
        all_gts.append(gt)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(imgs)} ({(time.perf_counter() - t0) / (i + 1):.2f}s/장)")

    # τ 그리드 채점 (탐지 1회 -> 임계값별 결과)
    sweep = []
    for tau in TAUS:
        stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASSES}
        tp_iou_sum = tp_n = 0
        for preds, gt in zip(all_preds, all_gts):
            keep = []
            for c, cf, box in preds:                  # τ 필터 + 클래스별 NMS
                if cf < tau:
                    continue
                if any(oc == c and iou(box, ob) >= NMS_IOU for oc, _, ob in keep):
                    continue
                keep.append((c, cf, box))
            used = [False] * len(gt)
            for c, cf, box in keep:
                hit, best = -1, IOU_THR
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
                    tp_n += 1
                else:
                    stats[CLASSES[c]]["fp"] += 1
            for j, (gc, _) in enumerate(gt):
                if not used[j]:
                    stats[CLASSES[gc]]["fn"] += 1
        T = {k: sum(s[k] for s in stats.values()) for k in ("tp", "fp", "fn")}
        p = T["tp"] / (T["tp"] + T["fp"]) if T["tp"] + T["fp"] else 0.0
        r = T["tp"] / (T["tp"] + T["fn"]) if T["tp"] + T["fn"] else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        sweep.append({"tau": tau, "precision": round(p, 4), "recall": round(r, 4),
                      "f1": round(f1, 4), **T,
                      "mean_tp_iou": round(tp_iou_sum / tp_n, 4) if tp_n else None,
                      "per_class": {c: {**s, "precision": round(s["tp"] / (s["tp"] + s["fp"]), 4) if s["tp"] + s["fp"] else 0.0,
                                        "recall": round(s["tp"] / (s["tp"] + s["fn"]), 4) if s["tp"] + s["fn"] else 0.0}
                                    for c, s in stats.items()}})
        print(f"  τ={tau}: P {p:.3f} / R {r:.3f} / F1 {f1:.3f}")

    report = {"engine": "sam-auto + clip-vit-l14 gallery(NN)", "n_images": len(imgs),
              "gallery_per_class": K_GALLERY, "gallery_source": "train GT crop (렌더 크롭 시뮬레이션, 동일 도메인이라 낙관적)",
              "sec_per_image": round((time.perf_counter() - t0) / len(imgs), 2),
              "sweep": sweep, "best_f1": max(sweep, key=lambda s: s["f1"])}
    (OUT / "samclip_eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(f"\n최고 F1 지점: {report['best_f1']['tau']} -> P {report['best_f1']['precision']} / R {report['best_f1']['recall']}")
    print(f"완료: {OUT / 'samclip_eval.json'}")


if __name__ == "__main__":
    main()
