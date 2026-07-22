"""exp_gallery_eval.py - SAM 후보 + 임베딩 갤러리 분류 평가 (임베더/마진 비교판).

exp_samclip_eval 의 확장: 외부 검토(제미나이) 제언 실증용.
  - 임베더 선택: --embedder clip | dinov2
    CLIP = 의미론 매칭 / DINOv2 = 미세 시각특징(질감·형상) -> 유사 부품 구분에 유리 가설
  - top1-top2 마진 규칙: 1·2위 유사도 차가 작으면(모델도 헷갈리면) 라벨 포기
  - (τ, margin) 그리드 채점으로 정밀도-재현율 운영점 표 산출

실행:
  CUDA_VISIBLE_DEVICES=0 ./venv_zs/bin/python zeroshot_labeler/exp_gallery_eval.py --embedder dinov2
결과: eval_out/gallery_eval_<embedder>.json
"""
import argparse
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

K_GALLERY = 10
PAD = 0.08
MIN_AREA_FRAC = 5e-4
MAX_AREA_FRAC = 0.35
NMS_IOU = 0.5
IOU_THR = 0.5
TAUS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]  # DINOv2 는 유사도 스케일이 낮아 넓게
MARGINS = [0.0, 0.03, 0.05]
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
    x1, y1, x2, y2 = map(int, box)
    px, py = int((x2 - x1) * PAD), int((y2 - y1) * PAD)
    return im[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]


def build_embedder(kind):
    """crop 리스트 -> 정규화 임베딩 함수 반환."""
    if kind == "clip":
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEV).eval()
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        @torch.no_grad()
        def embed(crops):
            out = []
            for i in range(0, len(crops), 64):
                pil = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops[i:i + 64]]
                x = proc(images=pil, return_tensors="pt").to(DEV)
                e = model.get_image_features(**x)
                out.append(e / e.norm(dim=-1, keepdim=True))
            return torch.cat(out)
        return embed

    # DINOv2: self-supervised 시각특징 (질감·형상). torch.hub 로 로드
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").to(DEV).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)

    @torch.no_grad()
    def embed(crops):
        out = []
        for i in range(0, len(crops), 64):
            batch = []
            for c in crops[i:i + 64]:
                rgb = cv2.cvtColor(cv2.resize(c, (224, 224)), cv2.COLOR_BGR2RGB)
                batch.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
            x = (torch.stack(batch).to(DEV) - mean) / std
            e = model(x)
            out.append(e / e.norm(dim=-1, keepdim=True))
        return torch.cat(out)
    return embed


def build_gallery(embed):
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
            if len(line.split()) < 5:
                continue
            c, box = yolo_to_xyxy(line, w, h)
            if len(per_class[c]) < K_GALLERY and (box[2] - box[0]) > 12 and (box[3] - box[1]) > 12:
                per_class[c].append(crop(im, box, w, h))
        if all(len(v) >= K_GALLERY for v in per_class.values()):
            break
    crops, owner = [], []
    for c, items in per_class.items():
        crops += items
        owner += [c] * len(items)
    print(f"[갤러리] 클래스당 {K_GALLERY}장")
    return embed(crops), torch.tensor(owner, device=DEV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", choices=["clip", "dinov2"], default="dinov2")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    ckpts = list(Path(os.path.expanduser("~")).rglob("sam_vit_h*.pth"))
    sam = sam_model_registry["vit_h"](checkpoint=str(ckpts[0])).to(DEV)
    gen = SamAutomaticMaskGenerator(sam, points_per_side=32, min_mask_region_area=64)
    print(f"[로드] SAM + 임베더 {args.embedder}")
    embed = build_embedder(args.embedder)
    gal_emb, gal_cls = build_gallery(embed)

    imgs = sorted((TEST / "images").glob("*.jpg"))
    all_preds, all_gts = [], []
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
            x, y, bw, bh = map(int, m["bbox"])
            frac = (bw * bh) / (w * h)
            if frac < MIN_AREA_FRAC or frac > MAX_AREA_FRAC or bw < 8 or bh < 8:
                continue
            cands.append((x, y, x + bw, y + bh))
            crops_.append(crop(im, (x, y, x + bw, y + bh), w, h))
        preds = []
        if cands:
            emb = embed(crops_)
            sims = emb @ gal_emb.T
            for j, box in enumerate(cands):
                cls_sims = sorted(((sims[j][gal_cls == c].max().item(), c)
                                   for c in range(len(CLASSES))), reverse=True)
                (s1, c1), (s2, _) = cls_sims[0], cls_sims[1]
                preds.append((c1, s1, s1 - s2, box))   # (클래스, 신뢰도, top1-top2 마진, 박스)
        all_preds.append(sorted(preds, key=lambda p: -p[1]))
        all_gts.append(gt)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(imgs)} ({(time.perf_counter() - t0) / (i + 1):.2f}s/장)")

    # (τ, margin) 그리드 채점
    sweep = []
    for tau in TAUS:
        for mg in MARGINS:
            stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASSES}
            for preds, gt in zip(all_preds, all_gts):
                keep = []
                for c, cf, margin, box in preds:
                    if cf < tau or margin < mg:
                        continue
                    if any(oc == c and iou(box, ob) >= NMS_IOU for oc, _, ob in keep):
                        continue
                    keep.append((c, cf, box))
                used = [False] * len(gt)
                for c, cf, box in keep:
                    hit, best = -1, IOU_THR
                    for j, (gc, gbox) in enumerate(gt):
                        if not used[j] and gc == c:
                            v = iou(box, gbox)
                            if v >= best:
                                best, hit = v, j
                    if hit >= 0:
                        used[hit] = True
                        stats[CLASSES[c]]["tp"] += 1
                    else:
                        stats[CLASSES[c]]["fp"] += 1
                for j, (gc, _) in enumerate(gt):
                    if not used[j]:
                        stats[CLASSES[gc]]["fn"] += 1
            T = {k: sum(s[k] for s in stats.values()) for k in ("tp", "fp", "fn")}
            p = T["tp"] / (T["tp"] + T["fp"]) if T["tp"] + T["fp"] else 0.0
            r = T["tp"] / (T["tp"] + T["fn"]) if T["tp"] + T["fn"] else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            sweep.append({"tau": tau, "margin": mg, "precision": round(p, 4),
                          "recall": round(r, 4), "f1": round(f1, 4), **T,
                          "per_class_p": {c: round(s["tp"] / (s["tp"] + s["fp"]), 3) if s["tp"] + s["fp"] else 0.0
                                          for c, s in stats.items()}})

    hi_p = [s for s in sweep if s["precision"] >= 0.85]
    report = {"engine": f"sam + {args.embedder} gallery(NN, margin)", "n_images": len(imgs),
              "gallery_per_class": K_GALLERY,
              "sec_per_image": round((time.perf_counter() - t0) / len(imgs), 2),
              "sweep": sweep,
              "best_f1": max(sweep, key=lambda s: s["f1"]),
              "best_recall_at_p85": max(hi_p, key=lambda s: s["recall"]) if hi_p else None}
    (OUT / f"gallery_eval_{args.embedder}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    b = report["best_f1"]
    print(f"\n최고 F1: τ{b['tau']}/m{b['margin']} -> P {b['precision']} R {b['recall']}")
    if report["best_recall_at_p85"]:
        s = report["best_recall_at_p85"]
        print(f"P>=0.85 달성 운영점: τ{s['tau']}/m{s['margin']} -> P {s['precision']} R {s['recall']}")
    else:
        print("P>=0.85 달성 운영점 없음")
    print("완료:", OUT / f"gallery_eval_{args.embedder}.json")


if __name__ == "__main__":
    main()
