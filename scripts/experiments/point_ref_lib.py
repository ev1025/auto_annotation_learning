# -*- coding: utf-8 -*-
"""point_ref_lib.py - 포인트 참조 라벨링 공용 헬퍼.

SAM 후보 분할 + DINOv2 임베딩 유사도 매칭에 쓰는 순수 함수 모음.
verify/autolabel.py 가 import(load_img·embed·candidates·nms·write_label·REF_TAU·SAM_CKPT).
(구 exp_selftrain_video.py 의 실험 파이프라인 코드는 SAM2 전환으로 폐기, 헬퍼만 남김.)
"""
from pathlib import Path

import cv2
import numpy as np  # noqa: F401  (호출측 호환용)
import torch

import config

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SAM_CKPT = config.BASE_DIR / "models" / "sam" / "sam_vit_h_4b8939.pth"
RESIZE_W = 1024
REF_TAU = 0.70        # 참조 크롭과의 DINOv2 코사인 유사도 채택 임계


def load_img(p):
    """이미지 읽고 가로 RESIZE_W 이하로 축소."""
    im = cv2.imread(str(p))
    h, w = im.shape[:2]
    if w > RESIZE_W:
        im = cv2.resize(im, (RESIZE_W, int(h * RESIZE_W / w)))
    return im


@torch.no_grad()
def embed(dino, crops):
    """크롭 목록 -> DINOv2 정규화 임베딩 (N x D)."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(crops), 32):
        batch = [torch.from_numpy(cv2.cvtColor(cv2.resize(c, (224, 224)), cv2.COLOR_BGR2RGB))
                 .permute(2, 0, 1).float() / 255.0 for c in crops[i:i + 32]]
        x = (torch.stack(batch).to(DEV) - mean) / std
        e = dino(x)
        out.append(e / e.norm(dim=-1, keepdim=True))
    return torch.cat(out)


def candidates(gen, im):
    """SAM 자동 마스크 -> 후보 (box, crop). 면적 필터(너무 작거나 큰 것 제외)."""
    h, w = im.shape[:2]
    cs = []
    for m in gen.generate(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)):
        x, y, bw, bh = map(int, m["bbox"])
        frac = (bw * bh) / (w * h)
        if frac < 0.01 or frac > 0.85 or bw < 20 or bh < 20:
            continue
        cs.append((x, y, x + bw, y + bh))
    crops = [im[b[1]:b[3], b[0]:b[2]] for b in cs]
    return cs, crops


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms(items, thr=0.5, contain=0.7):
    """겹침 억제 + 더 큰 박스에 포함되는 작은 박스 제거."""
    keep = []
    for box, sc in sorted(items, key=lambda t: -t[1]):
        if any(iou(box, b) >= thr for b, _ in keep):
            continue
        keep.append((box, sc))

    def area(b):
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    def inter(a, b):
        return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return [(b, sc) for b, sc in keep
            if not any(o is not b and area(o) > area(b) and inter(b, o) / max(area(b), 1) >= contain
                       for o, _ in keep)]


def write_label(lbl_path, boxes, w, h):
    """box 목록 -> YOLO 라벨(.txt). 단일 클래스 0."""
    lines = [f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
             for (x1, y1, x2, y2), _ in boxes]
    Path(lbl_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
