"""0_register_part.py - 부품 등록 (사용자 2D 사진 -> 자동 라벨 -> 데이터셋 반입).

운영 시나리오(A브랜치): 새 부품 등록 시 부품별 사진을 업로드하면 자동 라벨링해
학습 데이터로 반입한다. 여러 부품을 업로드 목록에 쌓아두고 배치로 처리한 뒤
재학습은 마지막에 한 번만 돈다(부품마다 재학습하지 않음).

입력 구조(배치, 권장):
  uploads/
  ├─ engine_valve/   # 폴더명 = 부품명(클래스). 폴더당 부품 1종
  │  ├─ img001.jpg ...
  └─ wrench/
     └─ ...

핵심 아이디어(상호 일관성 매칭): 같은 폴더의 사진들에는 '그 부품'이 반드시 매 장
등장하고 배경 물체는 사진마다 달라진다. 따라서
  1) SAM 이 각 사진의 물체 후보를 분할하고
  2) DINOv2 임베딩으로 "다른 모든 사진에도 비슷한 것이 있는 후보"를 고르면
  3) 그것이 등록 대상 부품이다 (클래스 분류 불필요 -> 제로샷 실험의 약점 제거)
일관성 τ 이상인 후보를 전부 채택(한 사진에 같은 부품이 여러 개 찍혀도 모두 라벨,
빠뜨리면 그 부품이 '배경'으로 학습돼 유해). τ 미달 사진은 라벨 포기(고순도 우선).

실행:
  python scripts/0_register_part.py --batch ./uploads              # 부품 여러 개 일괄
  python scripts/0_register_part.py --name bolt --src ./photos     # 단일 부품
  ... --dry-run                                                    # 반입 없이 검증만
반입 후: python scripts/2_train_pipeline.py  (증분 학습 1회 -> 게이트 -> 배포)
"""
import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

import config
from dataset_utils import normalize_names, register_classes

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MIN_AREA_FRAC = 0.005   # 등록 사진은 부품이 크게 나오므로 후보 하한을 높게
MAX_AREA_FRAC = 0.85
TOPK_PER_IMG = 8        # 사진당 상위 후보 수(크기 기준, 연산 상한)
CONSIST_TAU = 0.55      # 상호 일관성 임계값(DINOv2 코사인). 미달 사진은 라벨 포기
MIN_ACCEPT = 5          # 부품당 최소 채택 라벨 수(미만이면 그 부품 등록 보류)


def load_models():
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    ckpts = sorted(Path(os.path.expanduser("~")).rglob("sam_vit_h*.pth"))
    if not ckpts:
        raise SystemExit("[오류] SAM 체크포인트(sam_vit_h*.pth)가 없습니다. "
                         "zeroshot_labeler/requirements.txt 안내대로 1회 받아두세요.")
    sam = sam_model_registry["vit_h"](checkpoint=str(ckpts[0])).to(DEV)
    gen = SamAutomaticMaskGenerator(sam, points_per_side=32, min_mask_region_area=256)
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").to(DEV).eval()
    return gen, dino


@torch.no_grad()
def embed(dino, crops):
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(crops), 64):
        batch = [torch.from_numpy(cv2.cvtColor(cv2.resize(c, (224, 224)), cv2.COLOR_BGR2RGB))
                 .permute(2, 0, 1).float() / 255.0 for c in crops[i:i + 64]]
        x = (torch.stack(batch).to(DEV) - mean) / std
        e = dino(x)
        out.append(e / e.norm(dim=-1, keepdim=True))
    return torch.cat(out)


def candidates(gen, im):
    """SAM 후보 -> (box, crop). 등록 사진 특성(부품이 큼)에 맞춘 필터."""
    h, w = im.shape[:2]
    cands = []
    for m in gen.generate(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)):
        x, y, bw, bh = map(int, m["bbox"])
        frac = (bw * bh) / (w * h)
        if frac < MIN_AREA_FRAC or frac > MAX_AREA_FRAC or bw < 16 or bh < 16:
            continue
        cands.append(((x, y, x + bw, y + bh), frac))
    cands.sort(key=lambda c: -c[1])           # 큰 것 우선
    boxes = [c[0] for c in cands[:TOPK_PER_IMG]]
    crops = [im[b[1]:b[3], b[0]:b[2]] for b in boxes]
    return boxes, crops


def _nms(items, thr=0.5, contain_thr=0.7):
    """items = [(box, score)]. ① 점수순 그리디 NMS ② 포함 억제.

    포함 억제: SAM 은 물체를 계층적으로 쪼개(볼트 전체 + 나사산만) 부분-전체가
    동시에 후보로 남는다. IoU 는 낮아 NMS 를 통과하므로, 더 큰 박스 안에
    contain_thr 이상 들어가는 작은 박스는 중복 라벨로 보고 제거한다.
    """
    keep = []
    for box, sc in sorted(items, key=lambda t: -t[1]):
        if any(_iou(box, b) >= thr for b, _ in keep):
            continue
        keep.append((box, sc))

    def area(b):
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    def inter(a, b):
        return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

    return [(b, sc) for b, sc in keep
            if not any(o is not b and area(o) > area(b) and inter(b, o) / max(area(b), 1) >= contain_thr
                       for o, _ in keep)]


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def label_one_part(name, src, models):
    """부품 1종 라벨링: 채택 목록 [(img_path, im, [(box, score), ...])] 반환."""
    gen, dino = models
    imgs = [p for p in sorted(src.glob("*")) if p.suffix.lower() in config.IMG_EXTS]
    if len(imgs) < MIN_ACCEPT:
        print(f"[{name}] 사진 {len(imgs)}장 (<{MIN_ACCEPT}) - 보류")
        return []

    per_img = []
    for p in imgs:
        im = cv2.imread(str(p))
        if im is None:
            continue
        boxes, crops = candidates(gen, im)
        per_img.append((p, im, boxes, embed(dino, crops) if boxes else None))

    # 상호 일관성: 후보 점수 = (다른 사진들의 후보 중 최고 유사도)의 평균.
    # τ 이상 후보 전부 채택(같은 부품 여러 개 대응) + NMS 로 중복 제거.
    accepted = []
    n_boxes = 0
    for i, (p, im, boxes, emb_i) in enumerate(per_img):
        if emb_i is None:
            continue
        scores = torch.zeros(len(boxes), device=DEV)
        n_other = 0
        for j, (_, _, _, emb_j) in enumerate(per_img):
            if j == i or emb_j is None:
                continue
            scores += (emb_i @ emb_j.T).max(dim=1).values
            n_other += 1
        scores = (scores / max(n_other, 1)).cpu().numpy()
        keep = _nms([(boxes[k], float(scores[k])) for k in range(len(boxes))
                     if scores[k] >= CONSIST_TAU])
        if keep:
            accepted.append((p, im, keep))
            n_boxes += len(keep)

    rate = len(accepted) / max(len(per_img), 1) * 100
    print(f"[{name}] {len(per_img)}장 중 {len(accepted)}장 채택({rate:.0f}%), 박스 {n_boxes}개")

    # 미리보기(부품별 폴더에 저장, 육안 확인용)
    prev_dir = src / "_preview"
    prev_dir.mkdir(exist_ok=True)
    for p, im, keep in accepted[:8]:
        vis = im.copy()
        for (x1, y1, x2, y2), sc in keep:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{name} {sc:.2f}", (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(str(prev_dir / f"reg_{p.name}"), vis)
    return accepted


def ingest(name, accepted):
    """클래스 등록(가드) + datasets/ 플랫 반입. 반입 장수 반환."""
    class_names = register_classes([name])
    names = normalize_names(class_names)
    cid = [k for k, v in names.items() if v == name][0]
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    config.LABELS_DIR.mkdir(parents=True, exist_ok=True)
    for p, im, keep in accepted:
        h, w = im.shape[:2]
        stem = f"{name}_{p.stem}"
        shutil.copy2(p, config.IMAGES_DIR / f"{stem}{p.suffix.lower()}")
        lines = [f"{cid} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                 f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
                 for (x1, y1, x2, y2), _ in keep]
        (config.LABELS_DIR / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(accepted)


def main():
    ap = argparse.ArgumentParser(description="부품 등록: 사진 -> 자동 라벨 -> 데이터셋 반입(배치)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--batch", help="업로드 목록 폴더 (하위 폴더명 = 부품명)")
    g.add_argument("--name", help="단일 부품명(--src 와 함께)")
    ap.add_argument("--src", help="단일 부품 사진 폴더")
    ap.add_argument("--dry-run", action="store_true", help="반입 없이 라벨 품질 검증만")
    args = ap.parse_args()

    if args.batch:
        jobs = [(d.name, d) for d in sorted(Path(args.batch).resolve().iterdir()) if d.is_dir()
                and not d.name.startswith("_")]
    else:
        if not args.src:
            raise SystemExit("[오류] --name 은 --src 와 함께 사용")
        jobs = [(args.name, Path(args.src).resolve())]
    if not jobs:
        raise SystemExit("[오류] 처리할 부품 폴더가 없습니다.")
    print(f"[배치] 부품 {len(jobs)}종: {[n for n, _ in jobs]}")

    models = load_models()   # 모델은 배치 전체에 1회만 로드
    summary = []
    for name, src in jobs:   # 내부적으로 부품 1종씩 순차 처리
        accepted = label_one_part(name, src, models)
        if len(accepted) < MIN_ACCEPT:
            summary.append((name, len(accepted), "보류(라벨 부족)"))
            continue
        if args.dry_run:
            summary.append((name, len(accepted), "dry-run"))
        else:
            summary.append((name, ingest(name, accepted), "반입"))

    print("\n=== 등록 요약 ===")
    for name, n, status in summary:
        print(f"  {name:<20} {n:>4}장  {status}")
    if not args.dry_run and any(s == "반입" for _, _, s in summary):
        print("\n다음 단계(재학습은 배치당 1회): python scripts/2_train_pipeline.py")


if __name__ == "__main__":
    main()
