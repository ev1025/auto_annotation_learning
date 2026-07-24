"""scripts/labeling/register_part.py - 부품 등록 (사용자 2D 사진 -> 자동 라벨 -> 데이터셋 반입).

운영 시나리오(A브랜치): 새 부품 등록 시 부품별 사진을 업로드하면 자동 라벨링해
학습 데이터로 반입한다. 여러 부품을 업로드 목록에 쌓아두고 배치로 처리한 뒤
재학습은 마지막에 한 번만 돈다(부품마다 재학습하지 않음).

입력 구조(배치, 권장):
  uploads/
  ├─ engine_valve/   # 폴더명 = 부품명(클래스). 폴더당 부품 1종
  │  ├─ img001.jpg ...
  └─ wrench/
     └─ ...

라벨 생성 방식 2가지 (부품 폴더에 ref.txt 가 있으면 ①, 없으면 ② 폴백):
  ① 1탭 참조 매칭(권장): 사용자가 등록 화면에서 부품을 한 번 탭 -> 그 좌표를
     ref.txt("이미지파일명 x y")로 저장 -> SAM 포인트 분할로 참조 크롭 확보 ->
     전 사진의 SAM 후보를 DINOv2 유사도로 참조와 매칭. 배경이 고정된 영상
     프레임에서도 동작(실사 기어박스 영상으로 검증, 8장 기록).
  ② 상호 일관성 매칭(폴백): "다른 모든 사진에도 비슷한 것이 있는 후보 = 부품".
     배경이 사진마다 바뀌는 사진 묶음 전제. 한 장면 영상에서는 배경 오채택 위험.
공통: 임계값 이상 후보 전부 채택(다중 인스턴스) + NMS + 포함 억제. 미달 사진은 라벨 포기.

실행:
  python scripts/labeling/register_part.py --batch ./data/uploads  # 부품 여러 개 일괄
  python scripts/labeling/register_part.py --name bolt --src ./photos  # 단일 부품
  ... --dry-run                                                    # 반입 없이 검증만
반입 후: python scripts/training/train_pipeline.py  (증분 학습 1회 -> 게이트 -> 배포)
"""
import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ 공용(config 등)
import config
from data_import.dataset_utils import append_class

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MIN_AREA_FRAC = 0.005   # 등록 사진은 부품이 크게 나오므로 후보 하한을 높게
MAX_AREA_FRAC = 0.85
TOPK_PER_IMG = 8        # 사진당 상위 후보 수(크기 기준, 연산 상한)
CONSIST_TAU = 0.55      # 상호 일관성 임계값(DINOv2 코사인). 미달 사진은 라벨 포기
REF_TAU = 0.70          # 1탭 참조 매칭 임계값(참조 크롭과의 DINOv2 코사인)
MIN_ACCEPT = 5          # 부품당 최소 채택 라벨 수(미만이면 그 부품 등록 보류)


def load_models():
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    ckpts = sorted(Path(os.path.expanduser("~")).rglob("sam_vit_h*.pth"))
    if not ckpts:
        raise SystemExit("[오류] SAM 체크포인트(sam_vit_h*.pth)가 없습니다. "
                         "requirements.txt 의 부품 등록 섹션 안내대로 1회 받아두세요.")
    from segment_anything import SamPredictor
    sam = sam_model_registry["vit_h"](checkpoint=str(ckpts[0])).to(DEV)
    gen = SamAutomaticMaskGenerator(sam, points_per_side=32, min_mask_region_area=256)
    predictor = SamPredictor(sam)
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").to(DEV).eval()
    return gen, predictor, dino


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


def build_ref_embedding(src, predictor, dino, prev_dir):
    """ref.txt("이미지파일명 x y") -> SAM 포인트 분할 -> 참조 크롭 임베딩."""
    fname, rx, ry = (src / "ref.txt").read_text().split()[:3]
    im = cv2.imread(str(src / fname))
    predictor.set_image(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    masks, scores, _ = predictor.predict(point_coords=np.array([[int(rx), int(ry)]]),
                                         point_labels=np.array([1]), multimask_output=True)
    m = masks[int(np.argmax(scores))]
    ys, xs = np.where(m)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    crop_ = im[y1:y2, x1:x2]
    # 참조 확인용 미리보기(탭 분할이 엉뚱한 물체를 잡았는지 육안 점검)
    prev_dir.mkdir(exist_ok=True)
    vis = im.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 3)
    cv2.circle(vis, (int(rx), int(ry)), 12, (0, 0, 255), -1)
    cv2.imwrite(str(prev_dir / "ref_check.jpg"), vis)
    return embed(dino, [crop_])


def label_one_part(name, src, models, ref_tau=REF_TAU):
    """부품 1종 라벨링: 채택 목록 [(img_path, im, [(box, score), ...])] 반환."""
    gen, predictor, dino = models
    imgs = [p for p in sorted(src.glob("*")) if p.suffix.lower() in config.IMG_EXTS]
    if len(imgs) < MIN_ACCEPT:
        print(f"[{name}] 사진 {len(imgs)}장 (<{MIN_ACCEPT}) - 보류")
        return []

    prev_dir = src / "_preview"
    ref_emb = None
    if (src / "ref.txt").exists():
        ref_emb = build_ref_embedding(src, predictor, dino, prev_dir)
        print(f"[{name}] 1탭 참조 모드 (참조 확인: _preview/ref_check.jpg)")
    else:
        print(f"[{name}] 상호 일관성 모드 (ref.txt 없음 - 배경 고정 영상이면 오채택 위험)")

    per_img = []
    for p in imgs:
        im = cv2.imread(str(p))
        if im is None:
            continue
        boxes, crops = candidates(gen, im)
        per_img.append((p, im, boxes, embed(dino, crops) if boxes else None))

    accepted = []
    n_boxes = 0
    for i, (p, im, boxes, emb_i) in enumerate(per_img):
        if emb_i is None:
            continue
        if ref_emb is not None:
            # ① 참조 매칭: 후보 점수 = 참조 크롭과의 DINOv2 유사도
            scores = (emb_i @ ref_emb.T).squeeze(1).cpu().numpy()
            tau = ref_tau
        else:
            # ② 상호 일관성: 후보 점수 = (다른 사진들의 후보 중 최고 유사도)의 평균
            scores = torch.zeros(len(boxes), device=DEV)
            n_other = 0
            for j, (_, _, _, emb_j) in enumerate(per_img):
                if j == i or emb_j is None:
                    continue
                scores += (emb_i @ emb_j.T).max(dim=1).values
                n_other += 1
            scores = (scores / max(n_other, 1)).cpu().numpy()
            tau = CONSIST_TAU
        keep = _nms([(boxes[k], float(scores[k])) for k in range(len(boxes))
                     if scores[k] >= tau])
        if keep:
            accepted.append((p, im, keep))
            n_boxes += len(keep)

    rate = len(accepted) / max(len(per_img), 1) * 100
    print(f"[{name}] {len(per_img)}장 중 {len(accepted)}장 채택({rate:.0f}%), 박스 {n_boxes}개")

    # 미리보기(부품별 폴더에 저장, 육안 확인용)
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
    """클래스 등록(기존 번호 보존, 신규는 끝에 추가) + datasets/ 플랫 반입."""
    cid = append_class(name)
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
    ap.add_argument("--ref-tau", type=float, default=REF_TAU,
                    help="1탭 참조 매칭 임계값(부품 폴더에 ref.txt 있을 때 적용)")
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
        accepted = label_one_part(name, src, models, args.ref_tau)
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
        print("\n다음 단계(재학습은 배치당 1회): python scripts/training/train_pipeline.py")


if __name__ == "__main__":
    main()
