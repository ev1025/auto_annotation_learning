# -*- coding: utf-8 -*-
"""exp_selftrain_video.py - '포인트 참조(방법 7) + self-training(방법 1)' 결합 실증.

가설: 라벨 없는 영상 1개에서, 포인트 참조로 만든 라벨(방법 7)로 YOLO 를 학습한 뒤,
      그 모델이 '방법 7이 놓친 프레임'을 예측해 conf 높은 것만 다시 학습하면
      (self-training) 재현율이 더 오르는가? (미라벨 풀 = 같은 영상의 나머지 프레임)

단계:
  1) genlabels : gearbox2(193프레임)에 포인트 참조로 라벨 생성
                 → 채택 프레임 = seed / 미채택 프레임 = pool
  2) round1    : 순수 YOLO(yolo26s)에 seed 라벨만 학습
  3) selftrain : round1 모델이 pool 예측 → conf>=0.6 채택 → pseudo 라벨
  4) round2    : 순수 YOLO 에 [seed + pseudo] 학습
  5) eval      : gearbox1(49프레임, 미학습)에서 검출 프레임 비율(재현율) 비교
결과: exp_selftrain/report.json

실행: ./venv/Scripts/python.exe scripts/experiments/exp_selftrain_video.py
"""
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch

import config

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SAM_CKPT = config.BASE_DIR / "models" / "sam" / "sam_vit_h_4b8939.pth"
REG_DIR = config.DATA_DIR / "gearbox_register_trial2" / "gearbox"   # 등록 영상(gearbox2) 193
VAL_DIR = config.DATA_DIR / "gearbox_register_trial1" / "gearbox"   # 검증 영상(gearbox1) 49
WORK = config.BASE_DIR / "exp_selftrain"
RESIZE_W = 1024
PPS = 16
REF_TAU = 0.70        # 참조 크롭과의 DINOv2 유사도 채택 임계 (방법 7과 동일)
REF_POINT = (0.44, 0.30)   # 첫 프레임에서 기어박스 위 점(비율 좌표) = 사람이 찍은 1점
SELF_CONF = 0.6       # self-training 채택 신뢰도 (방법 1과 동일)
VAL_CONF = 0.4        # 검증 시 검출 판정 신뢰도 (원 실험과 동일)
EPOCHS = 100


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_img(p):
    im = cv2.imread(str(p))
    h, w = im.shape[:2]
    if w > RESIZE_W:
        im = cv2.resize(im, (RESIZE_W, int(h * RESIZE_W / w)))
    return im


# ---------- DINOv2 임베딩 ----------
@torch.no_grad()
def embed(dino, crops):
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
    """SAM 자동 마스크 -> 후보 (box, crop). 등록 프레임은 부품이 크므로 면적 필터."""
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
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms(items, thr=0.5, contain=0.7):
    keep = []
    for box, sc in sorted(items, key=lambda t: -t[1]):
        if any(iou(box, b) >= thr for b, _ in keep):
            continue
        keep.append((box, sc))

    def area(b):
        return max(0, b[2]-b[0]) * max(0, b[3]-b[1])

    def inter(a, b):
        return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return [(b, sc) for b, sc in keep
            if not any(o is not b and area(o) > area(b) and inter(b, o)/max(area(b), 1) >= contain
                       for o, _ in keep)]


def write_label(lbl_path, boxes, w, h):
    lines = [f"0 {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}"
             for (x1, y1, x2, y2), _ in boxes]
    lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ==================== 1) 포인트 참조 라벨 생성 ====================
def stage_genlabels():
    from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
    log("SAM/DINOv2 로드")
    sam = sam_model_registry["vit_h"](checkpoint=str(SAM_CKPT)).to(DEV)
    gen = SamAutomaticMaskGenerator(sam, points_per_side=PPS, min_mask_region_area=256)
    predictor = SamPredictor(sam)
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", verbose=False).to(DEV).eval()

    frames = sorted(REG_DIR.glob("*.jpg"))
    # 참조 크롭: 첫 프레임의 지정 점(REF_POINT) 포인트 분할
    im0 = load_img(frames[0])
    predictor.set_image(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
    px, py = int(REF_POINT[0]*im0.shape[1]), int(REF_POINT[1]*im0.shape[0])
    masks, scores, _ = predictor.predict(point_coords=np.array([[px, py]]),
                                         point_labels=np.array([1]), multimask_output=True)
    m = masks[int(np.argmax(scores))]
    ys, xs = np.where(m)
    ref_crop = im0[ys.min():ys.max()+1, xs.min():xs.max()+1]
    ref_emb = embed(dino, [ref_crop])
    log(f"참조 확보: 점({px},{py}) -> 크롭 {ref_crop.shape[1]}x{ref_crop.shape[0]}")

    seed_dir = WORK / "seed"; pool_dir = WORK / "pool"
    for d in (seed_dir/"images", seed_dir/"labels", pool_dir/"images"):
        d.mkdir(parents=True, exist_ok=True)

    n_seed = n_pool = 0
    for i, p in enumerate(frames):
        im = load_img(p)
        h, w = im.shape[:2]
        boxes, crops = candidates(gen, im)
        keep = []
        if crops:
            emb = embed(dino, crops)
            sims = (emb @ ref_emb.T).squeeze(1).cpu().numpy()
            keep = nms([(boxes[k], float(sims[k])) for k in range(len(boxes)) if sims[k] >= REF_TAU])
        stem = p.stem
        if keep:
            cv2.imwrite(str(seed_dir/"images"/f"{stem}.jpg"), im)
            write_label(seed_dir/"labels"/f"{stem}.txt", keep, w, h)
            n_seed += 1
        else:
            cv2.imwrite(str(pool_dir/"images"/f"{stem}.jpg"), im)
            n_pool += 1
        if (i+1) % 20 == 0:
            log(f"  {i+1}/{len(frames)} (seed {n_seed} / pool {n_pool})")

    del sam, gen, predictor, dino
    gc.collect(); torch.cuda.empty_cache()
    log(f"라벨 생성 완료: seed {n_seed} / pool {n_pool}")
    return {"reg_frames": len(frames), "seed": n_seed, "pool": n_pool}


# ==================== 데이터셋 yaml ====================
def write_yaml(name, train_img_dirs):
    y = WORK / f"{name}.yaml"
    dirs = "\n".join(f"  - {d.resolve().as_posix()}" for d in train_img_dirs)
    y.write_text(f"path: {WORK.resolve().as_posix()}\ntrain:\n{dirs}\nval:\n{dirs}\n"
                 f"names:\n  0: gearbox\n", encoding="utf-8")
    return y


def train(name, data_yaml):
    from ultralytics import YOLO
    model = YOLO(config.PRETRAINED)
    model.train(data=str(data_yaml), epochs=EPOCHS, imgsz=640, batch=8, device=0,
                project=str(WORK/"runs"), name=name, exist_ok=True, verbose=False, plots=False)
    gc.collect(); torch.cuda.empty_cache()
    return WORK/"runs"/name/"weights"/"best.pt"


def presence_recall(weights, conf):
    """검증 프레임 중 기어박스를 1개 이상 검출한 프레임 비율 (기어박스는 전 프레임 존재)."""
    from ultralytics import YOLO
    model = YOLO(str(weights))
    frames = sorted(VAL_DIR.glob("*.jpg"))
    hit = 0
    for p in frames:
        r = model.predict(source=str(p), conf=conf, imgsz=640, verbose=False)[0]
        if len(r.boxes) > 0:
            hit += 1
    gc.collect(); torch.cuda.empty_cache()
    return hit, len(frames)


# ==================== 3) self-training ====================
def stage_selftrain(round1_weights):
    from ultralytics import YOLO
    model = YOLO(str(round1_weights))
    pool = sorted((WORK/"pool"/"images").glob("*.jpg"))
    seed_lbl = WORK/"seed"/"labels"
    added = 0
    for p in pool:
        im = cv2.imread(str(p)); h, w = im.shape[:2]
        r = model.predict(source=str(p), conf=SELF_CONF, imgsz=640, verbose=False)[0]
        if len(r.boxes) == 0:
            continue
        # pool 프레임을 seed 폴더로 승격(이미지+pseudo 라벨)
        cv2.imwrite(str(WORK/"seed"/"images"/f"{p.stem}.jpg"), im)
        lines = []
        for b in r.boxes.xyxyn.cpu().numpy():
            x1, y1, x2, y2 = b
            lines.append(f"0 {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {(x2-x1):.6f} {(y2-y1):.6f}")
        (seed_lbl/f"{p.stem}.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
        added += 1
    gc.collect(); torch.cuda.empty_cache()
    log(f"self-training pseudo 라벨: pool {len(pool)}장 중 {added}장 채택(conf>={SELF_CONF})")
    return {"pool": len(pool), "pseudo_added": added}


def main():
    WORK.mkdir(exist_ok=True)
    rep = {}
    log("=== 1. 포인트 참조 라벨 생성 ===")
    rep["genlabels"] = stage_genlabels()

    log("=== 2. round1 학습 (seed 라벨만) ===")
    w1 = train("round1", write_yaml("round1", [WORK/"seed"/"images"]))
    hit1, tot = presence_recall(w1, VAL_CONF)
    rep["round1"] = {"seed_labels": rep["genlabels"]["seed"],
                     "val_detect": f"{hit1}/{tot}", "val_recall": round(hit1/tot, 4)}
    log(f"round1 검증 재현율 {hit1}/{tot}")

    log("=== 3. self-training pseudo 라벨 ===")
    rep["selftrain"] = stage_selftrain(w1)

    log("=== 4. round2 학습 (seed + pseudo) ===")
    w2 = train("round2", write_yaml("round2", [WORK/"seed"/"images"]))
    hit2, _ = presence_recall(w2, VAL_CONF)
    rep["round2"] = {"train_labels": rep["genlabels"]["seed"] + rep["selftrain"]["pseudo_added"],
                     "val_detect": f"{hit2}/{tot}", "val_recall": round(hit2/tot, 4)}
    log(f"round2 검증 재현율 {hit2}/{tot}")

    rep["conclusion"] = {"round1_recall": rep["round1"]["val_recall"],
                         "round2_recall": rep["round2"]["val_recall"],
                         "delta": round(rep["round2"]["val_recall"] - rep["round1"]["val_recall"], 4)}
    (WORK/"report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    log("=== 완료 ===")
    log(json.dumps(rep["conclusion"], ensure_ascii=False))


if __name__ == "__main__":
    main()
