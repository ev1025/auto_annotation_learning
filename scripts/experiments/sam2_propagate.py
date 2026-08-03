# -*- coding: utf-8 -*-
"""test_sam2.py - SAM2 영상 전파가 기어박스 '전체'를 프레임마다 일관되게 잡는지 검증.

첫 프레임에 부품 본체 양성 점 + 배경 음성 점을 주고, SAM2 로 영상 전체에 마스크를 전파.
프레임마다 마스크 오버레이 + 박스를 그려 저장 -> 육안으로 automask 방식보다 나은지 확인.

실행: ./venv/Scripts/python.exe scripts/experiments/test_sam2.py [영상stem] [프레임수]
결과: exp_sam2_test/masks/  (오버레이 이미지), report.json
"""
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch

import config
from sam2.build_sam import build_sam2_video_predictor

BASE = config.BASE_DIR
STEM = sys.argv[1] if len(sys.argv) > 1 else "gearbox2raw"
NFR = int(sys.argv[2]) if len(sys.argv) > 2 else 50
WORK = BASE / "exp_sam2_test"
CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
CKPT = BASE / "models" / "sam2" / "sam2.1_hiera_base_plus.pt"
# 첫 프레임 참조점(비율): 부품 본체 양성 4 + 배경 음성 2
POS = [(0.38, 0.60), (0.51, 0.45), (0.48, 0.38), (0.64, 0.52)]
NEG = [(0.42, 0.80), (0.64, 0.35)]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def rd(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def main():
    frames = sorted((config.DATA_DIR / "_frame_cache" / STEM).glob("*.jpg"))[:NFR]
    if not frames:
        raise SystemExit(f"프레임 없음: {STEM}")
    tmp = WORK / "frames"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for i, p in enumerate(frames):          # SAM2는 정수 이름 프레임 요구 -> 0.jpg..
        shutil.copy(p, tmp / f"{i}.jpg")
    h, w = rd(frames[0]).shape[:2]
    log(f"{STEM}: {len(frames)}프레임 ({w}x{h})")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam2_video_predictor(CFG, str(CKPT), device=dev)
    log("SAM2 로드 완료")

    with torch.inference_mode(), torch.autocast(dev, dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(tmp), offload_video_to_cpu=True,
                                     offload_state_to_cpu=True)
        pts = np.array([[rx * w, ry * h] for rx, ry in POS + NEG], dtype=np.float32)
        lbls = np.array([1] * len(POS) + [0] * len(NEG), dtype=np.int32)
        predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=1,
                                        points=pts, labels=lbls)
        log("참조점 등록, 전파 시작")

        masks = {}
        for fidx, obj_ids, logits in predictor.propagate_in_video(state):
            m = logits[0].cpu().numpy()
            if m.ndim == 3:
                m = m[0]
            masks[fidx] = m > 0.0

    out = WORK / "masks"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    areas = []
    for i, p in enumerate(frames):
        im = rd(p)
        mk = masks.get(i)
        rep_area = 0
        if mk is not None:
            if mk.shape != (h, w):
                mk = cv2.resize(mk.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            ys, xs = np.where(mk)
            if len(xs):
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                ov = im.copy(); ov[mk] = (0, 200, 0)
                im = cv2.addWeighted(ov, 0.4, im, 0.6, 0)
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 165, 255), 3)
                rep_area = float(mk.sum()) / (w * h)
        # 첫 프레임엔 참조점 표시
        if i == 0:
            for (rx, ry), lab in zip(POS + NEG, [1] * len(POS) + [0] * len(NEG)):
                cv2.circle(im, (int(rx * w), int(ry * h)), 10,
                           (0, 255, 0) if lab else (0, 0, 255), -1)
        cv2.imwrite(str(out / f"{p.stem}.jpg"), im)
        areas.append(round(rep_area, 4))

    rep = {"stem": STEM, "frames": len(frames), "mask_area_frac": areas,
           "area_mean": round(float(np.mean(areas)), 4), "area_min": min(areas), "area_max": max(areas),
           "empty_frames": sum(1 for a in areas if a == 0)}
    (WORK / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"완료: 마스크 면적 평균 {rep['area_mean']} (min {rep['area_min']} max {rep['area_max']}), 빈프레임 {rep['empty_frames']}")


if __name__ == "__main__":
    main()
