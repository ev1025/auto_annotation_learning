# -*- coding: utf-8 -*-
"""배경 합성 증강(누끼 → 실배경 copy-paste) — ablation 스크립트 공용 자립 헬퍼.

메인 파이프라인 sam2_autolabel._synth_augment 와 동일 로직이나, SAM1(segment_anything)
의존 없이 SAM2 이미지 예측기만 써서 서버(SAM1 미설치)에서도 임포트 가능하게 자립화.

학습셋(oi=이미지폴더, ol=라벨폴더)의 각 객체를 SAM2 마스크로 누끼 → 실배경에 회전·스케일
붙여넣기해 syn_*.jpg/txt 를 oi/ol 에 추가. 라벨 원 클래스 idx 유지.
"""
import random
import numpy as np
import cv2
import torch


def _rd(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def _bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def synth_augment(oi, ol, bgdir, cfg, ckpt, dev, log, n_syn=400):
    """oi(Path 이미지폴더)·ol(Path 라벨폴더)에 배경합성 syn_*.jpg/txt n_syn장 추가. 반환 추가장수."""
    from pathlib import Path
    oi, ol = Path(oi), Path(ol)
    random.seed(0)
    bgdir = Path(bgdir)
    bgs = [str(p) for p in bgdir.rglob("*.jpg")] if bgdir.exists() else []
    if not bgs:
        log(f"배경 이미지 없음({bgdir}) → 배경 합성 증강 생략"); return 0

    def _prep_bg(path):
        bg = _rd(path); h, w = bg.shape[:2]
        cw, ch = int(w * random.uniform(0.7, 1.0)), int(h * random.uniform(0.7, 1.0))
        x, y = random.randint(0, w - cw), random.randint(0, h - ch); bg = bg[y:y + ch, x:x + cw]
        s = 960 / max(bg.shape[:2])
        if s < 1:
            bg = cv2.resize(bg, (int(bg.shape[1] * s), int(bg.shape[0] * s)))
        if random.random() < 0.5:
            bg = cv2.flip(bg, 1)
        return np.clip(bg.astype(np.float32) * random.uniform(0.7, 1.25), 0, 255).astype(np.uint8)

    def _paste(bg, fg, x, y):
        H, W = bg.shape[:2]; h, w = fg.shape[:2]
        x0, y0 = max(0, x), max(0, y); x1, y1 = min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return None
        sub = fg[y0 - y:y1 - y, x0 - x:x1 - x]; al = sub[:, :, 3:4].astype(np.float32) / 255
        bg[y0:y1, x0:x1] = (sub[:, :, :3].astype(np.float32) * al + bg[y0:y1, x0:x1].astype(np.float32) * (1 - al)).astype(np.uint8)
        ys, xs = np.where(sub[:, :, 3] > 15)
        return None if len(xs) == 0 else (x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()), y0 + int(ys.max()))

    # 1) 누끼: 학습 프레임의 각 객체를 SAM2 마스크로 오려 RGBA + 클래스idx 보관
    log("배경 합성 증강: 객체 누끼 중...")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    pred = SAM2ImagePredictor(build_sam2(cfg, str(ckpt), device=dev))
    cuts = []
    for ip in sorted(oi.glob("*.jpg")):
        if ip.stem.startswith("syn_"):
            continue
        lp = ol / f"{ip.stem}.txt"
        if not lp.exists():
            continue
        im = _rd(ip); h, w = im.shape[:2]
        try:
            pred.set_image(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        except Exception:
            continue
        for line in lp.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            ci = int(p[0]); cx, cy, bw, bh = map(float, p[1:])
            box = np.array([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], np.float32)
            try:
                with torch.inference_mode(), torch.autocast(dev, dtype=torch.bfloat16):
                    masks, _, _ = pred.predict(box=box, multimask_output=False)
                m = masks[0]; m = (m[0] if m.ndim == 3 else m) > 0.0
            except Exception:
                continue
            bb = _bbox(m)
            if not bb or bb[2] - bb[0] < 12 or bb[3] - bb[1] < 12:
                continue
            xa, ya, xb, yb = bb
            a = cv2.GaussianBlur(m[ya:yb + 1, xa:xb + 1].astype(np.uint8) * 255, (3, 3), 0)
            cuts.append((np.dstack([im[ya:yb + 1, xa:xb + 1], a]), ci))
    del pred
    import gc; gc.collect(); torch.cuda.empty_cache()
    if not cuts:
        log("누끼 대상 없음 → 배경 합성 증강 생략"); return 0
    log(f"누끼 {len(cuts)}개 → 배경 합성 증강 생성 중...")

    # 2) 합성: 배경에 1~2개 랜덤 붙여넣기(회전·스케일), 클래스 라벨 기록
    made = 0
    for k in range(n_syn):
        bg = _prep_bg(random.choice(bgs)); H, W = bg.shape[:2]; labels = []
        for _ in range(1 if random.random() > 0.25 else 2):
            rgba, ci = random.choice(cuts)
            hh, ww = rgba.shape[:2]
            M = cv2.getRotationMatrix2D((ww / 2, hh / 2), random.uniform(-20, 20), 1.0)
            c, s = abs(M[0, 0]), abs(M[0, 1]); nw, nh = int(hh * s + ww * c), int(hh * c + ww * s)
            M[0, 2] += (nw - ww) / 2; M[1, 2] += (nh - hh) / 2
            cut = cv2.warpAffine(rgba, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
            tw = int(W * random.uniform(0.18, 0.42)); sc = tw / cut.shape[1]
            cut = cv2.resize(cut, (max(1, int(cut.shape[1] * sc)), max(1, int(cut.shape[0] * sc))))
            ch2, cw2 = cut.shape[:2]
            if cw2 >= W or ch2 >= H:
                continue
            bb = _paste(bg, cut, random.randint(0, W - cw2), random.randint(0, H - ch2))
            if not bb:
                continue
            cx = (bb[0] + bb[2]) / 2 / W; cy = (bb[1] + bb[3]) / 2 / H
            labels.append(f"{ci} {cx:.6f} {cy:.6f} {(bb[2] - bb[0]) / W:.6f} {(bb[3] - bb[1]) / H:.6f}")
        if labels:
            cv2.imwrite(str(oi / f"syn_{k:05d}.jpg"), bg)
            (ol / f"syn_{k:05d}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
            made += 1
    log(f"배경 합성 증강 완료: {made}장 추가")
    return made
