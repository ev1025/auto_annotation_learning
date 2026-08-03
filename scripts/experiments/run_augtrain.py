# -*- coding: utf-8 -*-
"""서버용 증강훈련 파이프라인(포터블): 라벨 + SAM2 + 실배경 → 누끼 → copy-paste 합성 → cp_mix_aug 학습.
BASE는 레포 루트 자동 감지(scripts/experiments/ 에 두는 전제). 로컬/서버 공용.
사용: python scripts/experiments/run_augtrain.py [--smoke] [--labels <train폴더>]
윈도우 multiprocessing 대비 __main__ 가드 필수."""
import os, sys, glob, json, random, argparse
from datetime import datetime
BASE = os.environ.get("XR_BASE") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, BASE + "/scripts"); sys.path.insert(0, BASE + "/scripts/verify")
import cv2, numpy as np
import sam2_autolabel as sa   # _img_predictor, _rd, _bbox, free_sam2, DEV 재사용

random.seed(0); np.random.seed(0)
LABELS = BASE + "/results/gearbox/260731_102307/train"                  # 재탭 라벨(실촬영)
CUT = BASE + "/data/bell412/gearbox/cutouts"
SYN = BASE + "/data/bell412/gearbox/synthetic"
BGS = glob.glob(BASE + "/data/bell412/backgrounds/hangar/*.jpg") + glob.glob(BASE + "/data/bell412/backgrounds/factory/*.jpg")
OCCL = [o for o in (cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in glob.glob(BASE + "/data/bell412/occluders/*.png")) if o is not None and o.ndim == 3 and o.shape[2] == 4]
AUG = dict(erasing=0.5, scale=0.9, mixup=0.15)


def yolo_xyxy(txt, w, h):
    out = []
    for l in open(txt, encoding="utf-8"):
        p = l.split()
        if len(p) == 5:
            cx, cy, bw, bh = map(float, p[1:])
            out.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return out


def gen_cutouts(limit=None):
    os.makedirs(CUT, exist_ok=True)
    import torch
    pred = sa._img_predictor()
    imgs = sorted(glob.glob(LABELS + "/images/*.jpg"))
    if limit:
        imgs = imgs[:limit]
    made = 0
    for ip in imgs:
        stem = os.path.splitext(os.path.basename(ip))[0]
        outp = CUT + f"/{stem}.png"
        if os.path.exists(outp):
            made += 1; continue
        lp = LABELS + f"/labels/{stem}.txt"
        if not os.path.exists(lp):
            continue
        im = sa._rd(ip); h, w = im.shape[:2]
        bx = yolo_xyxy(lp, w, h)
        if not bx:
            continue
        try:
            with torch.inference_mode(), torch.autocast(sa.DEV, dtype=torch.bfloat16):
                pred.set_image(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                masks, _, _ = pred.predict(box=np.array(bx[0], np.float32), multimask_output=False)
            m = masks[0]; m = (m[0] if m.ndim == 3 else m) > 0.0
        except Exception as e:
            print("[누끼]", stem, "실패", e, flush=True); continue
        bb = sa._bbox(m)
        if not bb or bb[2] - bb[0] < 12 or bb[3] - bb[1] < 12:
            continue
        x1, y1, x2, y2 = bb
        a = cv2.GaussianBlur(m[y1:y2 + 1, x1:x2 + 1].astype(np.uint8) * 255, (3, 3), 0)
        cv2.imwrite(outp, np.dstack([im[y1:y2 + 1, x1:x2 + 1], a])); made += 1
        if made % 60 == 0:
            print(f"[누끼] {made}/{len(imgs)}", flush=True)
    sa.free_sam2()
    return sorted(glob.glob(CUT + "/*.png"))


def rot(rgba, ang):
    h, w = rgba.shape[:2]; M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1]); nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += (nw - w) / 2; M[1, 2] += (nh - h) / 2
    return cv2.warpAffine(rgba, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


def paste(bg, fg, x, y):
    H, W = bg.shape[:2]; h, w = fg.shape[:2]
    x0, y0 = max(0, x), max(0, y); x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    sub = fg[y0 - y:y1 - y, x0 - x:x1 - x]; al = sub[:, :, 3:4].astype(np.float32) / 255
    bg[y0:y1, x0:x1] = (sub[:, :, :3].astype(np.float32) * al + bg[y0:y1, x0:x1].astype(np.float32) * (1 - al)).astype(np.uint8)
    ys, xs = np.where(sub[:, :, 3] > 15)
    return None if len(xs) == 0 else (x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()), y0 + int(ys.max()))


def prep_bg(path):
    bg = cv2.imread(path); h, w = bg.shape[:2]
    cw, ch = int(w * random.uniform(0.7, 1.0)), int(h * random.uniform(0.7, 1.0))
    x, y = random.randint(0, w - cw), random.randint(0, h - ch); bg = bg[y:y + ch, x:x + cw]
    s = 960 / max(bg.shape[:2])
    if s < 1:
        bg = cv2.resize(bg, (int(bg.shape[1] * s), int(bg.shape[0] * s)))
    if random.random() < 0.5:
        bg = cv2.flip(bg, 1)
    return np.clip(bg.astype(np.float32) * random.uniform(0.7, 1.25), 0, 255).astype(np.uint8)


def build_synthetic(cuts, n):
    si, sl = SYN + "/images", SYN + "/labels"; os.makedirs(si, exist_ok=True); os.makedirs(sl, exist_ok=True)
    if len(glob.glob(si + "/*.jpg")) >= n * 0.95:
        print("[합성] 캐시 재사용", flush=True); return
    for k in range(n):
        bg = prep_bg(random.choice(BGS)); H, W = bg.shape[:2]; labels = []
        for _ in range(1 if random.random() > 0.2 else 2):
            cut = cv2.imread(random.choice(cuts), cv2.IMREAD_UNCHANGED)
            if cut is None or cut.shape[2] != 4:
                continue
            cut = rot(cut, random.uniform(-20, 20)); tw = int(W * random.uniform(0.18, 0.42)); s = tw / cut.shape[1]
            cut = cv2.resize(cut, (max(1, int(cut.shape[1] * s)), max(1, int(cut.shape[0] * s))))
            ch, cw = cut.shape[:2]
            if cw >= W or ch >= H:
                continue
            bb = paste(bg, cut, random.randint(0, W - cw), random.randint(0, H - ch))
            if not bb:
                continue
            if OCCL and random.random() < 0.5:
                oc = rot(OCCL[random.randrange(len(OCCL))].copy(), random.uniform(-15, 15))
                ow = int((bb[2] - bb[0]) * random.uniform(0.4, 0.85)); so = ow / oc.shape[1]
                oc = cv2.resize(oc, (max(1, int(oc.shape[1] * so)), max(1, int(oc.shape[0] * so))))
                paste(bg, oc, random.randint(bb[0] - oc.shape[1] // 3, bb[2] - oc.shape[1] // 2),
                      random.randint(bb[1] - oc.shape[0] // 3, bb[3] - oc.shape[0] // 2))
            cx, cy = (bb[0] + bb[2]) / 2 / W, (bb[1] + bb[3]) / 2 / H; bw, bh = (bb[2] - bb[0]) / W, (bb[3] - bb[1]) / H
            labels.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        if labels:
            cv2.imwrite(si + f"/syn_{k:05d}.jpg", bg); open(sl + f"/syn_{k:05d}.txt", "w").write("\n".join(labels) + "\n")
        if (k + 1) % 300 == 0:
            print(f"[합성] {k+1}/{n}", flush=True)
    print(f"[합성] 완료 {len(glob.glob(si+'/*.jpg'))}장", flush=True)


def train(name, epochs):
    from ultralytics import YOLO
    import torch, gc
    files = [os.path.abspath(f) for f in sorted(glob.glob(LABELS + "/images/*.jpg"))] + \
            [os.path.abspath(f) for f in sorted(glob.glob(SYN + "/images/*.jpg"))]
    runid = datetime.now().strftime("%y%m%d_%H%M%S")
    mdir = BASE + f"/results/gearbox/{runid}/model/{name}"; os.makedirs(mdir, exist_ok=True)
    lst = mdir + "/train_list.txt"; open(lst, "w", encoding="utf-8").write("\n".join(files) + "\n")
    y = mdir + "/data.yaml"; open(y, "w", encoding="utf-8").write(f"train: {os.path.abspath(lst)}\nval: {os.path.abspath(lst)}\nnames:\n  0: part\n")
    m = YOLO("yolo26s.pt")
    m.train(data=y, epochs=epochs, imgsz=640, batch=8, device=0, project=mdir + "/runs", name="m",
            exist_ok=True, verbose=False, plots=False, degrees=15.0, **AUG)
    json.dump({"experiment": name, "source": "server_augtrain", "run": runid,
               "config": {"model": "yolo26s", "imgsz": 640, "epochs": epochs, "aug": "erasing0.5+scale0.9+mixup0.15",
                          "data": f"재탭{len(files)}(실촬영+합성)"}}, open(mdir + "/meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    del m; gc.collect(); torch.cuda.empty_cache()
    print(f"[학습] {name} → results/gearbox/{runid}/model/{name}/", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="빠른 검증(누끼 4장·합성 8장·학습 생략)")
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--epochs", type=int, default=100)
    a = ap.parse_args()
    LABELS = a.labels
    print(f"[시작] BASE={BASE}\n  라벨={LABELS} ({len(glob.glob(LABELS+'/images/*.jpg'))}장) 배경={len(BGS)} occluder={len(OCCL)}", flush=True)
    if a.smoke:
        cuts = gen_cutouts(limit=4); print("[검증] 누끼", len(cuts), flush=True)
        build_synthetic(cuts, 8); print("[검증] 합성", len(glob.glob(SYN + '/images/*.jpg')), "→ 파이프라인 정상. 실학습은 --smoke 빼고", flush=True)
    else:
        cuts = gen_cutouts(); print("[누끼]", len(cuts), flush=True)
        build_synthetic(cuts, 1500)
        train("cp_mix_aug", a.epochs)
    print("DONE", flush=True)
