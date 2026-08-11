# -*- coding: utf-8 -*-
"""참조샷(서로 다른 각도 프레임) 개수 스윕 — SAM2 전파 커버리지 + 실측 GT mAP.

질문: "부품 하나 라벨링에 참조 프레임을 몇 장 탭해야 최적인가?"
방법: 한 영상의 컷 프레임 중 N개를 '서로 다른 시점(각도)'으로 골라 탭 → SAM2 전파 → 라벨.
      탭 점은 자동화를 위해 '기존 전체 전파 라벨'의 박스 중심을 그 프레임 탭점으로 사용(사람 탭 대용, foreground 1점).
측정: ①커버리지(전파로 라벨된 프레임 / 전체) ②그 라벨로 학습(yolov8n·100ep·합성없음) → 실측 GT mAP.
      참조 프레임 효과만 격리하려고 모델 고정·합성 미적용.
N = 1·2·4·6, 부품 = gearbox·a_test.

사용: XR_BASE=/workspace/data2/jinwoolee/xr_autolearning CUDA_VISIBLE_DEVICES=2 \
        python scripts/experiments/ablation_refshots.py   (스모크: ABL_SMOKE=1)
결과: results/experiments/ablation_refshots/<시각>/{results.json,summary.txt,run.log}
"""
import os, sys, glob, json, time, shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

BASE = os.environ.get("XR_BASE") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, BASE + "/scripts/experiments")
sys.path.insert(0, BASE + "/scripts/verify")
import sam2_autolabel as sa       # CFG·CKPT·DEV·_rd·_bbox·free_sam2
import autolabel                  # _frames, FRAME_CACHE

LABELS_DIR = BASE + "/results/parts/260806_081743/train/labels"   # 기존 전체 전파 라벨(탭점 부트스트랩용)
PARTS = [
    {"name": "gearbox", "video": "Gearbox_gearbox1", "gt": BASE + "/data/bell412/gearbox/gt/gt.yaml"},
    {"name": "a_test",  "video": "train",            "gt": BASE + "/data/bell412/a_test/gt/gt.yaml"},
]
NS = [1, 2, 4, 6]
MODEL, EP = "yolov8n", 100


def frame_boxes(video, fs):
    """컷 프레임 인덱스 -> (cx,cy,bw,bh) 정규화 박스. 기존 라벨(<video>_<stem>.txt)이 있는 프레임만.
    (중심점 1개는 작은 부품 a_test 에서 SAM2가 과소분할 → 박스 프롬프트로 부품 전체를 정확히 잡게 함)"""
    out = {}
    for i, fp in enumerate(fs):
        lp = Path(LABELS_DIR) / f"{video}_{Path(fp).stem}.txt"
        if lp.exists():
            lines = [l for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]
            if lines:
                p = lines[0].split()
                if len(p) == 5:
                    out[i] = (float(p[1]), float(p[2]), float(p[3]), float(p[4]))
    return out


def pick_refs(idxs, n):
    """라벨된 인덱스 리스트에서 균등 간격 n개(서로 다른 시점=각도)."""
    L = len(idxs)
    if n >= L:
        return list(idxs)
    if n == 1:
        return [idxs[L // 2]]
    return [idxs[round(k * (L - 1) / (n - 1))] for k in range(n)]


def propagate(video, fs, shots, log):
    """SAM2 비디오 전파: shots=[[frame_idx,[[cx,cy,lab],...]],...] → {frame_idx: mask(bool)}."""
    from sam2.build_sam import build_sam2_video_predictor
    cache_dir = autolabel.FRAME_CACHE / video
    h, w = sa._rd(fs[0]).shape[:2]
    predictor = build_sam2_video_predictor(sa.CFG, str(sa.CKPT), device=sa.DEV)
    masks = {}
    with torch.inference_mode(), torch.autocast(sa.DEV, dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(cache_dir), offload_video_to_cpu=True, offload_state_to_cpu=True)
        for fi, (cx, cy, bw, bh) in shots:
            box = np.array([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], dtype=np.float32)
            predictor.add_new_points_or_box(inference_state=state, frame_idx=int(fi), obj_id=1, box=box)
        for fidx, _ids, logits in predictor.propagate_in_video(state):
            m = logits[0].cpu().numpy()
            masks[fidx] = (m[0] if m.ndim == 3 else m) > 0.0
    del predictor, state
    sa.free_sam2()
    return masks, (h, w)


def build_labels(part, fs, masks, hw, rundir, tag):
    """마스크 → 박스(>10px) → YOLO 라벨(class 0) + 이미지 복사. 반환 (images_dir, n_valid)."""
    h, w = hw
    di = Path(rundir) / tag / "images"; dl = Path(rundir) / tag / "labels"
    di.mkdir(parents=True, exist_ok=True); dl.mkdir(parents=True, exist_ok=True)
    n_valid = 0
    for fidx, m in masks.items():
        if m is None or m.sum() == 0:
            continue
        if m.shape != (h, w):
            import cv2
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        bb = sa._bbox(m)
        if not bb or (bb[2] - bb[0]) <= 10 or (bb[3] - bb[1]) <= 10:
            continue
        stem = f"{part}_{Path(fs[fidx]).stem}"
        shutil.copy(fs[fidx], di / f"{stem}.jpg")
        cx = (bb[0] + bb[2]) / 2 / w; cy = (bb[1] + bb[3]) / 2 / h
        bw = (bb[2] - bb[0]) / w; bh = (bb[3] - bb[1]) / h
        (dl / f"{stem}.txt").write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")
        n_valid += 1
    return di, n_valid


def train_eval(images_dir, gt_yaml, part, epochs, tag, rundir):
    from ultralytics import YOLO
    import gc
    y = str(Path(rundir) / f"data_{tag}.yaml")
    Path(y).write_text(f"train: {os.path.abspath(str(images_dir))}\nval: {os.path.abspath(str(images_dir))}\nnames:\n  0: {part}\n", encoding="utf-8")
    t = time.time()
    m = YOLO(MODEL + ".pt")
    m.train(data=y, epochs=epochs, imgsz=640, batch=8, device=0,
            project=rundir + "/runs", name=tag, exist_ok=True, verbose=False, plots=False)
    det = YOLO(str(m.trainer.best))
    r = det.val(data=gt_yaml, imgsz=640, device=0, verbose=False)
    res = {"gt_map50": round(float(r.box.map50), 4), "gt_map5095": round(float(r.box.map), 4),
           "min": round((time.time() - t) / 60, 1)}
    del m, det; gc.collect(); torch.cuda.empty_cache()
    return res


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rundir = BASE + f"/results/experiments/ablation_refshots/{ts}"; os.makedirs(rundir, exist_ok=True)

    def log(s):
        print(s, flush=True)
        with open(rundir + "/run.log", "a", encoding="utf-8") as f:
            f.write(s + "\n")

    smoke = bool(os.environ.get("ABL_SMOKE"))
    ns = [1, 2] if smoke else NS
    ep = 3 if smoke else EP
    results = {}

    def save():
        json.dump(results, open(rundir + "/results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    for part in PARTS:
        name, video, gt = part["name"], part["video"], part["gt"]
        fs = autolabel._frames(video)
        if not fs:
            log(f"[{name}] 프레임 없음(video={video}) — 건너뜀"); continue
        centers = frame_boxes(video, fs)
        labeled = sorted(centers.keys())
        log(f"[{name}] 컷 {len(fs)}장 · 기존라벨 {len(labeled)}장 (박스 프롬프트 부트스트랩)")
        if len(labeled) < 6:
            log(f"[{name}] 라벨 부족 — 건너뜀"); continue
        results[name] = []
        for n in ns:
            try:
                refs = pick_refs(labeled, n)
                shots = [[i, centers[i]] for i in refs]
                masks, hw = propagate(video, fs, shots, log)
                images_dir, n_valid = build_labels(name, fs, masks, hw, rundir, f"{name}_N{n}")
                cov = round(n_valid / len(fs), 4)
                r = train_eval(images_dir, gt, name, ep, f"{name}_N{n}", rundir)
                row = {"N": n, "ref_frames": refs, "coverage": cov, "n_labeled": n_valid,
                       "total_frames": len(fs), **r}
                results[name].append(row)
                log(f"[{name}] N={n}: 커버리지 {cov} ({n_valid}/{len(fs)}) · GT mAP50 {r['gt_map50']} / 50-95 {r['gt_map5095']} ({r['min']}분)")
            except Exception as e:
                log(f"[{name}] N={n} 실패: {type(e).__name__}: {e}")
            save()

    lines = ["참조샷 개수 스윕 — SAM2 전파 커버리지 + 실측 GT mAP (모델 yolov8n·100ep·합성없음)"]
    for name, rows in results.items():
        lines.append(f"\n== {name} ==")
        for r in rows:
            lines.append(f"  N={r['N']}: 커버리지 {r['coverage']:.3f} ({r['n_labeled']}/{r['total_frames']})  GT mAP50 {r['gt_map50']:.4f}  mAP50-95 {r['gt_map5095']:.4f}  ({r['min']}분)")
    open(rundir + "/summary.txt", "w", encoding="utf-8").write("\n".join(lines))
    log("DONE → " + rundir + "/summary.txt")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
