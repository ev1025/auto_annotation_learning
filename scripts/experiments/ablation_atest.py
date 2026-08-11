# -*- coding: utf-8 -*-
"""라벨데이터 개수별 벤치마크 — 부품별 자동라벨 개수(N)를 스윕해 학습 → 실측 GT mAP.

질문: "부품 하나를 쓸 만한 인식률로 만들려면 자동라벨 몇 장이 필요한가?"
방법: 부품의 전체 자동라벨(results/autolabels/<부품>/labels)에서 균등간격 N장 서브샘플 →
      yolo11s·100ep 단일클래스 학습 → 수동 GT(data/bell412/<부품>/gt)로 실측 mAP 측정.
      라벨수 효과만 격리하려고 모델·에포크 고정, 합성 미적용.
N = 20·50·100·200·전체, 부품 = a_test·gearbox (ABL_PART 로 단일 지정 가능).

입력(새 저장구조): results/autolabels/<부품>/{images/<영상>/<프레임>.jpg, labels/<영상>_<프레임>.txt}
      GT = data/bell412/<부품>/gt (gt.yaml 의 절대경로 무시, 서버경로로 재생성)
사용: XR_BASE=/workspace/data2/jinwoolee/xr_autolearning CUDA_VISIBLE_DEVICES=2 \
        ./venv/bin/python scripts/experiments/ablation_atest.py   (스모크: ABL_SMOKE=1)
결과: results/experiments/ablation_labels/<시각>/{results.json,summary.txt,run.log}
"""
import os, sys, glob, json, time, re, shutil
from pathlib import Path
from datetime import datetime

BASE = os.environ.get("XR_BASE") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
AUTOLABELS = os.path.join(BASE, "results", "autolabels")
BG = os.path.join(BASE, "data", "bell412", "backgrounds")   # 배경 합성 증강용 실배경
SYNTH = bool(os.environ.get("ABL_SYNTH"))                    # ABL_SYNTH=1이면 각 지점에 배경합성 증강 추가
N_SYN = int(os.environ.get("ABL_NSYN", "400"))
# SAM2 (누끼용, 증강 켤 때만 사용)
CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
CKPT = os.path.join(BASE, "models", "sam2", "sam2.1_hiera_base_plus.pt")
DEV = "cuda"

PARTS = [os.environ["ABL_PART"]] if os.environ.get("ABL_PART") else ["a_test", "gearbox"]
GRID = [20, 50, 100, 200]          # + 각 부품 전체(ALL) 자동 추가
DEF_MODEL, DEF_EP = "yolo11s", 100


def _video_stem(stem):
    """'train_00012'→'train', 'Gearbox_gearbox1_00007'→'Gearbox_gearbox1' (뒤 프레임번호 제거)."""
    return re.sub(r"_\d+$", "", stem)


def collect_pairs(part):
    """부품의 (이미지, 라벨) 쌍 목록. 라벨 <video>_<frame> → 이미지 images/<video>/<frame>.jpg."""
    ld = os.path.join(AUTOLABELS, part, "labels")
    pairs = []
    for lp in sorted(glob.glob(ld + "/*.txt")):
        stem = Path(lp).stem
        video = _video_stem(stem)
        frame = stem[len(video) + 1:]
        ip = os.path.join(AUTOLABELS, part, "images", video, f"{frame}.jpg")
        lines = [l for l in Path(lp).read_text(encoding="utf-8").splitlines() if l.strip()]
        if os.path.exists(ip) and lines:
            pairs.append((ip, lp, stem))
    return pairs


def subsample(pairs, m):
    n = len(pairs)
    if m >= n:
        return pairs
    return [pairs[int(i * (n - 1) / (m - 1))] for i in range(m)]   # 균등 간격


def stage(pairs, part, tag, rundir):
    """선택 쌍을 rundir/data_<tag>/{images,labels} 로 스테이징(이미지·라벨 파일명 일치, class 0 강제).
    반환 images_dir."""
    d = Path(rundir) / f"data_{tag}"
    di, dl = d / "images", d / "labels"
    di.mkdir(parents=True, exist_ok=True); dl.mkdir(parents=True, exist_ok=True)
    for ip, lp, stem in pairs:
        shutil.copy(ip, di / f"{stem}.jpg")
        out = []
        for l in Path(lp).read_text(encoding="utf-8").splitlines():
            p = l.split()
            if len(p) == 5:
                out.append("0 " + " ".join(p[1:]))     # 단일클래스 → class 0 강제
        (dl / f"{stem}.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    return di


def gt_yaml_for(part, rundir):
    """GT yaml 을 이 실행 기준 절대경로로 재생성(원본 gt.yaml 의 Windows 절대경로 무시)."""
    gt_dir = os.path.join(BASE, "data", "bell412", part, "gt")
    y = os.path.join(rundir, f"gt_{part}.yaml")
    with open(y, "w", encoding="utf-8") as f:   # ultralytics 8.4는 train·val 키 둘 다 요구(평가는 val 만 사용)
        f.write(f"path: {gt_dir}\ntrain:\n  - images\nval:\n  - images\nnames:\n  0: {part}\n")
    return y


def train_eval(images_dir, gt_yaml, part, model, epochs, tag, rundir):
    from ultralytics import YOLO
    import torch, gc
    y = str(Path(rundir) / f"data_{tag}.yaml")
    Path(y).write_text(f"train: {os.path.abspath(str(images_dir))}\nval: {os.path.abspath(str(images_dir))}\nnames:\n  0: {part}\n", encoding="utf-8")
    t = time.time()
    m = YOLO(model if model.endswith(".pt") else model + ".pt")
    m.train(data=y, epochs=epochs, imgsz=640, batch=8, device=0, workers=0,
            project=rundir + "/runs", name=tag, exist_ok=True, verbose=False, plots=False)
    det = YOLO(str(m.trainer.best))
    r = det.val(data=gt_yaml, imgsz=640, device=0, verbose=False)
    res = {"tag": tag, "model": model, "epochs": epochs,
           "gt_map50": round(float(r.box.map50), 4), "gt_map5095": round(float(r.box.map), 4),
           "min": round((time.time() - t) / 60, 1)}
    del m, det; gc.collect(); torch.cuda.empty_cache()
    return res


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    top = "ablation_labels_synth" if SYNTH else "ablation_labels"
    rundir = BASE + f"/results/experiments/{top}/{ts}"; os.makedirs(rundir, exist_ok=True)

    def log(s):
        print(s, flush=True)
        with open(rundir + "/run.log", "a", encoding="utf-8") as f:
            f.write(s + "\n")

    smoke = bool(os.environ.get("ABL_SMOKE"))
    ep = 3 if smoke else DEF_EP
    results = {}

    def save():
        json.dump(results, open(rundir + "/results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    for part in PARTS:
        gt = gt_yaml_for(part, rundir)
        pairs = collect_pairs(part)
        log(f"[{part}] 자동라벨 {len(pairs)}장 · GT {gt}")
        if len(pairs) < 5:
            log(f"[{part}] 라벨 부족 — 건너뜀"); continue
        alln = len(pairs)
        grid = [10, 20] if smoke else sorted(set([g for g in GRID if g < alln] + [alln]))
        results[part] = []
        for m in grid:
            try:
                sel = subsample(pairs, m)
                images_dir = stage(sel, part, f"{part}_N{m}", rundir)
                n_syn = 0
                if SYNTH:                                    # 배경 합성 증강: 오토라벨 → 누끼 → 실배경 합성 추가
                    from synth_aug import synth_augment
                    lbl_dir = Path(rundir) / f"data_{part}_N{m}" / "labels"
                    n_syn = synth_augment(images_dir, lbl_dir, BG, CFG, CKPT, DEV, log, n_syn=N_SYN)
                r = train_eval(images_dir, gt, part, DEF_MODEL, ep, f"{part}_N{m}", rundir)
                row = {"N": len(sel), "n_synth": n_syn, **r}
                results[part].append(row)
                log(f"[{part}] N={len(sel)}: 합성 {n_syn} · GT mAP50 {r['gt_map50']} / 50-95 {r['gt_map5095']} ({r['min']}분)")
            except Exception as e:
                log(f"[{part}] N={m} 실패: {type(e).__name__}: {e}")
            save()

    tag = f"합성증강 on(n_syn={N_SYN})" if SYNTH else "합성없음"
    lines = [f"라벨데이터 개수별 벤치마크 — 실측 GT mAP (모델 yolo11s·100ep·{tag})"]
    for part, rows in results.items():
        lines.append(f"\n== {part} ==")
        for r in rows:
            syn = f" +합성{r.get('n_synth', 0)}" if SYNTH else ""
            lines.append(f"  N={r['N']:4d}{syn}: mAP50 {r['gt_map50']:.4f}  mAP50-95 {r['gt_map5095']:.4f}  ({r['epochs']}ep, {r['min']}분)")
    open(rundir + "/summary.txt", "w", encoding="utf-8").write("\n".join(lines))
    log("DONE → " + rundir + "/summary.txt")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
