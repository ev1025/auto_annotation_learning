# -*- coding: utf-8 -*-
"""gearbox+a_test 2클래스 '함께 학습' + 배경합성 ablation — 실측 GT mAP(전체 + 클래스별).

ablation_2class.py 와 동일(gearbox=0, a_test=1 remap, 2클래스 통합 GT 평가)하되,
학습 전 sam2_autolabel._synth_augment 로 실배경 copy-paste 합성 N장을 통합 학습셋에 추가한다.
축은 모델 9종(전체 라벨·100ep 고정)으로 '합성 켠 멀티클래스에서 어느 모델이 최선인가'를 실측.
(기존 2클래스 ablation 은 합성 미적용 = gearbox 기여분 과소평가. 그 보정.)

사용:
  XR_BASE=/workspace/data2/jinwoolee/xr_autolearning CUDA_VISIBLE_DEVICES=2 \
    python scripts/experiments/ablation_2class_synth.py
  # 스모크: 위에 ABL_SMOKE=1 ABL_NSYN=8 추가
결과: results/ablation_synth/gearbox_atest/<시각>/{results.json,summary.txt,run.log}
"""
import os, sys, glob, json, time, shutil, re
from pathlib import Path
from datetime import datetime

BASE = os.environ.get("XR_BASE") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, BASE + "/scripts/experiments")
sys.path.insert(0, BASE + "/scripts/verify")
import sam2_autolabel as sa            # _synth_augment

# 영상 stem -> 부품(폴더명) 매핑을 자체 내장한다(서버 build_multiclass 구버전 의존 회피).
# 로컬 파이프라인 build_multiclass._video_part_map 과 동일 규칙: data/bell412/<부품>/videos/<영상>.
_VID2PART = None


def _vid2part():
    global _VID2PART
    if _VID2PART is None:
        m = {}
        for vp in glob.glob(BASE + "/data/bell412/*/videos/*.*"):
            vstem = os.path.splitext(os.path.basename(vp))[0]
            part = os.path.basename(os.path.dirname(os.path.dirname(vp)))   # <부품>/videos/<영상>
            m.setdefault(vstem, part)
        _VID2PART = m
    return _VID2PART


def stem_to_class(stem):
    """프레임 stem -> 부품 클래스. 폴더명 우선, 못 찾으면 <카테고리>_<부품> 폴백."""
    v = re.sub(r"_\d+$", "", stem).replace("_TEST", "")
    vm = _vid2part()
    if v in vm:
        return re.sub(r"\s+", "_", vm[v].lower())
    part = v.split("_", 1)[1] if "_" in v else v
    part = re.sub(r"\d+$", "", part).strip()
    return re.sub(r"\s+", "_", part.lower())

CLASSES = ["gearbox", "a_test"]        # 인덱스 = 순서(gearbox=0, a_test=1)
IDX = {c: i for i, c in enumerate(CLASSES)}
GT_SRC = {
    "gearbox": BASE + "/data/bell412/gearbox/gt",
    "a_test":  BASE + "/data/bell412/a_test/gt",
}
N_SYN = int(os.environ.get("ABL_NSYN", "500"))
OUT = BASE + "/results/ablation_synth/gearbox_atest"
MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolo11n", "yolo11s", "yolo11m", "yolo26n", "yolo26s", "yolo26m"]
DEF_EP = 100
NAMES_BLOCK = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))


def collect_frames():
    """클래스별 raw 학습 프레임 경로. {cls: [img_path,...]}"""
    per = {c: [] for c in CLASSES}
    for ip in sorted(glob.glob(BASE + "/results/parts/*/train/images/*.jpg")):
        cls = stem_to_class(os.path.splitext(os.path.basename(ip))[0])
        if cls in per:
            per[cls].append(os.path.abspath(ip))
    return per


def _remap_label(src_txt, idx):
    """단일클래스(0) 라벨 → 첫 필드를 idx 로 교체한 줄 리스트."""
    out = []
    if os.path.exists(src_txt):
        for l in open(src_txt, encoding="utf-8"):
            p = l.split()
            if len(p) == 5:
                out.append(f"{idx} {p[1]} {p[2]} {p[3]} {p[4]}")
    return out


def build_dataset(rundir):
    """2클래스 통합 학습셋(이미지 복사 + 라벨 remap)을 _ds 에 만든다. 반환 (images_dir(Path), counts)."""
    di = Path(rundir) / "_ds" / "images"
    dl = Path(rundir) / "_ds" / "labels"
    di.mkdir(parents=True, exist_ok=True); dl.mkdir(parents=True, exist_ok=True)
    counts = {c: 0 for c in CLASSES}
    for cls, ips in collect_frames().items():
        for ip in ips:
            stem = cls + "__" + Path(ip).stem
            shutil.copy(ip, di / (stem + ".jpg"))   # 심볼릭 대신 복사(합성 누끼 glob 안전)
            lines = _remap_label(ip.replace("/images/", "/labels/")[:-4] + ".txt", IDX[cls])
            (dl / (stem + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
            counts[cls] += 1
    return di, dl, counts


def build_gt(rundir):
    """2클래스 통합 GT(이미지 심볼릭 + 라벨 remap) + gt.yaml 경로 반환."""
    gi, gl = rundir + "/_gt/images", rundir + "/_gt/labels"
    os.makedirs(gi, exist_ok=True); os.makedirs(gl, exist_ok=True)
    for cls in CLASSES:
        src = GT_SRC[cls]
        for ip in sorted(glob.glob(src + "/images/*.*")):
            ext = os.path.splitext(ip)[1].lower()
            if ext not in (".jpg", ".jpeg", ".png"):
                continue
            base = os.path.splitext(os.path.basename(ip))[0]
            stem = cls + "__" + base
            gip = gi + "/" + stem + ext
            if not os.path.exists(gip):
                os.symlink(os.path.abspath(ip), gip)
            lines = _remap_label(src + "/labels/" + base + ".txt", IDX[cls])
            open(gl + "/" + stem + ".txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    y = rundir + "/_gt/gt.yaml"
    open(y, "w", encoding="utf-8").write(f"path: {rundir}/_gt\ntrain:\n  - images\nval:\n  - images\nnames:\n{NAMES_BLOCK}\n")
    return y


def make_yaml(images_dir, rundir):
    y = str(Path(rundir) / "data.yaml")
    Path(y).write_text(
        f"train: {os.path.abspath(str(images_dir))}\nval: {os.path.abspath(str(images_dir))}\nnames:\n{NAMES_BLOCK}\n",
        encoding="utf-8")
    return y


def train_eval(images_dir, model, epochs, tag, rundir, gt_yaml):
    from ultralytics import YOLO
    import torch, gc
    y = make_yaml(images_dir, rundir)
    t = time.time()
    m = YOLO(model if model.endswith(".pt") else model + ".pt")
    m.train(data=y, epochs=epochs, imgsz=640, batch=8, device=0,   # CUDA_VISIBLE_DEVICES=2 → 물리 GPU2
            project=rundir + "/runs", name=tag, exist_ok=True, verbose=False, plots=False)
    det = YOLO(str(m.trainer.best))
    r = det.val(data=gt_yaml, imgsz=640, device=0, verbose=False)
    per = {}
    try:
        for i, ci in enumerate(r.box.ap_class_index):
            per[CLASSES[int(ci)]] = round(float(r.box.ap50[i]), 4)
    except Exception:
        pass
    res = {"tag": tag, "model": model, "epochs": epochs,
           "gt_map50": round(float(r.box.map50), 4), "gt_map5095": round(float(r.box.map), 4),
           "per_class_map50": per, "min": round((time.time() - t) / 60, 1)}
    del m, det; gc.collect(); torch.cuda.empty_cache()
    return res


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rundir = OUT + f"/{ts}"; os.makedirs(rundir, exist_ok=True)

    def log(s):
        print(s, flush=True)
        with open(rundir + "/run.log", "a", encoding="utf-8") as f:
            f.write(s + "\n")

    def logln(s, lvl="info"):
        log("  [synth] " + s)

    images_dir, labels_dir, counts = build_dataset(rundir)
    gt_yaml = build_gt(rundir)
    ng, na = counts["gearbox"], counts["a_test"]
    log(f"2클래스 + 배경합성 — raw gearbox {ng} + a_test {na} = {ng + na}장 · GT {gt_yaml}")
    if ng < 5 or na < 5:
        log("프레임 부족 — 중단"); return

    made = sa._synth_augment(images_dir, labels_dir, logln, n_syn=N_SYN)   # _ds 에 syn 추가(클래스 idx 유지)
    log(f"학습셋: raw {ng + na} + 합성 {made} = {ng + na + made}장")
    if made == 0:
        log("경고: 합성 0장(배경 없음?) — raw 만으로 학습")

    smoke = bool(os.environ.get("ABL_SMOKE"))
    models = ["yolo26n"] if smoke else MODELS
    eps = 3 if smoke else DEF_EP

    results = {"model": []}

    def save():
        json.dump(results, open(rundir + "/results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    for md in models:                         # 모델 축(전체 라벨+합성·100ep)
        try:
            r = train_eval(images_dir, md, eps, f"model_{md}", rundir, gt_yaml); results["model"].append(r)
            log(f"[모델] {md}: GT mAP50 {r['gt_map50']} 클래스별 {r['per_class_map50']} / mAP50-95 {r['gt_map5095']} ({r['min']}분)")
        except Exception as e:
            log(f"[모델] {md} 실패: {type(e).__name__}: {e}")
        save()

    lines = [f"gearbox+a_test 2클래스 + 배경합성 — 실측 GT mAP (raw {ng + na} + 합성 {made} = {ng + na + made}장)"]
    lines.append("\n== model ==")
    for r in results["model"]:
        pc = r.get("per_class_map50", {})
        lines.append(f"  {r['tag']:16} mAP50 {r['gt_map50']:.4f}  gearbox {pc.get('gearbox','-')} a_test {pc.get('a_test','-')}  mAP50-95 {r['gt_map5095']:.4f}  ({r['min']}분)")
    open(rundir + "/summary.txt", "w", encoding="utf-8").write("\n".join(lines))
    log("DONE → " + rundir + "/summary.txt")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
