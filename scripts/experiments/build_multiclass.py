# -*- coding: utf-8 -*-
"""멀티클래스 부품 학습(포터블).
대시보드 SAM2 오토라벨이 만든 per-part 라벨(파일명=<영상>_<프레임>, class 0)을
영상명→부품→클래스로 remap 통합 → 34클래스 YOLO 학습.
클래스 정의 = data/bell412/parts/classes.txt. 라벨 = results/parts/<시각>/train (기본 최신, --labels로 지정).
사용: python scripts/experiments/build_multiclass.py [--labels <train>] [--epochs 100] [--build-only]"""
import os, sys, glob, re, shutil, json, argparse
from datetime import datetime
BASE = os.environ.get("XR_BASE") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CLASSES_TXT = BASE + "/data/bell412/parts/classes.txt"


def load_classes():
    names = [l.strip() for l in open(CLASSES_TXT, encoding="utf-8") if l.strip()]
    return names, {n: i for i, n in enumerate(names)}


def stem_to_class(stem):
    """라벨 파일 stem(<영상>_<프레임>) → 클래스명. 영상명 = 프레임번호 뗀 것, 카테고리 접두·끝숫자·_TEST 제거."""
    v = re.sub(r"_\d+$", "", stem).replace("_TEST", "")     # 프레임번호 제거
    part = v.split("_", 1)[1] if "_" in v else v            # 카테고리(Gearbox_/Tools_) 제거
    part = re.sub(r"\d+$", "", part).strip()                # bracket2 → bracket
    return re.sub(r"\s+", "_", part.lower())                # seal support → seal_support


def find_latest_labels():
    c = sorted(glob.glob(BASE + "/results/parts/*/train"), reverse=True)
    return c[0] if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=None, help="per-part 라벨 train 폴더(기본=results/parts 최신)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--build-only", action="store_true", help="데이터셋 통합만 하고 학습 생략")
    a = ap.parse_args()
    names, name2idx = load_classes()
    src = a.labels or find_latest_labels()
    if not src or not os.path.isdir(src):
        print(f"[중단] per-part 라벨 없음. 대시보드서 parts 폴더 부품 탭→라벨 생성 먼저. (찾은 곳: {src})"); sys.exit(1)
    print(f"[통합] 라벨 소스: {src}\n  클래스 {len(names)}종")
    runid = datetime.now().strftime("%y%m%d_%H%M%S")
    out = BASE + f"/results/parts/{runid}/multiclass"
    oi, ol = out + "/images", out + "/labels"
    os.makedirs(oi, exist_ok=True); os.makedirs(ol, exist_ok=True)
    per = {}; miss = {}
    for ip in sorted(glob.glob(src + "/images/*.jpg")):
        stem = os.path.splitext(os.path.basename(ip))[0]
        cls = stem_to_class(stem)
        idx = name2idx.get(cls)
        if idx is None:
            miss[cls] = miss.get(cls, 0) + 1; continue
        lp = src + f"/labels/{stem}.txt"
        if not os.path.exists(lp):
            continue
        lines = [f"{idx} {p[1]} {p[2]} {p[3]} {p[4]}" for p in (l.split() for l in open(lp, encoding="utf-8")) if len(p) == 5]
        if not lines:
            continue
        shutil.copy(ip, oi + f"/{stem}.jpg")
        open(ol + f"/{stem}.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
        per[cls] = per.get(cls, 0) + 1
    if miss:
        print("  ⚠️ classes.txt에 없는 부품(스킵):", {k: v for k, v in sorted(miss.items())})
    print(f"  통합 완료: {sum(per.values())}장 / {len(per)}클래스")
    for c, n in sorted(per.items()):
        print(f"    {c}: {n}")
    covered = set(per); allc = set(names)
    if allc - covered:
        print(f"  ⚠️ 아직 라벨 없는 클래스({len(allc-covered)}): {sorted(allc-covered)}")
    # data.yaml
    yml = out + "/data.yaml"
    nb = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
    open(yml, "w", encoding="utf-8").write(f"path: {os.path.abspath(out)}\ntrain:\n  - images\nval:\n  - images\nnames:\n{nb}\n")
    json.dump({"run": runid, "labels_src": src, "n_images": sum(per.values()), "per_class": per,
               "classes": names}, open(out + "/meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  데이터셋: {out}")
    if a.build_only:
        print("DONE(통합만)"); return
    from ultralytics import YOLO
    import torch, gc
    m = YOLO("yolo26s.pt")
    m.train(data=yml, epochs=a.epochs, imgsz=640, batch=8, device=0, project=out + "/runs", name="m",
            exist_ok=True, verbose=False, plots=False, degrees=15.0)
    del m; gc.collect(); torch.cuda.empty_cache()
    print(f"[학습 완료] {out}/runs/m/weights/best.pt")
    print("DONE")


if __name__ == "__main__":
    main()
