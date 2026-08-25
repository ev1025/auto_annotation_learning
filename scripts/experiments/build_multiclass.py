# -*- coding: utf-8 -*-
"""멀티클래스 부품 학습(포터블).
대시보드 SAM2 오토라벨이 만든 per-part 라벨(파일명=<영상>_<프레임>, class 0)을
영상명→부품→클래스로 remap 통합 → 34클래스 YOLO 학습.
클래스 정의 = data/bell412/parts/part_codes.json(전역 부품 코드표). 라벨 = results/parts/<시각>/train (기본 최신, --labels로 지정).
사용: python scripts/experiments/build_multiclass.py [--labels <train>] [--epochs 100] [--build-only]"""
import os, sys, glob, re, shutil, json, argparse
from datetime import datetime
BASE = os.environ.get("XR_BASE") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PART_CODES = BASE + "/data/bell412/parts/part_codes.json"


def load_classes():
    """등록된 부품 이름 목록 + 이름->순번. 원천은 전역 코드표(part_codes.json).

    예전에는 손으로 관리하는 classes.txt 였는데, 실제 등록 부품과 어긋나서(a_test·medicine 누락)
    코드표 하나로 합쳤다. 여기서 돌려주는 순번은 '목록 안의 위치'일 뿐이고,
    학습 시에는 실제 학습 부품만 0..N-1 로 다시 매긴다(sam2_autolabel).
    클라이언트가 쓰는 불변 번호는 part_codes.json 의 코드값이다."""
    import json
    with open(PART_CODES, encoding="utf-8") as f:
        table = json.load(f)
    names = [n for n, _ in sorted(table.items(), key=lambda kv: kv[1])]   # 코드 순
    return names, {n: i for i, n in enumerate(names)}


def _video_part_map():
    """영상 stem → 부품(폴더명). data/bell412/<부품>/videos/<영상> 구조 기준.

    부품 정체성은 **폴더**가 결정한다. 영상 파일명은 참고하지 않는다.

      gearbox/videos/{Gearbox_gearbox1, Gearbox_gearbox2, train1, train2}  -> gearbox
      shim/videos/{Gearbox_shim, Gearbox_shim2}                            -> shim
      seal_spring/videos/Gearbox_spring                                    -> seal_spring
      cap_spring/videos/Gearbox_spring2                                    -> cap_spring

    즉 뒤의 숫자는 '같은 부품의 다른 테이크'라는 뜻이고, 다른 부품이면 폴더(=이름)를
    다르게 둔다. spring/spring2 가 서로 다른 부품이라 seal_spring/cap_spring 으로 개명한
    이유가 이것이다.

    캐시하지 않는다. 예전에는 모듈 전역에 한 번만 담아 뒀는데, 부품 이름을 바꾸면
    대시보드를 껐다 켜기 전까지 옛 폴더명을 계속 돌려줘서 '학습됨' 배지가 어긋났다.
    영상이 40여 개라 매번 훑어도 비용이 없다.
    """
    m = {}
    for vp in glob.glob(BASE + "/data/bell412/*/videos/*.*"):
        vstem = os.path.splitext(os.path.basename(vp))[0]
        part = os.path.basename(os.path.dirname(os.path.dirname(vp)))   # <부품>/videos/<영상>
        m.setdefault(vstem, part)
    return m


def stem_to_class(stem):
    """프레임/영상 stem → 부품 클래스. 폴더명 우선(신규 규칙), 못 찾으면 옛 영상명 규칙(<카테고리>_<부품>) 폴백."""
    v = re.sub(r"_\d+$", "", stem).replace("_TEST", "")     # 프레임번호 제거 → 영상 stem
    vm = _video_part_map()
    if v in vm:                                             # 폴더명 기준(영상 파일명 무관)
        return re.sub(r"\s+", "_", vm[v].lower())
    # 폴백(옛 규칙): 부품 폴더에 없는 영상일 때만. 카테고리 접두사를 떼고 끝 숫자를 지운다.
    # 지금 규칙과 어긋난다는 점에 주의 — Gearbox_spring2 가 여기로 오면 cap_spring 이 아니라
    # spring 이 된다. 폴더에 있는 영상은 위에서 이미 끝나므로 정상 경로에서는 닿지 않는다.
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
        print("  ⚠️ 코드표에 없는 부품(스킵):", {k: v for k, v in sorted(miss.items())})
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
