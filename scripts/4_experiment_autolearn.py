"""4_experiment_autolearn.py - 오토러닝이 실제로 성능을 올리는지 숫자로 검증하는 실험.

목적: "라벨 일부만 손라벨(시드) -> 모델이 나머지를 자동 라벨 -> 재학습" 루프가
      정말 mAP 를 올리는지, 라벨 있는 공개 데이터로 자동 채점한다.

설계 (정답 숨기기 트릭):
  전체 라벨 데이터를 3분할
    - seed  (기본 15%) : '사람이 손라벨한 척' -> 라운드0 학습에 라벨 사용
    - pool  (나머지)   : '미라벨인 척' 이미지만 사용. 정답 라벨은 채점용으로만 보관
    - test  (기본 20%) : 성능 측정 전용. 어떤 라운드에도 학습에 안 씀

  라운드0: seed 만으로 학습            -> test mAP (베이스라인)
  라운드1: 라운드0 모델이 pool 자동라벨 -> seed+pseudo 재학습 -> test mAP
  추가로 pseudo 라벨 자체를 숨겨둔 정답과 IoU 매칭해 정밀도/재현율 채점.

실행:
  python 4_experiment_autolearn.py --src ./mechanical-parts.v1 --classes bolt nut
  python 4_experiment_autolearn.py --src ./mp.v1 --classes bolt nut gear --epochs 80 --conf 0.6
"""
import argparse
import gc
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import torch
import yaml
from PIL import Image
from ultralytics import YOLO

import config


def _free_cuda():
    """DDP 학습 직후 GPU 메모리가 곧바로 안 풀리는 경우가 있어, 라운드 사이마다 명시적으로 회수한다."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

EXP_DIR = config.BASE_DIR / "exp_autolearn"  # 실험 산출물 루트(원본 데이터는 건드리지 않음)


# ---------------------------------------------------------------- 데이터 준비

def load_src_names(src):
    """Roboflow export 의 data.yaml 에서 {id: name} 을 읽는다."""
    data = yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8"))
    names = data.get("names")
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def collect_pairs(src):
    """train/valid/test 를 전부 합쳐 (이미지, 라벨) 짝 리스트로 반환.

    실험은 우리가 직접 3분할하므로 원본 분할은 무시하고 전부 합친다.
    """
    pairs = []
    for split in ("train", "valid", "val", "test"):
        d = src / split
        if not (d / "images").is_dir():
            continue
        for img in sorted((d / "images").glob("*")):
            if img.suffix.lower() not in config.IMG_EXTS:
                continue
            lbl = d / "labels" / f"{img.stem}.txt"
            if lbl.exists():
                pairs.append((img, lbl))
    return pairs


def filter_remap_label(lbl_path, keep_ids):
    """라벨 파일에서 선택 클래스만 남기고 0..k-1 로 재매핑한 줄 목록을 반환.

    2~3개 부품만 실험하므로 나머지 클래스 라벨은 버린다.
    keep_ids = {원본id: 새id}
    """
    lines = []
    for ln in lbl_path.read_text(encoding="utf-8").splitlines():
        parts = ln.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        if cid in keep_ids:
            lines.append(" ".join([str(keep_ids[cid])] + parts[1:5]))
    return lines


def build_splits(pairs, keep_ids, seed_frac, test_frac, rng_seed=42):
    """3분할 폴더 생성. 선택 클래스가 1개도 없는 이미지는 제외.

    반환: (seed_n, pool_n, test_n)
    """
    # 재현성: 고정 시드 셔플 -> 같은 명령이면 언제나 같은 분할.
    usable = []
    for img, lbl in pairs:
        lines = filter_remap_label(lbl, keep_ids)
        if lines:
            usable.append((img, lines))
    random.Random(rng_seed).shuffle(usable)

    n = len(usable)
    n_test = int(n * test_frac)
    n_seed = int(n * seed_frac)
    splits = {
        "test": usable[:n_test],
        "seed": usable[n_test:n_test + n_seed],
        "pool": usable[n_test + n_seed:],
    }

    for name, items in splits.items():
        img_d = EXP_DIR / name / "images"
        lbl_d = EXP_DIR / name / "labels"
        img_d.mkdir(parents=True, exist_ok=True)
        lbl_d.mkdir(parents=True, exist_ok=True)
        for img, lines in items:
            shutil.copy2(img, img_d / img.name)
            # pool 의 정답은 'labels_gt' 로 숨겨 보관(학습이 절대 못 보게 폴더명을 다르게).
            (lbl_d / f"{img.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        if name == "pool":
            (EXP_DIR / "pool" / "labels").rename(EXP_DIR / "pool" / "labels_gt")

    return len(splits["seed"]), len(splits["pool"]), len(splits["test"])


def write_round_yaml(round_name, train_dirs, class_names):
    """라운드별 학습용 data yaml 생성. val 은 항상 test 로 고정해 라운드 간 공정 비교.

    (주의) 여기서 val=test 는 '학습 중 지표 표시'용이고, epoch 선택(best.pt)이
    test 에 최적화되는 편향이 생기지만, 라운드0/1 이 같은 조건이므로 비교는 공정하다.
    """
    y = {
        "path": str(EXP_DIR.resolve()),
        "train": [str(Path(d) / "images") for d in train_dirs],
        "val": str(EXP_DIR / "test" / "images"),
        "names": class_names,
    }
    p = EXP_DIR / f"data_{round_name}.yaml"
    p.write_text(yaml.safe_dump(y, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


# ---------------------------------------------------------------- 학습/평가

def train_and_eval(round_name, data_yaml, args):
    """한 라운드 학습 후 test mAP 반환. 매 라운드 같은 사전학습 가중치에서 새로 시작(B방식).

    warm start 로 하면 pseudo 라벨 오류가 가중치에 고착될 수 있어, self-training
    비교 실험에서는 매번 같은 출발점에서 다시 학습하는 쪽이 결과 해석이 깨끗하다.
    """
    model = YOLO(config.PRETRAINED)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(EXP_DIR / "runs"),
        name=round_name,
        exist_ok=True,
        verbose=False,
    )
    best = Path(model.trainer.best)
    del model
    _free_cuda()  # DDP 학습(device=args.device) 뒤 GPU 메모리를 다음 단계 전에 확실히 회수

    m = YOLO(str(best))
    res = m.val(data=str(data_yaml), split="val", verbose=False)  # val = test 이미지
    result = {
        "best": str(best),
        "map50": round(float(res.box.map50), 4),
        "map50_95": round(float(res.box.map), 4),
        "per_class_map50_95": {res.names[i]: round(float(v), 4)
                               for i, v in zip(res.box.ap_class_index.tolist(),
                                               res.box.maps[res.box.ap_class_index].tolist())}
        if len(res.box.maps) else {},
    }
    del m, res
    _free_cuda()
    return result


def parse_conf_per_class(spec, class_names):
    """'gear=0.45,bolt=0.7' -> {클래스id: 임계값}. 클래스 소외(쉬운 클래스만 라벨 양산) 완화용."""
    if not spec:
        return {}
    name_to_id = {v: k for k, v in class_names.items()}
    out = {}
    for part in spec.split(","):
        name, _, val = part.partition("=")
        name = name.strip()
        if name not in name_to_id:
            raise SystemExit(f"[오류] --conf-per-class 클래스가 데이터에 없음: {name} / 보유: {list(name_to_id)}")
        out[name_to_id[name]] = float(val)
    return out


def _boxes_of(r, flip=False):
    """Results -> [(클래스id, (cx,cy,w,h), conf)]. flip=True 면 좌우반전 좌표를 원본 기준으로 복원."""
    bs = []
    if r.boxes is not None and len(r.boxes) > 0:
        for (cx, cy, w, h), c, cf in zip(r.boxes.xywhn.cpu().numpy(),
                                         r.boxes.cls.cpu().numpy().astype(int),
                                         r.boxes.conf.cpu().numpy()):
            if flip:
                cx = 1.0 - cx
            bs.append((int(c), (float(cx), float(cy), float(w), float(h)), float(cf)))
    return bs


def _consistent(cand, others, iou_thr=0.8):
    """후보 박스가 다른 모든 TTA 뷰에서 같은 클래스 + IoU>=0.8 로 재현되는지."""
    c, box, _ = cand
    return all(any(oc == c and _iou(box, ob) >= iou_thr for oc, ob, _ in other)
               for other in others)


def pseudo_label_pool(weights, conf, class_names, tta=False, conf_per_class=None):
    """라운드0 모델로 pool 이미지에 pseudo 라벨 생성. (라벨이미지수, 박스수, 분포통계) 반환.

    - tta=True: TTA 일관성 필터. 원본/좌우반전/0.8배 축소 3회 추론에서 모두 재현되는
      예측만 채택한다. '자신 있게 틀리는(confidently wrong)' 예측은 픽셀이 흔들리면
      재현이 깨지므로 여기서 걸러진다(도메인 갭 방어).
    - conf_per_class: 클래스별 임계값 덮어쓰기(잘 찾는 클래스는 높게, 못 찾는 클래스는 낮게).
    - 분포통계(클래스 비율/평균 conf/박스 크기)는 오류 증폭 조기 진단 지표로 report 에 기록.
    """
    out = EXP_DIR / "pool" / "labels_pseudo"
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(weights)
    imgs = sorted((EXP_DIR / "pool" / "images").glob("*"))
    conf_per_class = conf_per_class or {}
    # 추론은 가장 낮은 임계값으로 하고, 채택 단계에서 클래스별 임계값을 적용한다.
    min_conf = min([conf, *conf_per_class.values()]) if conf_per_class else conf

    n_lbl, n_box, tta_drop = 0, 0, 0
    stats = {"per_class": Counter(), "conf_sum": 0.0, "area_sum": 0.0}

    def emit(img_path, boxes):
        """클래스별 conf 임계값을 통과한 박스를 .txt 로 기록하고 통계 누적."""
        nonlocal n_lbl, n_box
        lines = []
        for c, (cx, cy, w, h), cf in boxes:
            if cf < conf_per_class.get(c, conf):
                continue
            lines.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            stats["per_class"][class_names.get(c, str(c))] += 1
            stats["conf_sum"] += cf
            stats["area_sum"] += w * h
        if lines:
            (out / f"{img_path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            n_lbl += 1
            n_box += len(lines)

    if not tta:
        # (주의) 리스트 source 에서 r.path 는 'image0' 같은 가짜 이름이 될 수 있다(ultralytics 8.4).
        # stream 제너레이터는 입력 순서를 보존하므로 입력 리스트와 zip 으로 짝지어 원본 파일명을 쓴다.
        results = model.predict(source=[str(p) for p in imgs], conf=min_conf, stream=True, verbose=False)
        for img_path, r in zip(imgs, results):
            emit(img_path, _boxes_of(r))
        del results
    else:
        # TTA: 이미지당 3뷰(원본/좌우반전/0.8배 축소)를 한 배치로 추론.
        # 정규화 좌표(xywhn)라 축소 뷰도 좌표 보정 없이 비교 가능, 반전만 cx 복원.
        for img_path in imgs:
            im = Image.open(img_path).convert("RGB")
            variants = [im,
                        im.transpose(Image.FLIP_LEFT_RIGHT),
                        im.resize((max(32, int(im.width * 0.8)), max(32, int(im.height * 0.8))))]
            rs = model.predict(source=variants, conf=min_conf, verbose=False)
            cands = [b for b in _boxes_of(rs[0]) if b[2] >= conf_per_class.get(b[0], conf)]
            others = [_boxes_of(rs[1], flip=True), _boxes_of(rs[2])]
            kept = [b for b in cands if _consistent(b, others)]
            tta_drop += len(cands) - len(kept)
            # kept 는 이미 conf 필터를 통과했으므로 emit 의 재필터에 그대로 통과된다.
            emit(img_path, kept)

    del model
    _free_cuda()

    summary = {
        "per_class_boxes": dict(stats["per_class"]),
        "boxes_per_labeled_image": round(n_box / n_lbl, 2) if n_lbl else 0.0,
        "mean_conf": round(stats["conf_sum"] / n_box, 4) if n_box else 0.0,
        "mean_bbox_area": round(stats["area_sum"] / n_box, 4) if n_box else 0.0,
        "tta": tta,
        "tta_dropped_boxes": tta_drop,
        "conf_per_class": {class_names[k]: v for k, v in conf_per_class.items()} or None,
    }
    return n_lbl, n_box, summary


# ---------------------------------------------------------------- pseudo 라벨 채점

def _iou(a, b):
    """정규화 cxcywh 두 박스의 IoU."""
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / union if union > 0 else 0.0


def _read_boxes(p):
    out = []
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            v = ln.split()
            if len(v) >= 5:
                out.append((int(float(v[0])), tuple(float(x) for x in v[1:5])))
    return out


def score_pseudo_labels(iou_thr=0.5):
    """pseudo 라벨을 숨겨둔 정답(labels_gt)과 클래스일치+IoU>=0.5 그리디 매칭으로 채점.

    반환: precision(자동 라벨 중 맞은 비율), recall(정답 중 잡아낸 비율)
    """
    gt_dir = EXP_DIR / "pool" / "labels_gt"
    ps_dir = EXP_DIR / "pool" / "labels_pseudo"
    tp = fp = fn = 0
    for gt_file in sorted(gt_dir.glob("*.txt")):
        gts = _read_boxes(gt_file)
        prs = _read_boxes(ps_dir / gt_file.name)
        used = [False] * len(gts)
        for pc, pb in prs:
            best_i, best_v = -1, 0.0
            for i, (gc, gb) in enumerate(gts):
                if used[i] or gc != pc:
                    continue
                v = _iou(pb, gb)
                if v > best_v:
                    best_i, best_v = i, v
            if best_i >= 0 and best_v >= iou_thr:
                used[best_i] = True
                tp += 1
            else:
                fp += 1
        fn += used.count(False)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4), "recall": round(rec, 4)}


# ---------------------------------------------------------------- 메인

def main():
    ap = argparse.ArgumentParser(description="오토러닝 효과 검증 실험(라운드0 vs 라운드1)")
    ap.add_argument("--src", required=True, help="Roboflow YOLOv8 export 폴더")
    ap.add_argument("--classes", nargs="+", required=True, help="실험할 클래스명 2~3개 (예: bolt nut)")
    ap.add_argument("--seed-frac", type=float, default=0.15, help="시드(손라벨 흉내) 비율")
    ap.add_argument("--test-frac", type=float, default=0.20, help="테스트 비율")
    ap.add_argument("--conf", type=float, default=config.AUTO_LABEL_CONF, help="pseudo 라벨 conf 임계값")
    ap.add_argument("--tta", action=argparse.BooleanOptionalAction, default=False,
                    help="TTA 일관성 필터(원본/반전/축소 3뷰 모두 재현되는 예측만 채택)")
    ap.add_argument("--conf-per-class", default=None,
                    help="클래스별 임계값 덮어쓰기 (예: gear=0.45,bolt=0.7)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=config.IMG_SIZE)
    ap.add_argument("--batch", type=int, default=config.BATCH)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    src = Path(args.src).resolve()
    src_names = load_src_names(src)                     # {원본id: 이름}
    name_to_id = {v: k for k, v in src_names.items()}
    missing = [c for c in args.classes if c not in name_to_id]
    if missing:
        raise SystemExit(f"[오류] 데이터에 없는 클래스: {missing} / 보유: {list(name_to_id)}")
    keep_ids = {name_to_id[c]: i for i, c in enumerate(args.classes)}  # 원본id -> 새id
    class_names = {i: c for i, c in enumerate(args.classes)}

    if EXP_DIR.exists():
        print(f"[준비] 기존 실험 폴더 삭제 후 재생성: {EXP_DIR}")
        shutil.rmtree(EXP_DIR)

    n_seed, n_pool, n_test = build_splits(collect_pairs(src), keep_ids,
                                          args.seed_frac, args.test_frac)
    print(f"[분할] seed {n_seed} / pool {n_pool} / test {n_test} (classes={args.classes})")

    # ---- 라운드0: 시드만 학습(베이스라인) ----
    y0 = write_round_yaml("round0", ["seed"], class_names)
    print("\n=== 라운드0: seed 만 학습 ===")
    r0 = train_and_eval("round0", y0, args)
    print(f"[라운드0] test mAP50={r0['map50']}  mAP50-95={r0['map50_95']}")

    # ---- pseudo 라벨 생성 + 채점 ----
    print("\n=== pool 자동 라벨링 ===")
    cpc = parse_conf_per_class(args.conf_per_class, class_names)
    n_lbl, n_box, pstats = pseudo_label_pool(r0["best"], args.conf, class_names,
                                             tta=args.tta, conf_per_class=cpc)
    print(f"[pseudo] 라벨된 이미지 {n_lbl}/{n_pool}장, 박스 {n_box}건 "
          f"(conf>={args.conf}, tta={args.tta}, tta_drop={pstats['tta_dropped_boxes']})")
    print(f"[pseudo 분포] 클래스별 {pstats['per_class_boxes']} / 평균 conf {pstats['mean_conf']}")
    quality = score_pseudo_labels()
    print(f"[pseudo 품질] precision={quality['precision']}  recall={quality['recall']} "
          f"(tp={quality['tp']} fp={quality['fp']} fn={quality['fn']})")

    # ---- 라운드1: seed + pseudo 재학습 ----
    # 학습이 pseudo 라벨을 읽도록 pool/labels 이름으로 배치(정답 labels_gt 는 그대로 숨김).
    (EXP_DIR / "pool" / "labels_pseudo").rename(EXP_DIR / "pool" / "labels")
    y1 = write_round_yaml("round1", ["seed", "pool"], class_names)
    print("\n=== 라운드1: seed + pseudo 재학습 ===")
    r1 = train_and_eval("round1", y1, args)
    print(f"[라운드1] test mAP50={r1['map50']}  mAP50-95={r1['map50_95']}")

    # ---- 가드레일: 라운드1이 라운드0보다 나쁘면 해당 라운드 폐기 권고 ----
    # 오류 증폭(pseudo 오라벨이 성능을 깎는 상황)을 자동 감지하는 최소 안전장치.
    accepted = r1["map50"] >= r0["map50"]
    guardrail = {
        "accepted": accepted,
        "rule": "round1 mAP50 >= round0 mAP50 (미달 시 pseudo 라벨 폐기·롤백 권고)",
    }

    # ---- 결과 리포트 ----
    report = {
        "classes": args.classes,
        "split": {"seed": n_seed, "pool": n_pool, "test": n_test},
        "pseudo": {"labeled_images": n_lbl, "boxes": n_box, "conf": args.conf,
                   **quality, **pstats},
        "round0": r0, "round1": r1,
        "delta_map50": round(r1["map50"] - r0["map50"], 4),
        "delta_map50_95": round(r1["map50_95"] - r0["map50_95"], 4),
        "guardrail": guardrail,
    }
    out = EXP_DIR / "report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n================ 결과 ================")
    print(f"  라운드0 (seed {n_seed}장)          : mAP50 {r0['map50']}")
    print(f"  라운드1 (+pseudo {n_lbl}장)        : mAP50 {r1['map50']}")
    print(f"  변화량                              : {report['delta_map50']:+}")
    print(f"  pseudo 품질                         : P {quality['precision']} / R {quality['recall']}")
    print(f"  가드레일                            : {'통과(채택)' if accepted else '실패(라운드 폐기 권고)'}")
    print(f"  상세: {out}")
    print("======================================")


if __name__ == "__main__":
    main()
