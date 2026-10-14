# -*- coding: utf-8 -*-
"""도메인갭 진단 — '분류 실패'인가 '검출 실패'인가를 가른다.

배경
  36클래스 모델은 in-sample mAP50 0.99 인데 기체 장착 구간(test3·4)에서 무너진다.
  같은 레시피의 1클래스 gearbox 모델은 test4 를 85% 잡는다. 원인이 (a) 박스는 쳤는데
  이름을 틀린 것(분류 간섭)인지 (b) 박스 자체를 못 만든 것(객체성 붕괴)인지 갈라야
  대책(클래스 병합 vs 도메인 적응 vs 손실 재가중)이 정해진다.

무엇을 재나
  구간별(test1~4, 각 10장 · 사람 GT 박스)
    1) class-aware   맞는 클래스로 IoU>=0.5 맞춘 비율 + AP50
    2) class-agnostic 클래스를 무시하고 IoU>=0.5 맞춘 비율 + AP50   <- 둘의 차 = 분류 실패분
    3) agnostic 으로 맞췄을 때 모델이 부른 이름(오분류 분포)
    4) TIDE 6종 오차 분해(ΔAP: Cls·Loc·Both·Dupe·Bkg·Miss)
    5) GT 객체 크기 분포(구간별) — 학습셋 크기 분포와의 격차 확인용

사용
  XR_BASE=$PWD CUDA_VISIBLE_DEVICES=2 ./venv/bin/python scripts/experiments/diag_domaingap.py \
      --out results/diag/<시각>
결과: <out>/{summary.txt, results.json, tide/*.txt}
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get("XR_BASE") or Path(__file__).resolve().parents[2])
GT_DIR = BASE / "data/bell412/gearbox/gt"
TARGET = "gearbox"
ZONES = ["test1", "test2", "test3", "test4"]
CONFS = [0.7, 0.5, 0.25, 0.10, 0.05, 0.01]
IOU_HIT = 0.5

# 진단 대상 가중치 (이름 -> 경로). 없는 파일은 건너뛴다.
DEFAULT_MODELS = {
    "36cls_3x_11m": "results/bench/260824_retrain3x/yolo11m/weights/best.pt",
    "36cls_3x_11s": "results/bench/260824_retrain3x/yolo11s/weights/best.pt",
    "36cls_occl8_11m": "results/bench/260825_occl8/yolo11m/weights/best.pt",
    "1cls_gearbox_11s": "results/bench/260826_gearbox1/yolo11s/weights/best.pt",
}


def gt_index():
    """구간 -> [(이미지경로, [xyxy 픽셀박스...], (w,h))]"""
    out = {z: [] for z in ZONES}
    for ip in sorted(glob.glob(str(GT_DIR / "images/*.jpg"))):
        stem = Path(ip).stem
        zone = stem.split("_")[0]
        if zone not in out:
            continue
        import cv2
        h, w = cv2.imread(ip).shape[:2]
        boxes = []
        lp = GT_DIR / "labels" / f"{stem}.txt"
        if lp.exists():
            for ln in lp.read_text(encoding="utf-8").splitlines():
                f = ln.split()
                if len(f) >= 5:
                    cx, cy, bw, bh = map(float, f[1:5])
                    boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        out[zone].append((ip, boxes, (w, h)))
    return out


def iou_mat(dets, gts):
    """dets (N,4), gts (M,4) 픽셀 xyxy -> (N,M) IoU"""
    if not len(dets) or not len(gts):
        return np.zeros((len(dets), len(gts)), np.float32)
    d = np.asarray(dets, np.float32)[:, None, :]
    g = np.asarray(gts, np.float32)[None, :, :]
    ix1 = np.maximum(d[..., 0], g[..., 0]); iy1 = np.maximum(d[..., 1], g[..., 1])
    ix2 = np.minimum(d[..., 2], g[..., 2]); iy2 = np.minimum(d[..., 3], g[..., 3])
    iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    ad = (d[..., 2] - d[..., 0]) * (d[..., 3] - d[..., 1])
    ag = (g[..., 2] - g[..., 0]) * (g[..., 3] - g[..., 1])
    return inter / np.maximum(ad + ag - inter, 1e-9)


def ap50(scores, matched, n_gt):
    """단일 클래스 AP@0.5 (COCO 101점 보간). scores·matched 는 검출 단위."""
    if n_gt == 0:
        return float("nan")
    if not len(scores):
        return 0.0
    o = np.argsort(-np.asarray(scores))
    m = np.asarray(matched, bool)[o]
    tp = np.cumsum(m); fp = np.cumsum(~m)
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    # 우측 최대값으로 정밀도 단조화 후 101점 보간
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.mean(np.interp(np.linspace(0, 1, 101), rec, prec, left=prec[0], right=0.0)))


def greedy_match(dets_xyxy, scores, gts, thr=IOU_HIT):
    """점수 내림차순 그리디 매칭 -> 검출별 매칭 여부"""
    order = np.argsort(-np.asarray(scores)) if len(scores) else np.array([], int)
    M = iou_mat([dets_xyxy[i] for i in order], gts) if len(order) else np.zeros((0, len(gts)))
    used = set(); matched = np.zeros(len(order), bool)
    for k in range(len(order)):
        best, bj = thr, -1
        for j in range(len(gts)):
            if j in used:
                continue
            if M[k, j] >= best:
                best, bj = M[k, j], j
        if bj >= 0:
            used.add(bj); matched[k] = True
    inv = np.empty(len(order), int); inv[order] = np.arange(len(order))
    return matched[inv]


POS_T, BG_T = 0.5, 0.1


def _ap_greedy(dets, scores, gts, n_gt, iou_override=None):
    """주어진 검출로 그리디 매칭 후 AP50. iou_override: 특정 검출의 IoU 를 1 로 올려 쓰기(Loc 수정용)."""
    if n_gt == 0:
        return float("nan")
    if not len(dets):
        return 0.0
    M = iou_mat(dets, gts)
    if iou_override:
        for k in iou_override:
            if k < len(M):
                M[k, iou_override[k]] = 1.0
    order = np.argsort(-np.asarray(scores))
    used, matched = set(), np.zeros(len(order), bool)
    for rank, k in enumerate(order):
        best, bj = POS_T, -1
        for j in range(len(gts)):
            if j not in used and M[k, j] >= best:
                best, bj = M[k, j], j
        if bj >= 0:
            used.add(bj); matched[rank] = True
    return ap50(np.asarray(scores)[order], matched, n_gt)


def tide_single_class(rows, cid):
    """gearbox 한 클래스에 대한 TIDE 식 오차 분해(ΔAP). 부분라벨 왜곡이 없다.

    tidecv 를 그대로 쓰면 GT 가 gearbox 만 있는데 카테고리를 36개 등록해서, 나머지 35종의
    검출(실제로 화면에 있는 공구들)이 전부 배경 FP 로 잡혀 AP 가 무너진다. 그래서 관심
    클래스 하나에 대해 TIDE 정의를 그대로 구현한다.

      Cls  다른 이름을 붙였지만 자리는 맞음(IoU>=0.5)     -> 고치면: 그 검출을 정답 클래스로
      Loc  이름은 맞고 자리가 어긋남(0.1<=IoU<0.5)        -> 고치면: 그 검출의 IoU 를 1 로
      Both 이름도 자리도 어긋남                            -> 관심 클래스 AP 에는 영향 없음
      Dupe 이미 잡힌 GT 를 또 잡음                         -> 고치면: 제거
      Bkg  아무 GT 와도 안 겹침(IoU<0.1)                   -> 고치면: 제거
      Miss 위 어느 것도 그 GT 근처에 없음(= 아예 못 봄)    -> 고치면: 그 GT 를 분모에서 제외
    """
    dets, scores, clss, gt_all, gt_off = [], [], [], [], []
    for row in rows:                                   # 사진별 GT 를 하나의 인덱스 공간으로
        off = len(gt_all)
        gt_off.append(off)
        gt_all += [list(b) for b in row["gt"]]
        for b, c, cl in zip(row["xyxy"], row["conf"], row["cls"]):
            dets.append(list(b)); scores.append(float(c)); clss.append(int(cl))
    n_gt = len(gt_all)
    if n_gt == 0:
        return {}
    # 검출-GT IoU 는 같은 사진 안에서만 의미가 있다 -> 사진 밖은 0 으로 막는다
    M = np.zeros((len(dets), n_gt), np.float32)
    di = 0
    for r, row in enumerate(rows):
        n_d = len(row["xyxy"])
        if n_d and row["gt"]:
            M[di:di + n_d, gt_off[r]:gt_off[r] + len(row["gt"])] = iou_mat(row["xyxy"], row["gt"])
        di += n_d
    scores = np.asarray(scores, np.float32); clss = np.asarray(clss, int)
    order = np.argsort(-scores)
    kind = np.array(["ok"] * len(dets), dtype=object)
    best_gt = np.full(len(dets), -1)
    used = set()
    for k in order:
        j = int(np.argmax(M[k])) if n_gt else -1
        iou = float(M[k, j]) if j >= 0 else 0.0
        best_gt[k] = j
        if clss[k] == cid:
            free = [jj for jj in range(n_gt) if jj not in used and M[k, jj] >= POS_T]
            if free:
                jj = max(free, key=lambda x: M[k, x]); used.add(jj); best_gt[k] = jj; kind[k] = "TP"
            elif iou >= POS_T:
                kind[k] = "Dupe"
            elif iou >= BG_T:
                kind[k] = "Loc"
            else:
                kind[k] = "Bkg"
        else:
            if iou >= POS_T:
                kind[k] = "Cls"
            elif iou >= BG_T:
                kind[k] = "Both"
            else:
                kind[k] = "Other"      # 화면의 다른 물체 = 우리 GT 가 라벨하지 않은 것(오차 아님)
    near = {int(best_gt[k]) for k in range(len(dets)) if kind[k] in ("Cls", "Loc", "Both")}
    missed = [j for j in range(n_gt) if j not in used and j not in near]

    own = [k for k in range(len(dets)) if clss[k] == cid]
    base = _ap_greedy([dets[k] for k in own], scores[own], gt_all, n_gt)

    def ap_of(keep, extra=(), override=None, n=None):
        idx = list(keep) + list(extra)
        return _ap_greedy([dets[k] for k in idx], scores[idx], gt_all, n if n is not None else n_gt,
                          iou_override={i: int(best_gt[idx[i]]) for i in range(len(idx))
                                        if override and idx[i] in override})

    cls_e = [k for k in range(len(dets)) if kind[k] == "Cls"]
    loc_e = [k for k in own if kind[k] == "Loc"]
    dup_e = [k for k in own if kind[k] == "Dupe"]
    bkg_e = [k for k in own if kind[k] == "Bkg"]
    both_e = [k for k in range(len(dets)) if kind[k] == "Both"]
    d = {}
    d["Cls"] = round(100 * (ap_of(own, extra=cls_e) - base), 2)                       # 이름만 고치면
    d["Loc"] = round(100 * (ap_of(own, override=set(loc_e)) - base), 2)               # 자리만 고치면
    d["Both"] = 0.0                                                                   # 관심 클래스 AP 무관
    d["Dupe"] = round(100 * (ap_of([k for k in own if k not in dup_e]) - base), 2)
    d["Bkg"] = round(100 * (ap_of([k for k in own if k not in bkg_e]) - base), 2)
    d["Miss"] = round(100 * (_ap_greedy([dets[k] for k in own], scores[own],
                                        [g for j, g in enumerate(gt_all) if j not in missed],
                                        n_gt - len(missed)) - base), 2) if missed else 0.0
    d["_ap50"] = round(100 * base, 2)
    d["_counts"] = {"TP": int((kind == "TP").sum()), "Cls": len(cls_e), "Loc": len(loc_e),
                    "Both": len(both_e), "Dupe": len(dup_e), "Bkg": len(bkg_e),
                    "Miss": len(missed), "n_gt": n_gt,
                    "Other(미라벨 물체)": int((kind == "Other").sum())}
    return d


def run_model(tag, wpath, gts, out_dir, device=0):
    from ultralytics import YOLO
    m = YOLO(str(wpath))
    names = {int(k): v for k, v in m.names.items()}
    tgt_id = next((k for k, v in names.items() if v == TARGET), None)
    res = {"tag": tag, "weights": str(wpath), "n_classes": len(names),
           "target_class_id": tgt_id, "zones": {}}
    # COCO 형식 누적(TIDE 용)
    coco_gt = {"images": [], "annotations": [], "categories": []}
    coco_dt = []
    cat_ids = {i: i + 1 for i in names}
    coco_gt["categories"] = [{"id": cat_ids[i], "name": names[i], "supercategory": "part"} for i in sorted(names)]
    img_id = ann_id = 0
    all_rows = {}

    for zone in ZONES:
        items = gts[zone]
        paths = [p for p, _, _ in items]
        preds = []
        for i in range(0, len(paths), 20):
            preds += m.predict(paths[i:i + 20], conf=0.001, iou=0.7, imgsz=640,
                               max_det=300, device=device, verbose=False)
        rows = []
        n_gt = 0
        for (ip, gbox, (w, h)), r in zip(items, preds):
            img_id += 1
            coco_gt["images"].append({"id": img_id, "file_name": Path(ip).name, "width": w, "height": h})
            for b in gbox:
                ann_id += 1
                coco_gt["annotations"].append({
                    "id": ann_id, "image_id": img_id, "category_id": cat_ids.get(tgt_id, 1),
                    "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                    # tidecv 의 COCO 로더는 segmentation 을 필수로 읽는다(마스크 IoU 는 안 쓰지만
                    # 파싱 단계에서 KeyError). 박스를 그대로 사각 폴리곤으로 넣어 준다.
                    "segmentation": [[b[0], b[1], b[2], b[1], b[2], b[3], b[0], b[3]]],
                    "area": (b[2] - b[0]) * (b[3] - b[1]), "iscrowd": 0})
            n_gt += len(gbox)
            xyxy = r.boxes.xyxy.cpu().numpy().tolist() if r.boxes is not None else []
            conf = r.boxes.conf.cpu().numpy().tolist() if r.boxes is not None else []
            cls = [int(c) for c in (r.boxes.cls.cpu().numpy().tolist() if r.boxes is not None else [])]
            for b, c, cl in zip(xyxy, conf, cls):
                coco_dt.append({"image_id": img_id, "category_id": cat_ids[cl],
                                "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": float(c)})
            rows.append({"path": ip, "gt": gbox, "xyxy": xyxy, "conf": conf, "cls": cls})
        all_rows[zone] = rows

        # --- class-aware / class-agnostic 집계 ---
        zres = {"n_images": len(items), "n_gt": n_gt}
        for mode in ("aware", "agnostic"):
            sc_all, mt_all = [], []
            hit_at = {c: 0 for c in CONFS}
            best_conf_of_hit = []
            wrong_names = Counter()
            for row in rows:
                keep = [k for k in range(len(row["conf"]))
                        if mode == "agnostic" or row["cls"][k] == tgt_id]
                dets = [row["xyxy"][k] for k in keep]
                scs = [row["conf"][k] for k in keep]
                mt = greedy_match(dets, scs, row["gt"]) if dets else np.zeros(0, bool)
                sc_all += scs; mt_all += list(mt)
                # 구간 hit: 그 사진에서 IoU>=0.5 로 맞춘 검출의 최고 신뢰도
                hits = [scs[k] for k in range(len(scs)) if mt[k]]
                top = max(hits) if hits else 0.0
                for c in CONFS:
                    if top >= c:
                        hit_at[c] += 1
                if hits:
                    best_conf_of_hit.append(top)
                    if mode == "agnostic":
                        k = int(np.argmax([scs[i] if mt[i] else -1 for i in range(len(scs))]))
                        wrong_names[m.names[row["cls"][keep[k]]]] += 1
            zres[mode] = {
                "ap50": round(ap50(sc_all, mt_all, n_gt), 4),
                "hit_rate": {str(c): round(hit_at[c] / max(1, len(items)), 3) for c in CONFS},
                "hit_n": {str(c): hit_at[c] for c in CONFS},
                "mean_conf_of_hit": round(float(np.mean(best_conf_of_hit)), 3) if best_conf_of_hit else 0.0,
            }
            if mode == "agnostic":
                zres["agnostic_called_it"] = dict(wrong_names.most_common())
        # GT 크기(진단 5)
        sizes = []
        for _, gbox, (w, h) in items:
            for b in gbox:
                sizes.append(np.sqrt(((b[2] - b[0]) / w) * ((b[3] - b[1]) / h)))
        zres["gt_size_rel"] = {"mean": round(float(np.mean(sizes)), 4), "min": round(float(np.min(sizes)), 4),
                               "max": round(float(np.max(sizes)), 4)} if sizes else {}
        zres["gt_px_at640"] = round(float(np.mean(sizes)) * 640, 1) if sizes else 0.0
        zres["tide_1cls"] = tide_single_class(rows, tgt_id) if tgt_id is not None else {}
        res["zones"][zone] = zres

    # --- TIDE ---
    (out_dir / "coco").mkdir(parents=True, exist_ok=True)
    gtp = out_dir / "coco" / f"{tag}_gt.json"
    dtp = out_dir / "coco" / f"{tag}_dt.json"
    gtp.write_text(json.dumps(coco_gt), encoding="utf-8")
    dtp.write_text(json.dumps(coco_dt), encoding="utf-8")
    res["tide"] = tide_report(gtp, dtp, out_dir, tag, coco_gt, coco_dt)
    res["tide_1cls"] = {"zone12": tide_single_class(all_rows["test1"] + all_rows["test2"], tgt_id),
                        "zone34": tide_single_class(all_rows["test3"] + all_rows["test4"], tgt_id),
                        "test3": tide_single_class(all_rows["test3"], tgt_id),
                        "test4": tide_single_class(all_rows["test4"], tgt_id)} if tgt_id is not None else {}
    del m
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()
    return res


def tide_report(gtp, dtp, out_dir, tag, coco_gt, coco_dt):
    """TIDE 로 오차 6종 ΔAP 분해. 구간 전체 / test3+4 만 두 번 낸다."""
    try:
        from tidecv import TIDE
        import tidecv.datasets as td
    except Exception as e:  # noqa: BLE001
        return {"error": f"tidecv 없음: {e}"}
    out = {}
    (out_dir / "tide").mkdir(parents=True, exist_ok=True)
    subsets = {"all": None, "zone12": ("test1", "test2"), "zone34": ("test3", "test4")}
    for key, pref in subsets.items():
        if pref is None:
            g, d = coco_gt, coco_dt
        else:
            keep = {im["id"] for im in coco_gt["images"] if im["file_name"].split("_")[0] in pref}
            if not keep:
                continue
            g = {"images": [i for i in coco_gt["images"] if i["id"] in keep],
                 "annotations": [a for a in coco_gt["annotations"] if a["image_id"] in keep],
                 "categories": coco_gt["categories"]}
            d = [x for x in coco_dt if x["image_id"] in keep]
        gp = out_dir / "coco" / f"{tag}_{key}_gt.json"
        dp = out_dir / "coco" / f"{tag}_{key}_dt.json"
        gp.write_text(json.dumps(g), encoding="utf-8")
        dp.write_text(json.dumps(d), encoding="utf-8")
        try:
            import contextlib, io
            tide = TIDE()
            tide.evaluate(td.COCO(str(gp)), td.COCOResult(str(dp)), mode=TIDE.BOX, name=f"{tag}:{key}")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                tide.summarize()
            txt = buf.getvalue()
            (out_dir / "tide" / f"{tag}_{key}.txt").write_text(txt, encoding="utf-8")
            # 지표는 summarize() 표를 파싱해 얻는다. run.error_dict 는 tidecv 판마다 값 형태가
            # 달라(스칼라/리스트) 바로 float() 하면 깨진다.
            out[key] = {"ap50": _parse_ap(txt), "errors": _parse_tide(txt)}
        except Exception as e:  # noqa: BLE001
            out[key] = {"error": f"{type(e).__name__}: {e}"}
    return out


def _parse_ap(text):
    """'bbox AP @ 50: 17.16' 줄에서 AP 를 뽑는다."""
    for ln in text.splitlines():
        if "AP @ 50" in ln and ":" in ln:
            try:
                return round(float(ln.split(":")[-1].strip()), 2)
            except ValueError:
                pass
    return None


def _parse_tide(text):
    """summarize() 표의 dAP 행에서 Cls/Loc/Both/Dupe/Bkg/Miss 를 뽑는다.
    ΔAP = 그 오차를 전부 고쳤을 때 올라가는 AP. 무엇을 고쳐야 이득이 큰지를 알려준다."""
    keys = ["Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"]
    lines = text.splitlines()
    out = {}
    for i, ln in enumerate(lines):
        if ln.split()[:1] == ["Type"] and all(k in ln.split() for k in keys):
            hdr = ln.split()                      # ['Type','Cls','Loc','Both','Dupe','Bkg','Miss']
            for j in range(i + 1, min(i + 4, len(lines))):
                v = lines[j].split()
                if v[:1] == ["dAP"] and len(v) == len(hdr):
                    out = {k: float(x) for k, x in zip(hdr[1:], v[1:])}
                    break
            if out:
                break
    # 특수 오차(FalsePos/FalseNeg)도 같이
    for i, ln in enumerate(lines):
        if ln.split()[:1] == ["Type"] and "FalsePos" in ln:
            hdr = ln.split()
            for j in range(i + 1, min(i + 4, len(lines))):
                v = lines[j].split()
                if v[:1] == ["dAP"] and len(v) == len(hdr):
                    out.update({k: float(x) for k, x in zip(hdr[1:], v[1:])})
                    break
    return out


def _isnum(s):
    try:
        float(s); return True
    except ValueError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--models", nargs="*", help="tag=경로 형식. 생략 시 기본 목록")
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    out = Path(a.out) if a.out else BASE / "results/diag" / datetime.now().strftime("%y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    models = DEFAULT_MODELS.copy()
    if a.models:
        models = {}
        for s in a.models:
            t, _, p = s.partition("=")
            models[t] = p
    gts = gt_index()
    print(f"GT: " + " · ".join(f"{z} {len(gts[z])}장" for z in ZONES), flush=True)
    rows = []
    for tag, rel in models.items():
        wp = Path(rel) if Path(rel).is_absolute() else BASE / rel
        if not wp.exists():
            print(f"  건너뜀(없음): {wp}"); continue
        print(f"\n[{tag}] {wp}", flush=True)
        r = run_model(tag, wp, gts, out, device=a.device)
        rows.append(r)
        for z in ZONES:
            zz = r["zones"][z]
            print(f"  {z}: aware AP50 {zz['aware']['ap50']:.3f} hit@.25 {zz['aware']['hit_n']['0.25']}/10"
                  f" | agnostic AP50 {zz['agnostic']['ap50']:.3f} hit@.25 {zz['agnostic']['hit_n']['0.25']}/10"
                  f" | GT크기 {zz['gt_px_at640']}px@640  {zz.get('agnostic_called_it', {})}", flush=True)
        (out / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(rows, out)
    print(f"\nDONE {out}")


def write_summary(rows, out):
    L = []
    L.append(f"도메인갭 진단 — 분류실패 vs 검출실패  ({datetime.now():%Y-%m-%d %H:%M})")
    L.append(f"GT: data/bell412/gearbox/gt · 구간별 10장 · IoU>=0.5 · AP50 은 conf 전구간")
    L.append("")
    L.append("[1] 구간별 맞춤 개수 (conf 0.25) — aware = 이름까지 맞음, agnostic = 박스만 맞음")
    L.append(f"{'모델':<20}{'클래스':>6}  " + "".join(f"{z+' aware/agn':>18}" for z in ZONES))
    for r in rows:
        cells = ""
        for z in ZONES:
            zz = r["zones"][z]
            cells += f"{zz['aware']['hit_n']['0.25']:>8}/{zz['agnostic']['hit_n']['0.25']:<10}"
        L.append(f"{r['tag']:<20}{r['n_classes']:>6}  {cells}")
    L.append("")
    L.append("[2] 구간별 AP50 (aware / agnostic)")
    L.append(f"{'모델':<20}  " + "".join(f"{z:>20}" for z in ZONES))
    for r in rows:
        cells = ""
        for z in ZONES:
            zz = r["zones"][z]
            cells += f"{zz['aware']['ap50']:>9.3f}/{zz['agnostic']['ap50']:<11.3f}"
        L.append(f"{r['tag']:<20}  {cells}")
    L.append("")
    L.append("[3] 신뢰도 임계별 맞춤 개수 /10  (agnostic 기준 = 박스라도 만들었나)")
    for r in rows:
        L.append(f"  {r['tag']}")
        for z in ZONES:
            hz = r["zones"][z]["agnostic"]["hit_n"]
            L.append(f"    {z}: " + "  ".join(f"conf{c} {hz[str(c)]}" for c in CONFS))
    L.append("")
    L.append("[4] agnostic 으로 맞춘 박스를 모델이 부른 이름 (오분류 실태)")
    for r in rows:
        L.append(f"  {r['tag']}")
        for z in ZONES:
            L.append(f"    {z}: {r['zones'][z].get('agnostic_called_it', {})}")
    L.append("")
    L.append("[5] TIDE 오차 분해 — gearbox 단일클래스 기준(부분라벨 왜곡 없음). ΔAP = 그 오차만 고쳤을 때 오르는 AP")
    L.append(f"{'모델':<20}{'구간':<8}{'AP50':>7}{'Cls':>8}{'Loc':>7}{'Dupe':>7}{'Bkg':>7}{'Miss':>7}   개수")
    for r in rows:
        for k in ("zone12", "zone34", "test3", "test4"):
            t = (r.get("tide_1cls") or {}).get(k) or {}
            if not t:
                continue
            L.append(f"{r['tag']:<20}{k:<8}{t['_ap50']:>7.1f}{t['Cls']:>8.2f}{t['Loc']:>7.2f}"
                     f"{t['Dupe']:>7.2f}{t['Bkg']:>7.2f}{t['Miss']:>7.2f}   {t['_counts']}")
    L.append("")
    L.append("[5b] 참고: tidecv 원본 출력(카테고리 36종 등록 -> 미라벨 물체가 전부 Bkg 로 잡혀 AP 왜곡됨)")
    for r in rows:
        L.append(f"  {r['tag']}")
        for k, v in (r.get("tide") or {}).items():
            L.append(f"    {k}: {v}")
    L.append("")
    L.append("[6] GT 객체 크기(640 입력 기준 평균 변길이 px) — 작을수록 소형객체")
    if rows:
        L.append("    " + "  ".join(f"{z} {rows[0]['zones'][z]['gt_px_at640']}px" for z in ZONES))
    (out / "summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
