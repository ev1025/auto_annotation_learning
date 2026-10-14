# -*- coding: utf-8 -*-
"""사람 GT 70장 표준 채점기 — 모든 실험을 같은 자로 잰다.

평가셋(사람이 그린 GT 만 쓴다)
    data/bell412/gearbox/gt   40장  test1·2 = 작업대 단품(도메인 안) · test3·4 = 기체 장착(도메인 밖)
    data/bell412/a_test/gt    30장  사무실에서 손에 든 롱노즈(또 다른 도메인)
  합계 70장 · 2클래스. 학습 로그의 mAP 는 학습셋 재채점이라 0.99 로 포화돼 아무것도 구분 못 한다.

내는 값
    mAP50 / mAP50-95   COCO 101점 보간, 두 클래스 평균
    aware              이름까지 맞아야 정답
    agnostic           클래스 무시(박스만 맞으면 정답). aware 와의 차 = 순수 분류 실패분
    구간표             test1~4 · a_test 별로 어디서 무너지는지
    TIDE 식 ΔAP        Cls·Loc·Dupe·Bkg·Miss 중 무엇이 점수를 깎는지

1클래스로 병합 학습한 모델(Plan B 1단계 검출기)도 그대로 넣으면 된다. aware 가 성립하지
않으므로 agnostic 열만 읽는다(클래스 이름이 GT 와 안 맞으면 자동으로 merged 로 표시).

사용
  XR_BASE=$PWD CUDA_VISIBLE_DEVICES=2 ./venv/bin/python scripts/experiments/eval_gt70.py \
      base=results/bench/260824_retrain3x/yolo11m/weights/best.pt \
      real=results/bench/260829_realonly/yolo11m/weights/best.pt --out results/gt70/260829
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get("XR_BASE") or Path(__file__).resolve().parents[2])
GT_SETS = {"gearbox": BASE / "data/bell412/gearbox/gt", "a_test": BASE / "data/bell412/a_test/gt"}
ZONE_ORDER = ["test1", "test2", "test3", "test4", "a_test"]
IOUS = np.arange(0.5, 1.0, 0.05)
CONFS = [0.7, 0.5, 0.25, 0.10, 0.01]
POS_T, BG_T = 0.5, 0.1


# ---------------- GT ----------------
def load_gt():
    import cv2
    items = []
    for cls, d in GT_SETS.items():
        for ip in sorted(glob.glob(str(Path(d) / "images/*.jpg"))):
            stem = Path(ip).stem
            lp = Path(d) / "labels" / f"{stem}.txt"
            if not lp.exists() or not lp.read_text(encoding="utf-8").strip():
                continue                       # 라벨 없는 사진은 뺀다(배경으로 오해되면 수치 왜곡)
            h, w = cv2.imread(ip).shape[:2]
            boxes = []
            for ln in lp.read_text(encoding="utf-8").splitlines():
                f = ln.split()
                if len(f) >= 5:
                    cx, cy, bw, bh = map(float, f[1:5])
                    boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
            if boxes:
                items.append({"path": ip, "gt_cls": cls, "boxes": boxes, "wh": (w, h),
                              "zone": stem.split("_")[0] if cls == "gearbox" else "a_test"})
    return items


# ---------------- 공통 도구 ----------------
def iou_mat(dets, gts):
    if not len(dets) or not len(gts):
        return np.zeros((len(dets), len(gts)), np.float32)
    d = np.asarray(dets, np.float32)[:, None, :]
    g = np.asarray(gts, np.float32)[None, :, :]
    iw = np.clip(np.minimum(d[..., 2], g[..., 2]) - np.maximum(d[..., 0], g[..., 0]), 0, None)
    ih = np.clip(np.minimum(d[..., 3], g[..., 3]) - np.maximum(d[..., 1], g[..., 1]), 0, None)
    inter = iw * ih
    ad = (d[..., 2] - d[..., 0]) * (d[..., 3] - d[..., 1])
    ag = (g[..., 2] - g[..., 0]) * (g[..., 3] - g[..., 1])
    return inter / np.maximum(ad + ag - inter, 1e-9)


def ap_from(scores, matched, n_gt):
    if n_gt == 0:
        return float("nan")
    if not len(scores):
        return 0.0
    o = np.argsort(-np.asarray(scores, np.float64))
    m = np.asarray(matched, bool)[o]
    tp = np.cumsum(m); fp = np.cumsum(~m)
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.mean(np.interp(np.linspace(0, 1, 101), rec, prec, left=prec[0], right=0.0)))


def match_at(dets, scores, gts, thr):
    """점수 내림차순 그리디 매칭 -> (정렬된 점수, 매칭여부)"""
    order = np.argsort(-np.asarray(scores)) if len(scores) else np.array([], int)
    M = iou_mat([dets[i] for i in order], gts) if len(order) else np.zeros((0, len(gts)))
    used, mm = set(), np.zeros(len(order), bool)
    for k in range(len(order)):
        best, bj = thr, -1
        for j in range(len(gts)):
            if j not in used and M[k, j] >= best:
                best, bj = M[k, j], j
        if bj >= 0:
            used.add(bj); mm[k] = True
    return np.asarray(scores)[order] if len(order) else np.array([]), mm


def ap_pair(per_img, n_gt):
    """per_img: [(dets, scores, gts)] -> (AP50, AP50-95)"""
    aps = []
    for thr in IOUS:
        sc, mt = [], []
        for dets, scores, gts in per_img:
            if not len(dets):
                continue
            s, m = match_at(dets, scores, gts, thr)
            sc += list(s); mt += list(m)
        aps.append(ap_from(sc, mt, n_gt))
    return aps[0], float(np.nanmean(aps))


# ---------------- TIDE 식 분해 ----------------
def tide_decompose(rows, cid):
    """cid=None 이면 클래스 구분 없이(=병합 모델) 본다."""
    dets, scores, dcls, gt_all, offs = [], [], [], [], []
    for r in rows:
        offs.append(len(gt_all))
        gt_all += [list(b) for b in r["boxes"]]
        for b, c, cl in zip(r["dxyxy"], r["dconf"], r["dcls"]):
            dets.append(list(b)); scores.append(float(c)); dcls.append(int(cl))
    n_gt = len(gt_all)
    if n_gt == 0 or not dets:
        return {}
    M = np.zeros((len(dets), n_gt), np.float32)
    di = 0
    for i, r in enumerate(rows):                       # 사진 경계를 넘는 매칭 차단
        nd = len(r["dxyxy"])
        if nd and r["boxes"]:
            M[di:di + nd, offs[i]:offs[i] + len(r["boxes"])] = iou_mat(r["dxyxy"], r["boxes"])
        di += nd
    scores = np.asarray(scores, np.float32); dcls = np.asarray(dcls, int)
    kind = np.array(["?"] * len(dets), dtype=object)
    bgt = np.full(len(dets), -1)
    used = set()
    for k in np.argsort(-scores):
        j = int(np.argmax(M[k])); iou = float(M[k, j]); bgt[k] = j
        if cid is None or dcls[k] == cid:
            free = [x for x in range(n_gt) if x not in used and M[k, x] >= POS_T]
            if free:
                x = max(free, key=lambda t: M[k, t]); used.add(x); bgt[k] = x; kind[k] = "TP"
            elif iou >= POS_T:
                kind[k] = "Dupe"
            elif iou >= BG_T:
                kind[k] = "Loc"
            else:
                kind[k] = "Bkg"
        else:
            kind[k] = "Cls" if iou >= POS_T else ("Both" if iou >= BG_T else "Other")
    near = {int(bgt[k]) for k in range(len(dets)) if kind[k] in ("Cls", "Loc", "Both")}
    missed = [j for j in range(n_gt) if j not in used and j not in near]
    own = [k for k in range(len(dets)) if cid is None or dcls[k] == cid]

    def ap_of(idx, override=None, gts=None, n=None):
        gts = gt_all if gts is None else gts
        n = n_gt if n is None else n
        if not idx or n == 0:
            return 0.0
        Mi = iou_mat([dets[k] for k in idx], gts)
        if override:
            for r, k in enumerate(idx):
                if k in override and 0 <= bgt[k] < len(gts):
                    Mi[r, bgt[k]] = 1.0
        sc = scores[idx]; order = np.argsort(-sc); usd = set(); mm = np.zeros(len(idx), bool)
        for r, k in enumerate(order):
            best, bj = POS_T, -1
            for j in range(len(gts)):
                if j not in usd and Mi[k, j] >= best:
                    best, bj = Mi[k, j], j
            if bj >= 0:
                usd.add(bj); mm[r] = True
        return ap_from(sc[order], mm, n)

    base = ap_of(own)
    cls_e = [k for k in range(len(dets)) if kind[k] == "Cls"]
    loc_e = [k for k in own if kind[k] == "Loc"]
    dup_e = [k for k in own if kind[k] == "Dupe"]
    bkg_e = [k for k in own if kind[k] == "Bkg"]
    keep = [g for j, g in enumerate(gt_all) if j not in missed]
    return {"ap50": round(100 * base, 2),
            "Cls": round(100 * (ap_of(own + cls_e) - base), 2),
            "Loc": round(100 * (ap_of(own, override=set(loc_e)) - base), 2),
            "Dupe": round(100 * (ap_of([k for k in own if k not in dup_e]) - base), 2),
            "Bkg": round(100 * (ap_of([k for k in own if k not in bkg_e]) - base), 2),
            "Miss": round(100 * (ap_of(own, gts=keep, n=len(keep)) - base), 2) if missed else 0.0,
            "counts": {"TP": int((kind == "TP").sum()), "Cls": len(cls_e), "Loc": len(loc_e),
                       "Dupe": len(dup_e), "Bkg": len(bkg_e), "Miss": len(missed), "n_gt": n_gt}}


# ---------------- 모델 채점 ----------------
def predict_rows(wpath, gt_items, device="0"):
    from ultralytics import YOLO
    import gc, torch
    m = YOLO(str(wpath))
    names = {int(k): v for k, v in m.names.items()}
    paths = [g["path"] for g in gt_items]
    preds = []
    for i in range(0, len(paths), 20):
        preds += m.predict(paths[i:i + 20], conf=0.001, iou=0.7, imgsz=640,
                           max_det=300, device=device, verbose=False)
    rows = []
    for g, r in zip(gt_items, preds):
        b = r.boxes
        rows.append({**g,
                     "dxyxy": b.xyxy.cpu().numpy().tolist() if b is not None else [],
                     "dconf": b.conf.cpu().numpy().tolist() if b is not None else [],
                     "dcls": [int(c) for c in (b.cls.cpu().numpy().tolist() if b is not None else [])]})
    del m; gc.collect(); torch.cuda.empty_cache()
    return rows, names


def score(tag, wpath, gt_items, device="0"):
    rows, names = predict_rows(wpath, gt_items, device)
    gt_classes = sorted({g["gt_cls"] for g in gt_items})
    cid_of = {c: next((i for i, n in names.items() if n == c), None) for c in gt_classes}
    merged = all(v is None for v in cid_of.values())      # 클래스 이름이 하나도 안 맞으면 병합 모델
    res = {"tag": tag, "weights": str(wpath), "n_classes": len(names), "merged": merged,
           "cls_index": cid_of, "per_class": {}, "per_zone": {}, "overall": {}, "tide": {},
           "called_it": {}}

    for mode in ("aware", "agnostic"):
        # 클래스별
        per_cls50, per_cls5095 = {}, {}
        for c in gt_classes:
            cid = cid_of[c]
            if mode == "aware" and cid is None:
                per_cls50[c] = float("nan"); per_cls5095[c] = float("nan"); continue
            sub = [r for r in rows if r["gt_cls"] == c]
            per_img, n_gt = [], 0
            for r in sub:
                keep = [k for k in range(len(r["dconf"])) if mode == "agnostic" or r["dcls"][k] == cid]
                per_img.append(([r["dxyxy"][k] for k in keep], [r["dconf"][k] for k in keep], r["boxes"]))
                n_gt += len(r["boxes"])
            a50, a5095 = ap_pair(per_img, n_gt)
            per_cls50[c] = round(a50, 4); per_cls5095[c] = round(a5095, 4)
        res["per_class"][mode] = {"ap50": per_cls50, "ap50_95": per_cls5095}
        vals50 = [v for v in per_cls50.values() if not np.isnan(v)]
        vals95 = [v for v in per_cls5095.values() if not np.isnan(v)]
        res["overall"][mode] = {"map50": round(float(np.mean(vals50)), 4) if vals50 else float("nan"),
                                "map50_95": round(float(np.mean(vals95)), 4) if vals95 else float("nan")}
        # 구간별
        zres = {}
        for z in ZONE_ORDER:
            sub = [r for r in rows if r["zone"] == z]
            if not sub:
                continue
            c = sub[0]["gt_cls"]; cid = cid_of[c]
            if mode == "aware" and cid is None:
                continue
            per_img, n_gt, hit = [], 0, {cf: 0 for cf in CONFS}
            for r in sub:
                keep = [k for k in range(len(r["dconf"])) if mode == "agnostic" or r["dcls"][k] == cid]
                d = [r["dxyxy"][k] for k in keep]; s = [r["dconf"][k] for k in keep]
                per_img.append((d, s, r["boxes"])); n_gt += len(r["boxes"])
                sc, mm = match_at(d, s, r["boxes"], POS_T) if d else (np.array([]), np.zeros(0, bool))
                top = max([sc[i] for i in range(len(sc)) if mm[i]], default=0.0)
                for cf in CONFS:
                    hit[cf] += int(top >= cf)
            a50, a5095 = ap_pair(per_img, n_gt)
            zres[z] = {"n_img": len(sub), "n_gt": n_gt, "ap50": round(a50, 4), "ap50_95": round(a5095, 4),
                       "hit": {str(cf): hit[cf] for cf in CONFS}}
        res["per_zone"][mode] = zres

    # 오분류 실태: agnostic 으로 맞춘 박스를 모델이 뭐라고 불렀나
    for z in ZONE_ORDER:
        sub = [r for r in rows if r["zone"] == z]
        if not sub:
            continue
        cnt = Counter()
        for r in sub:
            sc, mm = match_at(r["dxyxy"], r["dconf"], r["boxes"], POS_T) if r["dxyxy"] else (np.array([]), np.zeros(0, bool))
            order = np.argsort(-np.asarray(r["dconf"])) if r["dconf"] else []
            best, bi = -1, None
            for i in range(len(sc)):
                if mm[i] and sc[i] > best:
                    best, bi = sc[i], int(order[i])
            if bi is not None and best >= 0.25:
                cnt[names.get(r["dcls"][bi], "?")] += 1
        res["called_it"][z] = dict(cnt.most_common())

    # TIDE
    for c in gt_classes:
        sub = [r for r in rows if r["gt_cls"] == c]
        res["tide"][c] = tide_decompose(sub, cid_of[c] if not merged else None)
    for z in ("test3", "test4"):
        sub = [r for r in rows if r["zone"] == z]
        if sub:
            res["tide"][z] = tide_decompose(sub, cid_of[sub[0]["gt_cls"]] if not merged else None)
    return res


# ---------------- 출력 ----------------
def report(rows_out, out: Path):
    L = [f"GT 70장 채점  ({datetime.now():%Y-%m-%d %H:%M})",
         "평가셋: gearbox 40(test1·2 작업대 / test3·4 기체장착) + a_test 30(사무실 롱노즈) · conf 전구간",
         ""]
    L.append("[1] 종합 mAP (두 클래스 평균)")
    L.append(f"{'모델':<22}{'클래스':>5}{'mAP50':>9}{'mAP50-95':>10}{'agn mAP50':>11}{'agn 50-95':>11}"
             f"{'gearbox':>9}{'a_test':>9}")
    for r in rows_out:
        o, oa = r["overall"]["aware"], r["overall"]["agnostic"]
        pc = r["per_class"]["aware"]["ap50"]
        L.append(f"{r['tag']:<22}{r['n_classes']:>5}{o['map50']:>9.4f}{o['map50_95']:>10.4f}"
                 f"{oa['map50']:>11.4f}{oa['map50_95']:>11.4f}"
                 f"{pc.get('gearbox', float('nan')):>9.4f}{pc.get('a_test', float('nan')):>9.4f}")
    L.append("")
    L.append("[2] 구간별 AP50 — aware(이름까지) / agnostic(박스만)")
    L.append(f"{'모델':<22}" + "".join(f"{z:>19}" for z in ZONE_ORDER))
    for r in rows_out:
        cells = ""
        for z in ZONE_ORDER:
            a = r["per_zone"]["aware"].get(z); g = r["per_zone"]["agnostic"].get(z)
            av = f"{a['ap50']:.3f}" if a else "  -  "
            gv = f"{g['ap50']:.3f}" if g else "  -  "
            cells += f"{av+'/'+gv:>19}"
        L.append(f"{r['tag']:<22}{cells}")
    L.append("")
    L.append("[3] 구간별 '한 장이라도 맞춘 사진 수' (agnostic · IoU>=0.5)")
    for r in rows_out:
        L.append(f"  {r['tag']}")
        for z in ZONE_ORDER:
            g = r["per_zone"]["agnostic"].get(z)
            if g:
                L.append(f"    {z:<7} n={g['n_img']:<3} " + "  ".join(f"conf{c} {g['hit'][str(c)]}" for c in CONFS))
    L.append("")
    L.append("[4] agnostic 으로 맞춘 박스를 모델이 부른 이름(conf>=0.25)")
    for r in rows_out:
        L.append(f"  {r['tag']}: " + " | ".join(f"{z} {r['called_it'].get(z, {})}" for z in ZONE_ORDER))
    L.append("")
    L.append("[5] TIDE 식 ΔAP — 그 오차만 고치면 오르는 AP(클수록 그게 주범)")
    L.append(f"{'모델':<22}{'대상':<10}{'AP50':>7}{'Cls':>8}{'Loc':>7}{'Dupe':>7}{'Bkg':>7}{'Miss':>8}   개수")
    for r in rows_out:
        for k, t in r["tide"].items():
            if not t:
                continue
            L.append(f"{r['tag']:<22}{k:<10}{t['ap50']:>7.1f}{t['Cls']:>8.2f}{t['Loc']:>7.2f}"
                     f"{t['Dupe']:>7.2f}{t['Bkg']:>7.2f}{t['Miss']:>8.2f}   {t['counts']}")
    txt = "\n".join(L)
    (out / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="tag=가중치경로")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    out = Path(a.out) if a.out else BASE / "results/gt70" / datetime.now().strftime("%y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    gt = load_gt()
    print(f"GT {len(gt)}장 · " + " · ".join(f"{z} {sum(1 for g in gt if g['zone'] == z)}" for z in ZONE_ORDER))
    rows_out = []
    for spec in a.models:
        tag, _, p = spec.partition("=")
        wp = Path(p) if Path(p).is_absolute() else BASE / p
        if not wp.exists():
            print(f"  건너뜀(없음): {wp}"); continue
        print(f"[{tag}] {wp}", flush=True)
        r = score(tag, wp, gt, a.device)
        rows_out.append(r)
        (out / "results.json").write_text(json.dumps(rows_out, ensure_ascii=False, indent=2), encoding="utf-8")
    report(rows_out, out)
    print(f"\nDONE {out}")


if __name__ == "__main__":
    main()
