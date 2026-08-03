# -*- coding: utf-8 -*-
"""dashboard_core.py - 대시보드 공용 로직 (UI 프레임워크 무관).

verify/dashboard_api.py(FastAPI)와 verify/build_report.py(HTML 내보내기)가 공유하는
데이터 계층. **콘텐츠(방법 설명·flow·코드조각·정적표·용어·실험)는 코드에 박지 않고
dashboard_content.yaml 에 둔다.** 이 파일은 그 콘텐츠를 읽어 렌더하고, 결과 json 파싱·
즉석 비교 추론 같은 '로직'만 담는다. 콘텐츠만 바꾸면 다른 프로젝트에도 재사용 가능.
"""
import base64
import json
from pathlib import Path

import cv2
import yaml

import config

CONTENT = yaml.safe_load((Path(__file__).resolve().parent / "dashboard_content.yaml").read_text(encoding="utf-8"))

# 프로젝트별 경로·상수도 콘텐츠에서 (코드 재사용 위해)
_P = CONTENT.get("paths", {})
PREV_DIR = config.BASE_DIR / _P.get("previews", "dashboard/previews")
TEST_IMG = config.BASE_DIR / _P.get("test_images", "data/robo/yolo/test/images")
TEST_LBL = config.BASE_DIR / _P.get("test_labels", "data/robo/yolo/test/labels")
GT_CLASSES = CONTENT.get("gt_classes", [])
CLASS_COLORS = {k: tuple(v) for k, v in CONTENT.get("class_colors", {}).items()}
TECH_DEFS = CONTENT.get("tech_defs", {})
METHODS = CONTENT.get("methods", [])
GLOSSARY = CONTENT.get("glossary", [])
EXPERIMENTS = CONTENT.get("experiments", {})
_AUTOLEARN_CONDS = CONTENT.get("autolearn_conditions", [])


def jload(rel):
    p = config.BASE_DIR / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ==================== 지표 로더 (결과 json → 표) ====================
def load_autolearn(rel):
    d = jload(rel)
    if not d:
        return None
    sp, ps = d.get("split", {}), d.get("pseudo", {})
    r0, r1 = d.get("round0", {}), d.get("round1", {})

    def delta(a, b):
        return f"+{round((b - a) * 100, 1)}%p" if a is not None and b is not None else ""

    # 표 1: 데이터·자동 라벨 (설정과 자동 라벨 품질)
    data_rows = [
        ["데이터 구성", f"1차 라벨 {sp.get('seed')}장 / 무라벨 {sp.get('pool')}장 / 평가셋 {sp.get('test')}장"],
        ["자동 라벨 생성", f"{ps.get('labeled_images')}장 (박스 {ps.get('boxes')}개)"],
        ["자동 라벨 정밀도 / 재현율", f"{ps.get('precision')} / {ps.get('recall')}"],
    ]
    # 표 2: 모델 성능 (1차 vs 2차)
    perf_rows = [
        ["mAP50", r0.get("map50"), r1.get("map50"), delta(r0.get("map50"), r1.get("map50"))],
        ["mAP50-95", r0.get("map50_95"), r1.get("map50_95"), delta(r0.get("map50_95"), r1.get("map50_95"))],
    ]
    # 표 3: 클래스별 정확도(mAP50-95)
    pc0, pc1 = r0.get("per_class_map50_95", {}), r1.get("per_class_map50_95", {})
    cls_rows = [[cls, pc0.get(cls), pc1.get(cls), delta(pc0.get(cls), pc1.get(cls))] for cls in pc1]

    subtables = [{"title": "모델 성능 (1차 → 2차)",
                  "headers": ["지표", "1차 모델", "2차 모델", "변화율"], "rows": perf_rows}]
    if cls_rows:
        subtables.append({"title": "클래스별 정확도 (mAP50-95)",
                          "headers": ["클래스", "1차 모델", "2차 모델", "변화율"], "rows": cls_rows})
    d50 = delta(r0.get("map50"), r1.get("map50"))
    summary = "\n".join([
        f"자동 생성된 2차 라벨 {ps.get('labeled_images')}장, 정밀도 {ps.get('precision')} / 재현율 {ps.get('recall')}",
        f"1차 모델 mAP50 {r0.get('map50')} → 2차 모델 {r1.get('map50')} ({d50 or '변화 미미'})",
    ])
    return (["항목", "값"], data_rows, summary, subtables)


def load_zeroshot(rel):
    d = jload(rel)
    if not d:
        return None
    rows = [[c, s.get("precision"), s.get("recall"), s.get("tp"), s.get("fp"), s.get("fn")]
            for c, s in d.get("per_class", {}).items()]
    m = d.get("micro", {})
    rows.append(["전체", m.get("precision"), m.get("recall"), m.get("tp"), m.get("fp"), m.get("fn")])
    extra = f" / 맞은 박스 평균 IoU {d['mean_tp_iou']}" if d.get("mean_tp_iou") else ""
    return (["클래스", "정밀도", "재현율", "맞음(TP)", "오탐(FP)", "누락(FN)"], rows,
            f"정밀도 {m.get('precision')}{extra} -> 무검수 라벨 기준(0.87) " +
            ("충족" if (m.get("precision") or 0) >= 0.87 else "미달"))


def load_sweep(rel):
    d = jload(rel)
    if not d:
        return None
    # 기본 스윕(격차 규칙 없는 margin 0)만, 대표 임계값 4개로 압축
    base = [s for s in d.get("sweep", []) if s.get("margin", 0.0) == 0.0]
    key = (0.6, 0.7, 0.8, 0.85)
    rows = [[s["tau"], s["precision"], s["recall"]] for s in base if s["tau"] in key]
    bp = max(base, key=lambda s: s["precision"]) if base else {}
    p = bp.get("precision", 0)
    ok = "충족" if p >= 0.87 else "미달"
    summary = "\n".join([
        "유사도 임계값을 높이면 정밀도(맞은 라벨 비율)는 오르고 재현율(찾아낸 비율)은 내려간다",
        f"최고 정밀도는 {p} (임계값 {bp.get('tau')})",
        f"사람 검수 없이 쓸 기준(정밀도 0.87)에는 {ok}",
    ])
    return (["유사도 임계값", "정밀도", "재현율"], rows, summary)


def load_bench(rel, summary):
    d = jload(rel)
    if not d:
        return None
    rows = [[r["model"], r["imgsz"], r["map50"], r["map50_95"], r["latency_ms"], r["fps"],
             r["weight_MB"], r["train_min"]] for r in d]
    return (["모델", "입력", "mAP50", "mAP50-95", "지연(ms)", "FPS", "크기(MB)", "학습(분)"], rows, summary)


def load_followup(rel, summary):
    d = jload(rel)
    if not d:
        return None
    rows = [[r.get("name"), r.get("base"), f"{r.get('epochs_run')}/{r.get('epochs_set')}",
             r.get("map50"), r.get("map50_95"), r.get("train_min")] for r in d]
    return (["실험", "기반 모델", "epochs(실행/설정)", "mAP50", "mAP50-95", "학습(분)"], rows, summary)


# 방법 지표용 로더(파일 → 표). static 은 아래 resolve_metrics 에서 직접 처리.
METHOD_LOADERS = {"autolearn": load_autolearn, "zeroshot": load_zeroshot, "sweep": load_sweep}
# 실험 지표용 로더(파일 + 요약문 → 표).
EXP_LOADERS = {"bench": load_bench, "followup": load_followup}


# ==================== 기술 정의 ====================
def resolve_tech(m):
    """(이름[, 용도]) 목록 -> [{name, desc(공통 정의), usage(이 방법에서의 쓰임)}]."""
    out = []
    for entry in m.get("tech", []):
        name = entry[0]
        usage = entry[1] if len(entry) > 1 else ""
        out.append({"name": name, "desc": TECH_DEFS.get(name, ""), "usage": usage})
    return out


# ==================== 방법 레지스트리 조회 ====================
def method_by_id(mid):
    for m in METHODS:
        if m["id"] == mid:
            return m
    return None


def resolve_metrics(spec):
    """콘텐츠의 metrics 스펙 -> (headers, rows, summary[, subtables]).

    spec = {loader: static, headers?, rows, summary, subtables?}  또는
           {loader: autolearn|zeroshot|sweep, file: <상대경로>}
    """
    if not spec:
        return None
    kind = spec.get("loader")
    if kind == "static":
        return (spec.get("headers", ["항목", "결과"]), spec.get("rows", []),
                spec.get("summary", ""), spec.get("subtables", []))
    fn = METHOD_LOADERS.get(kind)
    return fn(spec.get("file")) if fn else None


def method_metrics(m):
    res = resolve_metrics(m.get("metrics"))
    if res is None:
        return {"headers": ["항목"], "rows": [], "summary": "결과 파일 없음", "subtables": []}
    headers, rows, summary = res[:3]
    subtables = res[3] if len(res) > 3 else []
    return {"headers": headers, "rows": rows, "summary": summary, "subtables": subtables}


def autolearn_conditions_table():
    """방법 1 콜아웃: 조건(부품 수·1차 라벨 비율·TTA)별 오토러닝 결과 요약표."""
    rows = []
    for name, rel in _AUTOLEARN_CONDS:
        d = jload(rel)
        if not d:
            continue
        ps, r0, r1 = d.get("pseudo", {}), d.get("round0", {}), d.get("round1", {})
        a, b = r0.get("map50"), r1.get("map50")
        dl = f"+{round((b - a) * 100, 1)}%p" if a is not None and b is not None else ""
        rows.append([name, ps.get("precision"), a, b, dl])
    return {"headers": ["조건", "자동 라벨 정밀도", "1차 모델 mAP50", "2차 모델 mAP50", "변화"], "rows": rows}


# 방법 결과 뒤에 붙일 관련 표를 이름으로 만드는 함수 맵 (콘텐츠의 extras_table 로 지정)
EXTRA_TABLES = {"autolearn_conditions": autolearn_conditions_table}


def method_extras(m):
    """방법 결과 뒤에 콜아웃으로 붙일 관련 실험들. [{title, desc, table}]"""
    key = m.get("extras_table")
    if key and key in EXTRA_TABLES:
        return [{"title": "조건별 오토러닝 결과", "desc": "", "table": EXTRA_TABLES[key]()}]
    return m.get("extras", [])


def method_gallery(m):
    sub = m.get("gallery")
    if not sub:
        return []
    d = PREV_DIR / sub
    return [{"url": f"/previews/{sub}/{p.name}",
             "caption": p.stem.split("_", 1)[1].replace("_", " ")}
            for p in sorted(d.glob("*.jpg"))]


# ==================== 즉석 비교 (모델 박스 | 정답 박스) ====================
def get_model():
    from ultralytics import YOLO
    if not config.SERVE_MODEL.exists():
        return None
    if not hasattr(get_model, "m"):
        get_model.m = YOLO(str(config.SERVE_MODEL))
    return get_model.m


def _b64(im, max_w=900, q=85):
    h, w = im.shape[:2]
    if w > max_w:
        im = cv2.resize(im, (max_w, int(h * max_w / w)))
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def color_for(name):
    return CLASS_COLORS.get(str(name).lower(), (128, 128, 128))


def render_detections(img, dets, thick=2):
    """dets = [(x1,y1,x2,y2, label, color)]. 박스 그리고, 라벨을 박스에 붙여서 배치.

    라벨은 자기 박스에 붙는 4개 후보 위치(위/아래/안쪽 위/안쪽 아래)만 시도하고,
    그중 다른 라벨과 겹침이 가장 적은 자리를 고른다. 멀리 떨어져 뜨지 않는다.
    글씨는 작게(scale 0.5) 해서 점유를 줄인다.
    """
    font, scale, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    H, W = img.shape[:2]
    for x1, y1, x2, y2, _, color in dets:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)

    placed = []

    def overlap_area(r):
        a = 0
        for q in placed:
            ix = max(0, min(r[2], q[2]) - max(r[0], q[0]))
            iy = max(0, min(r[3], q[3]) - max(r[1], q[1]))
            a += ix * iy
        return a

    for x1, y1, x2, y2, label, color in sorted(dets, key=lambda d: d[1]):
        (tw, th), bl = cv2.getTextSize(label, font, scale, ft)
        lh, lw = th + bl + 4, tw + 6
        lx = max(0, min(x1 - thick // 2, W - lw))
        # 자기 박스에 붙는 후보: 위 / 아래 / 안쪽 위 / 안쪽 아래
        best, best_area = None, None
        for top in (y1 - lh, y2, y1, y2 - lh):
            if top < 0 or top + lh > H:
                continue
            area = overlap_area((lx, top, lx + lw, top + lh))
            if area == 0:
                best = top
                break
            if best_area is None or area < best_area:
                best, best_area = top, area
        if best is None:
            best = max(0, min(y1 - lh, H - lh))
        cv2.rectangle(img, (lx, best), (lx + lw, best + lh), color, -1)
        cv2.putText(img, label, (lx + 3, best + th + 2), font, scale, (255, 255, 255), ft, cv2.LINE_AA)
        placed.append((lx, best, lx + lw, best + lh))


def test_images():
    return sorted(TEST_IMG.glob("*.jpg"))


def compare(idx=0, conf=0.6):
    m = get_model()
    if m is None:
        return {"error": "서빙 모델 없음 (models/new_model.pt)"}
    imgs = test_images()
    if not imgs:
        return {"error": "테스트 이미지 없음 (data/robo/yolo/test)"}
    idx = int(idx) % len(imgs)
    p = imgs[idx]
    src = cv2.imread(str(p))
    h, w = src.shape[:2]
    from collections import Counter
    pred = src.copy()
    r = m.predict(source=str(p), conf=conf, verbose=False)[0]
    pred_names, gt_names = [], []
    pred_dets = []
    for b, c, cf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
        x1, y1, x2, y2 = map(int, b)
        name = m.names[int(c)]
        pred_dets.append((x1, y1, x2, y2, f"{name} ({float(cf):.2f})", color_for(name)))
        pred_names.append(name)
    render_detections(pred, pred_dets)

    gt = src.copy()
    gt_dets = []
    lbl = TEST_LBL / f"{p.stem}.txt"
    if lbl.exists():
        for line in lbl.read_text().splitlines():
            f = line.split()
            if len(f) < 5:
                continue
            c, cx, cy, bw, bh = int(f[0]), *map(float, f[1:5])
            x1, y1 = round((cx - bw / 2) * w), round((cy - bh / 2) * h)
            x2, y2 = round((cx + bw / 2) * w), round((cy + bh / 2) * h)
            name = GT_CLASSES[c] if c < len(GT_CLASSES) else str(c)
            gt_dets.append((x1, y1, x2, y2, name, color_for(name)))
            gt_names.append(name)
    render_detections(gt, gt_dets)

    pc, gc = Counter(pred_names), Counter(gt_names)
    cats = sorted(set(pc) | set(gc))
    counts = [{"name": nm, "pred": pc.get(nm, 0), "gt": gc.get(nm, 0)} for nm in cats]
    legend = [{"name": nm, "color": "#%02x%02x%02x" % color_for(nm)[::-1]} for nm in cats]
    return {"pred": _b64(pred), "gt": _b64(gt), "file": p.name, "counts": counts,
            "legend": legend, "idx": idx, "total": len(imgs)}


def experiment_metrics(cat, topic):
    entry = EXPERIMENTS.get(cat, {}).get(topic)
    if not entry:
        return {"headers": [], "rows": [], "summary": "선택하세요.", "desc": ""}
    desc = entry.get("desc", "")
    if entry.get("loader") == "static":            # 파일 없이 rows 직접(정적 표)
        res = resolve_metrics(entry)
    else:
        fn = EXP_LOADERS.get(entry.get("loader"))
        res = fn(entry.get("file"), entry.get("summary", "")) if fn else None
    if res is None:
        return {"headers": [], "rows": [], "summary": f"결과 파일 없음: {entry.get('file')}",
                "subtables": [], "desc": desc}
    headers, rows, summary = res[:3]
    subtables = res[3] if len(res) > 3 else []
    return {"headers": headers, "rows": rows, "summary": summary, "subtables": subtables, "desc": desc}


def export_report():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "report_builder", Path(__file__).resolve().parent / "build_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.build()
    return str(mod.OUT)
