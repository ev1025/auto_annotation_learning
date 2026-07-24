# -*- coding: utf-8 -*-
"""dashboard_core.py - 대시보드 공용 로직 (UI 프레임워크 무관).

verify/dashboard_api.py(FastAPI)와 verify/build_report.py(HTML 내보내기)가 공유하는
데이터 계층: 실험 결과 json 로더, 오토라벨링 방법 레지스트리, 즉석 비교 추론.
"""
import base64
import json
from pathlib import Path

import cv2

import config

PREV_DIR = config.BASE_DIR / "docs" / "method_previews"
TEST_IMG = config.DATA_DIR / "robo_yolo" / "test" / "images"
TEST_LBL = config.DATA_DIR / "robo_yolo" / "test" / "labels"
GT_CLASSES = ["bearing", "bolt", "gear", "nut"]
PALETTE = [(0, 255, 0), (255, 160, 0), (0, 160, 255), (255, 0, 200), (160, 255, 0),
           (0, 255, 255), (255, 80, 80), (180, 120, 255)]


def jload(rel):
    p = config.BASE_DIR / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ==================== 지표 로더 ====================
def load_autolearn(rel):
    d = jload(rel)
    if not d:
        return None
    sp, ps = d.get("split", {}), d.get("pseudo", {})
    r0, r1 = d.get("round0", {}), d.get("round1", {})
    rows = [
        ["데이터 구성", f"초기 라벨셋 {sp.get('seed')}장 / 미라벨 풀 {sp.get('pool')}장 / 평가셋 {sp.get('test')}장"],
        ["자동 라벨 생성", f"{ps.get('labeled_images')}장 (박스 {ps.get('boxes')}개)"],
        ["자동 라벨 정밀도 / 재현율", f"{ps.get('precision')} / {ps.get('recall')}"],
        ["1차 모델 mAP50 (초기 라벨만 학습)", r0.get("map50")],
        ["2차 모델 mAP50 (자동 라벨 추가 학습)", r1.get("map50")],
    ]
    pc0, pc1 = r0.get("per_class_map50_95", {}), r1.get("per_class_map50_95", {})
    for cls in pc1:
        v0, v1 = pc0.get(cls), pc1.get(cls)
        dv = f" (+{round((v1 - v0) * 100, 1)}%p)" if v0 is not None and v1 is not None else ""
        rows.append([f"└ {cls} 정확도 mAP50-95 (1차 → 2차)", f"{v0} → {v1}{dv}"])
    rows.append(["효과", f"mAP50 +{round(d.get('delta_map50', 0) * 100, 1)}%p / mAP50-95 +{round(d.get('delta_map50_95', 0) * 100, 1)}%p"])
    return (["항목", "값"], rows, "")


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
    has_margin = any("margin" in s for s in d.get("sweep", []))
    if has_margin:
        rows = [[s["tau"], s["margin"], s["precision"], s["recall"], s["f1"]] for s in d["sweep"]]
        headers = ["유사도 임계값", "1·2위 격차 조건", "정밀도", "재현율", "F1"]
    else:
        rows = [[s["tau"], s["precision"], s["recall"], s["f1"]] for s in d["sweep"]]
        headers = ["유사도 임계값", "정밀도", "재현율", "F1"]
    b = d.get("best_f1", {})
    hp = d.get("best_recall_at_p85")
    summary = f"균형점: 임계값 {b.get('tau')} -> P {b.get('precision')} / R {b.get('recall')}"
    summary += (f" / 고정밀 운영점: P {hp['precision']} R {hp['recall']} (임계값 {hp['tau']})"
                if hp else " / 정밀도 0.85 달성 지점 없음")
    return headers, rows, summary


def static_metrics(rows, summary):
    return lambda _: (["항목", "결과"], rows, summary)


def load_bench(_):
    d = jload("bench_results/benchmark.json")
    if not d:
        return None
    rows = [[r["model"], r["imgsz"], r["map50"], r["map50_95"], r["latency_ms"], r["fps"],
             r["weight_MB"], r["train_min"]] for r in d]
    return (["모델", "입력", "mAP50", "mAP50-95", "지연(ms)", "FPS", "크기(MB)", "학습(분)"], rows,
            "선정 = yolo26s@640 (정확도 동급 + 시드 분산 최소 + 지연 최단)")


def load_followup(_):
    d = jload("bench_results/exp_epochs.json")
    if not d:
        return None
    rows = [[r.get("name"), r.get("base"), f"{r.get('epochs_run')}/{r.get('epochs_set')}",
             r.get("map50"), r.get("map50_95"), r.get("train_min")] for r in d]
    return (["실험", "기반 모델", "epochs(실행/설정)", "mAP50", "mAP50-95", "학습(분)"], rows,
            "300ep 연장 무의미 / 26n 레시피 역효과 / 시드 3반복으로 v8s 1위는 시드 운 판명")


# ==================== 오토라벨링 방법 레지스트리 ====================
# badge: adopt(채택) / drop(탈락) / partial(부분 성공)
METHODS = [
    dict(
        id="m1", no=1, title="Pseudo-labeling", badge="adopt", badge_label="채택 (운영 루프)",
        subtitle="의사 라벨링(Pseudo-labeling)은 라벨이 없는 데이터에 모델이 예측한 값을 임시 라벨로 지정해 학습에 재활용하는 준지도학습 기법입니다.",
        ordered=True,
        bullets=[
            "**1차 모델**: 순수 YOLO 모델에 기계부품 데이터(bolt·nut·gear·bearing) 10~15%(149장)를 학습시킨다.",
            "**임시 학습 데이터 생성**: 1차 모델에게 라벨이 없는 데이터 65~70%(647장)를 예측시키고, 신뢰도(confidence)가 0.6 이상인 라벨만 채택.",
            "**2차 모델**: 순수 YOLO 모델에 1의 학습데이터와 2의 결과(0.6 이상 데이터)를 학습시킨다.",
            "**임시 학습 데이터 효과 검증**: 1차 모델과 2차 모델의 평가지표를 비교한다."],
        gallery=None, live=True,
        code=[dict(
            file="scripts/labeling/pseudo_utils.py - predict_boxes()",
            note="예측 + conf 필터. 1차 모델이 미라벨 이미지를 추론하고 신뢰도 0.6 이상만 통과시킴 (읽기 쉽게 풀어 쓴 것)",
            src="""results = model.predict(source=unlabeled_images, stream=True)
for image, result in zip(unlabeled_images, results):
    accepted = [(cls, box) for cls, box, conf in result.boxes
                if conf >= 0.6]           # 신뢰도 0.6 미만 예측은 버림
    yield image, accepted"""),
              dict(
            file="scripts/labeling/auto_labeling.py",
            note="저장. predict_boxes 가 통과시킨 박스를 YOLO 라벨(.txt)로 기록",
            src="""for image, accepted in predict_boxes(model, unlabeled_images, conf=0.6):
    # 이미지와 같은 파일명 .txt 가 YOLO 의 짝 규칙
    save_label(labels_dir / f"{image.stem}.txt", accepted)"""),
              dict(
            file="scripts/training/train_pipeline.py",
            note="재학습. 1차 모델 가중치가 아니라 순수 YOLO 에서 다시 시작",
            src="""model = YOLO("yolo26s.pt")            # 순수(COCO 사전학습) 가중치에서 시작
model.train(data=시드_라벨 + 자동_라벨,  # 사람 라벨과 자동 라벨을 합쳐 학습
            epochs=100)
# 게이트: 새 모델 mAP50 가 기존보다 낮으면 배포하지 않고 폐기"""),],
        metrics=("exp_results/report_2cls_seed15.json", load_autolearn),
    ),
    dict(
        id="m2", no=2, title="텍스트 제로샷 (Grounding DINO)", badge="drop", badge_label="탈락",
        bullets=[
            "**방식**: 학습 없이 영어 프롬프트 4종('metal hex bolt screw' 등)만으로 박스 생성",
            "**데이터**: 정답을 숨긴 평가셋 204장, 숨긴 정답과 IoU 0.5 기준 채점",
            "**탈락 사유**: 유사 금속 부품 간 클래스 혼동 (둥근 접시를 gear 로 오인)"],
        gallery="dino_text", live=False,
        metrics=("exp_results/zeroshot/zeroshot_eval_post.json", load_zeroshot),
    ),
    dict(
        id="m3", no=3, title="Grounded-SAM 타이트박스", badge="drop", badge_label="탈락",
        bullets=[
            "**방식**: 방법 2와 동일 + SAM 마스크로 박스를 픽셀 경계까지 타이트하게 교정",
            "**결과**: 정밀도 개선 없음. '박스 여백이 문제'라는 가설이 실측으로 기각됨 (맞은 박스 평균 IoU: DINO 원본 0.908 > SAM 0.876)",
            "**비고**: 증거 이미지는 서버 유실로 미보존, 지표로 확인"],
        gallery=None, live=False,
        metrics=("exp_results/zeroshot/zeroshot_eval_gsam_tight.json", load_zeroshot),
    ),
    dict(
        id="m4", no=4, title="SAM + CLIP 갤러리", badge="drop", badge_label="탈락",
        bullets=[
            "**방식**: SAM 이 물체 후보를 전부 분할 -> 각 후보를 참조 갤러리(정답에서 오린 크롭, 클래스당 10장)와 CLIP 임베딩 유사도로 분류",
            "**의의**: 텍스트 -> 시각 매칭 전환으로 정밀도 2.5배 도약 (0.24 -> 0.60)",
            "**비고**: 증거 이미지 미보존"],
        gallery=None, live=False,
        metrics=("exp_results/zeroshot/gallery_eval_clip.json", load_sweep),
    ),
    dict(
        id="m5", no=5, title="SAM + DINOv2 갤러리", badge="adopt", badge_label="고정밀 달성",
        bullets=[
            "**방식**: 방법 4와 동일하되 임베딩을 CLIP -> **DINOv2**(질감·형상 특징)로 교체",
            "**성과**: 유사도 임계값 0.85에서 정밀도 0.927 = 무검수 기준(0.87) 최초 충족",
            "**의의**: 1탭 참조 방식(방법 7)의 이론적 기반",
            "**비고**: 증거 이미지 미보존"],
        gallery=None, live=False,
        metrics=("exp_results/zeroshot/gallery_eval_dinov2.json", load_sweep),
    ),
    dict(
        id="m6", no=6, title="상호 일관성 매칭", badge="partial", badge_label="사진 성공 / 영상 실패",
        bullets=[
            "**방식**: 등록 폴더(부품 1종)의 SAM 후보 중 '다른 모든 사진에도 비슷한 물체가 있는 후보'를 DINOv2 로 식별해 라벨링",
            "**적용 ①**: 사진 묶음 시뮬 (부품 크롭 2종 x 15장) -> 성공",
            "**적용 ②**: 실사 기어박스 영상 49프레임 -> **실패** (한 장면 영상은 배경도 매 프레임 등장해 매트·드릴·사람까지 오채택)",
            "**교훈**: 채택률 98%라는 수치만 보면 합격이었으나, 육안 검증이 실패를 적발"],
        gallery="mutual", live=False,
        code=[dict(
            file="scripts/labeling/register_part.py - label_one_part()",
            note="상호 일관성 점수 - 읽기 쉽게 풀어 쓴 것. 실패 원인이 이 수식 자체에 있음",
            src="""# 사진 i 의 후보 하나하나에 대해:
# "다른 모든 사진에도 이것과 닮은 물체가 있는가" 를 점수로 만든다
score = 0
for j in range(len(all_photos)):
    if j == i:
        continue
    best = max(similarity(candidate, other) for other in photo_j.candidates)
    score += best                        # 사진 j 에서 가장 닮은 후보와의 유사도
score = score / (len(all_photos) - 1)    # 전체 평균

if score >= 0.55:                        # 임계값 통과 -> 부품으로 채택
    keep(candidate)

# 함정: 배경(매트·드릴·사람)도 모든 프레임에 등장한다
# -> 배경 후보도 score 가 높게 나와 오채택 (한 장면 영상에서 실패한 이유)"""),],
        metrics=static_metrics(
            [["사진 묶음 시뮬 (볼트·너트 15장씩)", "채택률 100%, 부품만 정확히 라벨"],
             ["실사 영상 (기어박스 49프레임)", "채택률 98%였으나 배경(매트·드릴·사람) 오채택 -> 폐기"],
             ["폐기 사유", "'배경은 사진마다 바뀐다'는 가정이 한 장면 영상에서 성립하지 않음"]],
            "판정: 배경이 바뀌는 사진 묶음에서만 유효. 영상 등록에는 부적합 -> 1탭 참조(방법 7)로 대체"),
    ),
    dict(
        id="m7", no=7, title="1탭 참조 매칭", badge="adopt", badge_label="최종 채택",
        bullets=[
            "**방식**: 등록 화면에서 부품을 **한 번 클릭** -> SAM 포인트 분할로 참조 크롭 확보 -> 모든 프레임의 SAM 후보를 참조와 DINOv2 유사도(임계값 0.7)로 매칭",
            "**데이터**: 실사 기어박스 영상 (1차 33프레임 -> 완결 193프레임)",
            "**검증**: 생성 라벨로 학습 후, 학습에 안 쓴 별도 영상(16프레임)에서 탐지 확인"],
        gallery="one_tap", live=False,
        code=[dict(
            file="scripts/labeling/register_part.py - build_ref_embedding()",
            note="1단계: 탭 한 번 -> 부품의 '기준 사진' 확보",
            src="""point = read("ref.txt")            # 사용자가 등록 화면에서 탭한 좌표 1개
mask = sam.predict(point)           # SAM: 그 점이 속한 물체 영역을 돌려줌
reference = image[mask]             # 그 영역을 오려냄 = 부품의 기준 사진
save("_preview/ref_check.jpg")      # 탭이 엉뚱한 물체를 잡았는지 육안 확인용"""),
              dict(
            file="scripts/labeling/register_part.py - label_one_part()",
            note="2단계: 모든 프레임의 후보를 기준 사진과 생김새 비교",
            src="""for frame in all_frames:
    for candidate in sam_candidates(frame):      # SAM 이 찾은 물체 후보들
        sim = similarity(candidate, reference)    # 기준 사진과 얼마나 닮았나 (DINOv2)
        if sim >= 0.70:                           # 0.70 이상은 전부 채택
            keep(candidate)                       #   (한 프레임에 여러 개 가능)
    remove_duplicates()                           # 겹침(NMS) + 부분-전체 중복 제거

# 방법 6과의 차이: 비교 대상이 '다른 사진들'이 아니라 '사용자가 찍어준 기준'
# -> 배경은 기준과 안 닮았으므로 오채택이 원천 차단됨 (오채택 0 실측)"""),],
        metrics=static_metrics(
            [["라벨 생성 (1차, 33프레임)", "20장 채택, 배경 오채택 0"],
             ["라벨 생성 (완결, 193프레임)", "114장 채택 (59%)"],
             ["학습 후 검증 (라벨 20장 학습)", "미학습 영상 탐지 5/16장 (31%, conf 0.4)"],
             ["학습 후 검증 (라벨 114장 학습)", "미학습 영상 탐지 14/16장 (88%, conf 0.4), 오탐 0"],
             ["결론", "라벨 수량이 성능 직접 좌우 (20장=31% vs 114장=88%)"]],
            "판정: 최종 채택. 사람 개입은 탭 1회, 영상만 충분히 길면 성능 확보"),
    ),
]

GLOSSARY = [
    ["라벨 / 바운딩박스", "이미지 속 부품 위치를 표시한 네모 상자. AI 학습의 '정답지'"],
    ["정밀도", "만든 라벨 중 맞은 비율 (오답 라벨이 적을수록 높음)"],
    ["재현율", "실제 부품 중 찾아낸 비율 (놓친 것이 적을수록 높음)"],
    ["mAP50", "탐지 성능 종합 점수 (0~1, 높을수록 좋음)"],
    ["conf (신뢰도)", "모델이 스스로 확신하는 정도. 0.6 이상만 라벨로 채택"],
    ["평가셋", "학습에 쓰지 않고 채점에만 쓰는 별도 문제지"],
]

EXPERIMENTS = {
    "모델 벤치마크": {
        "7모델 × 2크기 = 14조합": ("", load_bench),
        "후속 검증 (300ep·레시피·시드)": ("", load_followup),
    },
    "오토러닝 실증 (조건별)": {
        "2클래스 · 초기 라벨 10%": ("exp_results/report_2cls_seed10.json", load_autolearn),
        "2클래스 · 초기 라벨 15%": ("exp_results/report_2cls_seed15.json", load_autolearn),
        "2클래스 · TTA 필터": ("exp_results/report_2cls_tta.json", load_autolearn),
        "3클래스": ("exp_results/report_3cls.json", load_autolearn),
        "4클래스": ("exp_results/report_4cls.json", load_autolearn),
    },
}


def method_by_id(mid):
    for m in METHODS:
        if m["id"] == mid:
            return m
    return None


def method_metrics(m):
    rel_loader = m["metrics"]
    res = rel_loader(None) if callable(rel_loader) else rel_loader[1](rel_loader[0])
    if res is None:
        return {"headers": ["항목"], "rows": [], "summary": "결과 파일 없음"}
    headers, rows, summary = res
    return {"headers": headers, "rows": rows, "summary": summary}


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


# 클래스 이름 기준 고정 색(BGR). 예측·정답에서 같은 부품은 항상 같은 색으로 표기.
CLASS_COLORS = {
    "bearing": (0, 190, 0),     # 초록
    "bolt": (0, 150, 255),      # 주황
    "gear": (255, 130, 0),      # 파랑
    "nut": (200, 0, 200),       # 자홍
}


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
        return {"error": "테스트 이미지 없음 (data/robo_yolo/test)"}
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
        return {"headers": [], "rows": [], "summary": "선택하세요."}
    rel, loader = entry
    res = loader(rel)
    if res is None:
        return {"headers": [], "rows": [], "summary": f"결과 파일 없음: {rel}"}
    headers, rows, summary = res
    return {"headers": headers, "rows": rows, "summary": summary}


def export_report():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "report_builder", Path(__file__).resolve().parent / "build_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.build()
    return str(mod.OUT)
