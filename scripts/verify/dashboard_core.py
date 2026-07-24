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
        ["라운드0 mAP50 (초기 라벨만)", r0.get("map50")],
        ["라운드1 mAP50 (자동 라벨 추가)", r1.get("map50")],
        ["효과", f"mAP50 +{round(d.get('delta_map50', 0) * 100, 1)}%p / mAP50-95 +{round(d.get('delta_map50_95', 0) * 100, 1)}%p"],
    ]
    return (["항목", "값"], rows,
            f"자동 라벨로 재학습하자 mAP50 {r0.get('map50')} -> {r1.get('map50')}. "
            "4개 조건(2·3·4클래스) 전부 상승 = 오토러닝 효과 입증 (다른 조건은 '기타 실험')")


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
        id="m1", no=1, title="self-training 오토라벨", badge="adopt", badge_label="채택 (운영 루프)",
        bullets=[
            "**1차 모델 만들기**: 순수 YOLO(COCO 사전학습 yolo26s)에 Roboflow 기계부품 데이터(bolt·nut·gear·bearing)의 **10~15%(시드)만** 학습시킴",
            "**자동 라벨 생성**: 이 1차 모델이 **미라벨 풀(65~70%, 647~1,463장)**에 예측을 만들고, **신뢰도(conf) 0.6 이상 박스만** 라벨로 채택 (전체의 20%는 채점용 평가셋으로 따로 떼둠)",
            "**재학습**: 순수 YOLO 에 [시드 + 자동 라벨]을 합쳐 **처음부터 다시** 학습 (1차 모델에 이어붙이지 않음 - 성능 변화가 오직 '자동 라벨 추가분' 때문이도록)",
            "**채점**: 어느 학습에도 안 쓴 별도 평가셋에서 1차 모델(라운드0) vs 재학습 모델(라운드1) 점수 비교"],
        gallery=None, live=True,
        code=[dict(
            file="scripts/labeling/pseudo_utils.py · auto_labeling.py",
            note="자동 라벨 생성 - 핵심 로직을 읽기 쉽게 풀어 쓴 것 (원문은 파일 참조)",
            src="""# 시드로 학습한 1차 모델이 미라벨 이미지 전체를 추론
results = model.predict(source=unlabeled_images, stream=True)

for image, result in zip(unlabeled_images, results):
    accepted = []
    for (cls, box, confidence) in result.boxes:
        if confidence >= 0.6:                  # 신뢰도 0.6 미만 예측은 버림
            accepted.append((cls, box))

    # 채택 박스를 YOLO 라벨(.txt)로 저장 - 이미지와 같은 이름이 짝 규칙
    save_label(labels_dir / f"{image.stem}.txt", accepted)"""),
              dict(
            file="scripts/training/train_pipeline.py",
            note="재학습 - 1차 모델 가중치가 아니라 순수 YOLO 에서 다시 시작",
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
    pred = src.copy()
    r = m.predict(source=str(p), conf=conf, verbose=False)[0]
    n = 0
    for i, (b, c, cf) in enumerate(zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf), start=1):
        x1, y1, x2, y2 = map(int, b)
        color = PALETTE[int(c) % len(PALETTE)]
        cv2.rectangle(pred, (x1, y1), (x2, y2), color, 2)
        cv2.putText(pred, f"#{i} {m.names[int(c)]} {float(cf):.2f}", (x1, max(y1 - 6, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        n += 1
    gt = src.copy()
    lbl = TEST_LBL / f"{p.stem}.txt"
    if lbl.exists():
        for i, line in enumerate(lbl.read_text().splitlines(), start=1):
            f = line.split()
            if len(f) < 5:
                continue
            c, cx, cy, bw, bh = int(f[0]), *map(float, f[1:5])
            x1, y1 = round((cx - bw / 2) * w), round((cy - bh / 2) * h)
            x2, y2 = round((cx + bw / 2) * w), round((cy + bh / 2) * h)
            color = PALETTE[c % len(PALETTE)]
            cv2.rectangle(gt, (x1, y1), (x2, y2), color, 2)
            cv2.putText(gt, f"#{i} {GT_CLASSES[c] if c < len(GT_CLASSES) else c}",
                        (x1, max(y1 - 6, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    note = (f"{p.name} / 모델 탐지 {n}개 (conf {conf}) - 현재 모델은 임시 2클래스(bolt·nut)라 "
            "bearing·gear 는 원래 못 잡습니다. 서버 복구 후 5클래스 모델로 교체 예정")
    return {"pred": _b64(pred), "gt": _b64(gt), "note": note, "idx": idx, "total": len(imgs)}


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
