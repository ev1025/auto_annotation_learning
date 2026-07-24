"""8_dashboard.py - 오토라벨링 방법 비교 대시보드 (Gradio).

메인 화면(①): 오토라벨링 시도 방법을 고르면
  1. 어떤 데이터를 어떻게 사용했는지 (설정 설명)
  2. 모델이 만든 바운딩박스 vs 정답 라벨 비교 이미지 (증거 갤러리 / 즉석 비교)
  3. 실험 지표 결과 (표 + 판정)
를 한 화면에 보여준다. 비교 이미지 원본은 docs/method_previews/, 지표 원본은
리포에 커밋된 결과 json (exp_results/, zeroshot_labeler/eval_out/).

실행: python scripts/8_dashboard.py  ->  http://127.0.0.1:7862 (로그인 없음, 로컬 전용 바인딩)
"""
import json
from pathlib import Path

import cv2
import gradio as gr
import yaml

import config

PORT = 7862
PREV_DIR = config.BASE_DIR / "docs" / "method_previews"
TEST_IMG = config.BASE_DIR / "mechanical-parts-yolo" / "test" / "images"
TEST_LBL = config.BASE_DIR / "mechanical-parts-yolo" / "test" / "labels"
GT_CLASSES = ["bearing", "bolt", "gear", "nut"]
PALETTE = [(0, 255, 0), (255, 160, 0), (0, 160, 255), (255, 0, 200), (160, 255, 0),
           (0, 255, 255), (255, 80, 80), (180, 120, 255)]


def class_names():
    if not config.DATA_YAML.exists():
        return {}
    raw = yaml.safe_load(config.DATA_YAML.read_text(encoding="utf-8")).get("names", {})
    if isinstance(raw, list):
        return {i: n for i, n in enumerate(raw)}
    return {int(k): v for k, v in raw.items()}


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
            "4개 조건(2·3·4클래스) 전부 상승 = 오토러닝 효과 입증 (다른 조건은 ②탭)")


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


# ==================== ① 오토라벨링 방법 레지스트리 ====================
METHODS = {
    "1. self-training 오토라벨 (운영 루프, 채택)": dict(
        data="""- **방식**: 학습된 모델이 미라벨 이미지에 예측을 만들고, 신뢰도(conf) **0.6 이상만** 라벨로 채택해 학습 데이터에 누적
- **데이터**: 공개 기계부품 데이터. **초기 라벨셋 10~15%만** 학습에 사용, 나머지 미라벨 풀(647~1,463장)은 라벨을 숨겨 자동 라벨 대상으로 사용
- **채점**: 별도 평가셋에서 라운드0(초기 라벨만) vs 라운드1(자동 라벨 추가) 성능 비교""",
        gallery=None,
        live=True,
        metrics=("exp_results/report_2cls_seed15.json", load_autolearn),
    ),
    "2. 텍스트 제로샷 (Grounding DINO, 탈락)": dict(
        data="""- **방식**: 학습 없이 영어 프롬프트 4종('metal hex bolt screw' 등)만으로 박스 생성
- **데이터**: 정답을 숨긴 평가셋 204장
- **채점**: 숨긴 정답과 IoU 0.5 기준 대조
- **탈락 사유**: 유사 금속 부품 간 클래스 혼동 (둥근 접시를 gear 로 오인)""",
        gallery="dino_text",
        live=False,
        metrics=("zeroshot_labeler/eval_out/zeroshot_eval_post.json", load_zeroshot),
    ),
    "3. Grounded-SAM 타이트박스 (탈락)": dict(
        data="""- **방식**: 2번과 동일 + SAM 마스크로 박스를 픽셀 경계까지 타이트하게 교정
- **데이터·채점**: 2번과 동일 (평가셋 204장, IoU 0.5)
- **결과**: 정밀도 개선 없음. '박스 여백이 문제'라는 가설이 실측으로 기각됨 (맞은 박스 평균 IoU: DINO 원본 0.908 > SAM 0.876)
- **비고**: 비교 이미지는 서버 유실로 미보존, 지표로 확인""",
        gallery=None,
        live=False,
        metrics=("zeroshot_labeler/eval_out/zeroshot_eval_gsam_tight.json", load_zeroshot),
    ),
    "4. SAM + CLIP 갤러리 (탈락)": dict(
        data="""- **방식**: SAM 이 물체 후보를 전부 분할 -> 각 후보를 참조 갤러리와 CLIP 임베딩 유사도로 분류
- **참조 갤러리**: train 정답에서 오린 크롭 클래스당 10장 (라벨링 대상에는 정답 미사용)
- **데이터·채점**: 평가셋 204장, IoU 0.5
- **비고**: 비교 이미지 미보존""",
        gallery=None,
        live=False,
        metrics=("zeroshot_labeler/eval_out/gallery_eval_clip.json", load_sweep),
    ),
    "5. SAM + DINOv2 갤러리 (고정밀 달성)": dict(
        data="""- **방식**: 4번과 동일하되 임베딩을 CLIP -> **DINOv2**(질감·형상 특징)로 교체
- **성과**: 유사도 임계값 0.85에서 정밀도 0.927 = 무검수 기준(0.87) 최초 충족
- **의의**: 1탭 참조 방식(7번)의 이론적 기반
- **비고**: 비교 이미지 미보존""",
        gallery=None,
        live=False,
        metrics=("zeroshot_labeler/eval_out/gallery_eval_dinov2.json", load_sweep),
    ),
    "6. 상호 일관성 매칭 (사진 성공 / 영상 실패)": dict(
        data="""- **방식**: 등록 폴더(부품 1종)의 SAM 후보 중 '다른 모든 사진에도 비슷한 물체가 있는 후보'를 DINOv2 로 식별해 라벨링
- **적용 ①**: 사진 묶음 시뮬 (부품 크롭 2종 x 15장) -> 성공
- **적용 ②**: 실사 기어박스 영상 49프레임 -> **실패** (한 장면 영상은 배경도 매 프레임 등장해 매트·드릴·사람까지 오채택)
- **교훈**: 채택률 98%라는 수치만 보면 합격이었으나, 육안 검증이 실패를 적발""",
        gallery="mutual",
        live=False,
        metrics=static_metrics(
            [["사진 묶음 시뮬 (볼트·너트 15장씩)", "채택률 100%, 부품만 정확히 라벨"],
             ["실사 영상 (기어박스 49프레임)", "채택률 98%였으나 배경(매트·드릴·사람) 오채택 -> 폐기"],
             ["폐기 사유", "'배경은 사진마다 바뀐다'는 가정이 한 장면 영상에서 성립하지 않음"]],
            "판정: 배경이 바뀌는 사진 묶음에서만 유효. 영상 등록에는 부적합 -> 1탭 참조(7번)로 대체"),
    ),
    "7. 1탭 참조 매칭 (최종 채택)": dict(
        data="""- **방식**: 등록 화면에서 부품을 **한 번 클릭** -> SAM 포인트 분할로 참조 크롭 확보 -> 모든 프레임의 SAM 후보를 참조와 DINOv2 유사도(임계값 0.7)로 매칭
- **데이터**: 기어박스 영상2 (1차 33프레임 -> 완결 193프레임)
- **검증**: 생성 라벨로 학습 후, 학습에 안 쓴 영상1(16프레임)에서 탐지 확인""",
        gallery="one_tap",
        live=False,
        metrics=static_metrics(
            [["라벨 생성 (1차, 33프레임)", "20장 채택, 배경 오채택 0"],
             ["라벨 생성 (완결, 193프레임)", "114장 채택 (59%)"],
             ["학습 후 검증 (라벨 20장 학습)", "미학습 영상 탐지 5/16장 (31%, conf 0.4)"],
             ["학습 후 검증 (라벨 114장 학습)", "미학습 영상 탐지 14/16장 (88%, conf 0.4), 오탐 0"],
             ["결론", "라벨 수량이 성능 직접 좌우 (20장=31% vs 114장=88%)"]],
            "판정: 최종 채택. 사람 개입은 탭 1회, 영상만 충분히 길면 성능 확보"),
    ),
}


def method_gallery(key):
    info = METHODS[key]
    sub = info.get("gallery")
    if not sub:
        return []
    d = PREV_DIR / sub
    return [(str(p), p.stem.split("_", 1)[1].replace("_", " ")) for p in sorted(d.glob("*.jpg"))]


def show_method(key):
    info = METHODS[key]
    rel_loader = info["metrics"]
    if callable(rel_loader):
        res = rel_loader(None)
    else:
        rel, loader = rel_loader
        res = loader(rel)
    if res is None:
        headers, rows, summary = ["항목"], [], "결과 파일 없음"
    else:
        headers, rows, summary = res
    gal = method_gallery(key)
    gal_note = ""
    return (info["data"] + gal_note, gr.update(value=gal, visible=bool(gal)),
            gr.update(value=rows, headers=headers), summary,
            gr.update(visible=info.get("live", False)))


# ---- self-training 즉석 비교 (모델 박스 | 정답 박스) ----
def get_model():
    from ultralytics import YOLO
    if not config.SERVE_MODEL.exists():
        return None
    if not hasattr(get_model, "m"):
        get_model.m = YOLO(str(config.SERVE_MODEL))
    return get_model.m


def draw_gt(im, lbl_path, w, h):
    if lbl_path.exists():
        for i, line in enumerate(lbl_path.read_text().splitlines(), start=1):
            f = line.split()
            if len(f) < 5:
                continue
            c, cx, cy, bw, bh = int(f[0]), *map(float, f[1:5])
            x1, y1 = round((cx - bw / 2) * w), round((cy - bh / 2) * h)
            x2, y2 = round((cx + bw / 2) * w), round((cy + bh / 2) * h)
            color = PALETTE[c % len(PALETTE)]
            cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
            cv2.putText(im, f"#{i} {GT_CLASSES[c] if c < len(GT_CLASSES) else c}",
                        (x1, max(y1 - 6, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return im


def live_compare(conf=0.6):
    import random as _r
    m = get_model()
    if m is None:
        return None, None, "서빙 모델 없음 (models/new_model.pt)"
    imgs = sorted(TEST_IMG.glob("*.jpg"))
    if not imgs:
        return None, None, "테스트 이미지 없음 (mechanical-parts-yolo/test)"
    p = _r.choice(imgs)
    src = cv2.imread(str(p))
    h, w = src.shape[:2]
    # 왼쪽: 모델 예측
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
    # 오른쪽: 정답 라벨
    gt = draw_gt(src.copy(), TEST_LBL / f"{p.stem}.txt", w, h)
    note = (f"{p.name} / 모델 탐지 {n}개 (conf {conf}) - 현재 모델은 임시 2클래스(bolt·nut)라 "
            "bearing·gear 는 원래 못 잡습니다. 서버 복구 후 5클래스 모델로 교체 예정")
    return cv2.cvtColor(pred, cv2.COLOR_BGR2RGB), cv2.cvtColor(gt, cv2.COLOR_BGR2RGB), note


# ==================== ② 기타 실험 결과 ====================
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


def show_experiment(cat, topic):
    entry = EXPERIMENTS.get(cat, {}).get(topic)
    if not entry:
        return gr.update(value=[]), "선택하세요."
    rel, loader = entry
    res = loader(rel)
    if res is None:
        return gr.update(value=[]), f"결과 파일 없음: {rel}"
    headers, rows, summary = res
    return gr.update(value=rows, headers=headers), summary


def on_category(cat):
    ts = list(EXPERIMENTS.get(cat, {}).keys())
    first = ts[0] if ts else None
    table, summary = show_experiment(cat, first) if first else (gr.update(value=[]), "")
    return gr.update(choices=ts, value=first), table, summary


# ==================== UI ====================
FIRST = list(METHODS.keys())[0]

THEME = gr.themes.Soft(
    primary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Noto Sans KR"), "system-ui", "sans-serif"],
).set(body_text_size="16px")

CSS = """
.gradio-container {max-width: 1320px !important; margin: 0 auto !important;}
.gradio-container table {font-variant-numeric: tabular-nums;}      /* 표 숫자 자릿수 정렬 */
.gradio-container thead th {font-weight: 600;}
#page-title h2 {font-weight: 700; letter-spacing: -0.01em; margin-bottom: 0;}
#page-sub {color: var(--body-text-color-subdued); margin-top: 2px;}
.section-h h3 {font-weight: 600; margin: 0 0 2px 0;}
.section-h {border-top: 1px solid var(--border-color-primary); padding-top: 18px; margin-top: 14px;}
#intro-acc {margin-top: 4px;}
#live-btn {max-width: 200px; align-self: center;}
#judge textarea, #judge2 textarea {
  border-left: 4px solid var(--primary-500);
  font-weight: 600;
  background: var(--background-fill-secondary);
}
"""

with gr.Blocks(title="오토라벨링 방법 비교", theme=THEME, css=CSS) as app:
    gr.Markdown("## 오토라벨링 방법 비교", elem_id="page-title")
    gr.Markdown("헬기 정비 부품을 자동 인식하는 AI 를 만들기 위해, 학습용 정답 박스(라벨)를 사람 대신 "
                "**자동으로 만드는 방법 7가지**를 실험했습니다. 방법을 선택하면 ①어떤 데이터로 "
                "②라벨이 실제로 어떻게 생겼고 ③성적이 어땠는지를 보여줍니다.", elem_id="page-sub")
    with gr.Accordion("이 화면이 처음이라면 - 용어 안내", open=False, elem_id="intro-acc"):
        gr.Markdown("""| 용어 | 뜻 |
|---|---|
| 라벨 / 바운딩박스 | 이미지 속 부품 위치를 표시한 네모 상자. AI 학습의 '정답지' |
| 정밀도 (P) | 만든 라벨 중 맞은 비율. 오답 라벨이 적을수록 높음 |
| 재현율 (R) | 실제 부품 중 찾아낸 비율. 놓친 것이 적을수록 높음 |
| mAP50 | 탐지 성능 종합 점수 (0~1, 높을수록 좋음) |
| conf (신뢰도) | 모델이 스스로 확신하는 정도. 0.6 이상만 라벨로 채택 |
| 평가셋 | 학습에 쓰지 않고 채점에만 쓰는 별도 문제지 |""")

    with gr.Tab("① 오토라벨링 방법 비교"):
        method = gr.Dropdown(list(METHODS.keys()), value=FIRST,
                             label="시도 방법 (번호 = 시도 순서)")
        gr.Markdown("### 1. 어떤 데이터를 어떻게 사용했나", elem_classes="section-h")
        desc_md = gr.Markdown()
        gr.Markdown("### 2. 모델이 만든 바운딩박스 vs 정답 라벨", elem_classes="section-h")
        gal = gr.Gallery(label="비교 이미지 (초록/색 박스 = 모델·자동 라벨, 빨강 = 정답)", columns=2, height=400)
        with gr.Column(visible=False) as live_grp:
            with gr.Row(equal_height=True):
                gr.Markdown("평가셋에서 무작위 이미지를 뽑아 나란히 비교합니다. 신뢰도 기준 = 오토라벨 채택 조건(conf 0.6) 고정.",
                            scale=4)
                live_btn = gr.Button("다른 이미지 보기", variant="primary", size="sm",
                                     elem_id="live-btn", scale=1)
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**모델이 만든 바운딩박스**")
                    live_pred = gr.Image(show_label=False, buttons=[], height=420)
                with gr.Column():
                    gr.Markdown("**정답 라벨**")
                    live_gt = gr.Image(show_label=False, buttons=[], height=420)
            live_note = gr.Textbox(label="비고", interactive=False)
        gr.Markdown("### 3. 실험 지표 결과", elem_classes="section-h")
        met_summary = gr.Textbox(label="판정·요약", lines=2, interactive=False, elem_id="judge")
        met_table = gr.Dataframe(label="지표", interactive=False)

        method.change(show_method, method, [desc_md, gal, met_table, met_summary, live_grp])
        live_btn.click(live_compare, None, [live_pred, live_gt, live_note])
        app.load(show_method, method, [desc_md, gal, met_table, met_summary, live_grp])
        app.load(live_compare, None, [live_pred, live_gt, live_note])

    with gr.Tab("② 기타 실험 결과"):
        with gr.Row():
            cat = gr.Dropdown(list(EXPERIMENTS.keys()), value=list(EXPERIMENTS.keys())[0],
                              label="카테고리", scale=2)
            topic = gr.Dropdown(list(EXPERIMENTS[list(EXPERIMENTS.keys())[0]].keys()), label="주제", scale=2)
        exp_summary = gr.Textbox(label="요약·판정", lines=2, interactive=False, elem_id="judge2")
        exp_table = gr.Dataframe(label="결과", interactive=False)
        cat.change(on_category, cat, [topic, exp_table, exp_summary])
        topic.change(show_experiment, [cat, topic], [exp_table, exp_summary])
        app.load(on_category, cat, [topic, exp_table, exp_summary])

if __name__ == "__main__":
    # 로그인 제거에 맞춰 로컬 전용 바인딩(127.0.0.1). 외부 공개 시에는 auth 를 다시 걸 것.
    app.launch(server_name="127.0.0.1", server_port=PORT)
