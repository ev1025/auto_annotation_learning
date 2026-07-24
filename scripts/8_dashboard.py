"""8_dashboard.py - 오토라벨링 방법 비교 대시보드 (Gradio).

메인 화면(①): 오토라벨링 시도 방법을 고르면
  1. 어떤 데이터를 어떻게 사용했는지 (설정 설명)
  2. 모델이 만든 바운딩박스 vs 정답 라벨 비교 이미지 (증거 갤러리 / 즉석 비교)
  3. 실험 지표 결과 (표 + 판정)
를 한 화면에 보여준다. 비교 이미지 원본은 docs/method_previews/, 지표 원본은
리포에 커밋된 결과 json (exp_results/, zeroshot_labeler/eval_out/).

실행: python scripts/8_dashboard.py  ->  http://127.0.0.1:7862 (로그인 AUTH)
"""
import json
from pathlib import Path

import cv2
import gradio as gr
import yaml

import config

PORT = 7862
AUTH = ("suri", "suri")
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
        data="**어떤 데이터를 어떻게**: 학습된 모델이 미라벨 이미지에 예측을 만들고, 신뢰도(conf) 0.6 이상만 라벨로 채택해 학습 데이터에 누적한다. "
             "실증 설정: 공개 기계부품 데이터에서 **초기 라벨셋 10~15%만** 학습에 쓰고, 나머지 미라벨 풀(647~1,463장)은 라벨을 숨겨 자동 라벨 대상으로 사용. 별도 평가셋으로 채점.",
        gallery=None,
        live=True,
        metrics=("exp_results/report_2cls_seed15.json", load_autolearn),
    ),
    "2. 텍스트 제로샷 (Grounding DINO, 탈락)": dict(
        data="**어떤 데이터를 어떻게**: 학습 없이, 정답을 숨긴 test 204장에 영어 프롬프트 4종('metal hex bolt screw' 등)만 주고 박스를 생성 -> 숨긴 정답과 IoU 0.5로 채점. "
             "탈락 사유: 유사 금속 부품 클래스 혼동(둥근 접시를 gear 로 오인).",
        gallery="dino_text",
        live=False,
        metrics=("zeroshot_labeler/eval_out/zeroshot_eval_post.json", load_zeroshot),
    ),
    "3. Grounded-SAM 타이트박스 (탈락)": dict(
        data="**어떤 데이터를 어떻게**: 2번과 동일 데이터·프롬프트에 SAM 마스크로 박스를 타이트하게 교정. "
             "결과: 정밀도 개선 없음 -> '박스 여백이 문제'라는 가설이 실측으로 기각됨 (맞은 박스 평균 IoU 는 DINO 원본 0.908 > SAM 0.876). "
             "비교 이미지는 서버 유실로 미보존 (지표만 확인 가능).",
        gallery=None,
        live=False,
        metrics=("zeroshot_labeler/eval_out/zeroshot_eval_gsam_tight.json", load_zeroshot),
    ),
    "4. SAM + CLIP 갤러리 (탈락)": dict(
        data="**어떤 데이터를 어떻게**: SAM 이 test 204장의 물체 후보를 전부 분할하고, 각 후보를 **참조 갤러리**(train 정답에서 오린 크롭, 클래스당 10장)와 CLIP 임베딩 유사도로 분류. "
             "정답 라벨은 갤러리 구성에만 사용(라벨링 대상에는 미사용). 비교 이미지 미보존.",
        gallery=None,
        live=False,
        metrics=("zeroshot_labeler/eval_out/gallery_eval_clip.json", load_sweep),
    ),
    "5. SAM + DINOv2 갤러리 (고정밀 달성)": dict(
        data="**어떤 데이터를 어떻게**: 4번과 동일하되 임베딩을 CLIP -> DINOv2(질감·형상 특징)로 교체. "
             "유사도 임계값 0.85에서 정밀도 0.927 = 무검수 기준 최초 충족. 1탭 참조 방식(7번)의 이론적 기반. 비교 이미지 미보존.",
        gallery=None,
        live=False,
        metrics=("zeroshot_labeler/eval_out/gallery_eval_dinov2.json", load_sweep),
    ),
    "6. 상호 일관성 매칭 (사진 성공 / 영상 실패)": dict(
        data="**어떤 데이터를 어떻게**: 등록 폴더(부품 1종) 사진들에서 SAM 후보를 뽑고, '다른 모든 사진에도 비슷한 물체가 있는 후보'를 DINOv2 로 식별해 라벨링. "
             "적용 데이터: ① 사진 묶음 시뮬(부품 크롭 2종 x 15장) -> 성공 ② 실사 기어박스 영상 49프레임 -> **실패** (한 장면 영상은 배경도 매 프레임 등장해 매트·드릴·사람까지 오채택). "
             "교훈: 채택률 98%라는 수치만 보면 합격이었으나 육안 검증에서 실패 적발.",
        gallery="mutual",
        live=False,
        metrics=static_metrics(
            [["사진 묶음 시뮬 (볼트·너트 15장씩)", "채택률 100%, 부품만 정확히 라벨"],
             ["실사 영상 (기어박스 49프레임)", "채택률 98%였으나 배경(매트·드릴·사람) 오채택 -> 폐기"],
             ["폐기 사유", "'배경은 사진마다 바뀐다'는 가정이 한 장면 영상에서 성립하지 않음"]],
            "판정: 배경이 바뀌는 사진 묶음에서만 유효. 영상 등록에는 부적합 -> 1탭 참조(7번)로 대체"),
    ),
    "7. 1탭 참조 매칭 (최종 채택)": dict(
        data="**어떤 데이터를 어떻게**: 사용자가 등록 화면에서 부품을 **한 번 클릭** -> SAM 포인트 분할로 참조 크롭 확보 -> 모든 프레임의 SAM 후보를 참조와 DINOv2 유사도(임계값 0.7)로 매칭해 라벨링. "
             "적용 데이터: 기어박스 영상2 (1차 33프레임 -> 완결 193프레임). 생성된 라벨로 학습 후, 학습에 안 쓴 영상1(16프레임)로 탐지 검증.",
        gallery="one_tap",
        live=False,
        metrics=static_metrics(
            [["라벨 생성 (1차, 33프레임)", "20장 채택, 배경 오채택 0"],
             ["라벨 생성 (완결, 193프레임)", "114장 채택 (59%)"],
             ["학습 후 검증 (미학습 영상1, 라벨 20장 학습)", "탐지 5/16장 (31%, conf 0.4)"],
             ["학습 후 검증 (미학습 영상1, 라벨 114장 학습)", "탐지 14/16장 (88%, conf 0.4) / 15/16장 (94%, conf 0.25), 오탐 0"],
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
    gal_note = "" if gal else " (이 방법은 저장된 비교 이미지가 없습니다. 지표로 확인하세요)"
    return (info["data"] + gal_note, gal,
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


def live_compare(conf):
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

with gr.Blocks(title="오토라벨링 방법 비교") as app:
    gr.Markdown("## 오토라벨링 방법 비교 대시보드")

    with gr.Tab("① 오토라벨링 방법 비교"):
        method = gr.Dropdown(list(METHODS.keys()), value=FIRST,
                             label="시도 방법 (번호 = 시도 순서)")
        gr.Markdown("### 1. 어떤 데이터를 어떻게 사용했나")
        desc_md = gr.Markdown()
        gr.Markdown("### 2. 모델이 만든 바운딩박스 vs 정답 라벨")
        gal = gr.Gallery(label="비교 이미지 (초록/색 박스 = 모델·자동 라벨, 빨강 = 정답)", columns=2, height=480)
        with gr.Group(visible=False) as live_grp:
            gr.Markdown("**즉석 비교**: 평가셋에서 무작위 이미지를 뽑아 왼쪽 = 모델 박스, 오른쪽 = 정답 라벨")
            with gr.Row():
                live_conf = gr.Slider(0.1, 0.9, value=0.6, step=0.05, label="conf (오토라벨 채택 기준 = 0.6)")
                live_btn = gr.Button("무작위 이미지 비교", variant="primary")
            with gr.Row():
                live_pred = gr.Image(label="모델이 만든 바운딩박스")
                live_gt = gr.Image(label="정답 라벨")
            live_note = gr.Textbox(label="비고", interactive=False)
        gr.Markdown("### 3. 실험 지표 결과")
        met_summary = gr.Textbox(label="판정·요약", lines=2, interactive=False)
        met_table = gr.Dataframe(label="지표", interactive=False)

        method.change(show_method, method, [desc_md, gal, met_table, met_summary, live_grp])
        live_btn.click(live_compare, live_conf, [live_pred, live_gt, live_note])
        app.load(show_method, method, [desc_md, gal, met_table, met_summary, live_grp])

    with gr.Tab("② 기타 실험 결과"):
        with gr.Row():
            cat = gr.Dropdown(list(EXPERIMENTS.keys()), value=list(EXPERIMENTS.keys())[0],
                              label="카테고리", scale=2)
            topic = gr.Dropdown(list(EXPERIMENTS[list(EXPERIMENTS.keys())[0]].keys()), label="주제", scale=2)
        exp_summary = gr.Textbox(label="요약·판정", lines=2, interactive=False)
        exp_table = gr.Dataframe(label="결과", interactive=False)
        cat.change(on_category, cat, [topic, exp_table, exp_summary])
        topic.change(show_experiment, [cat, topic], [exp_table, exp_summary])
        app.load(on_category, cat, [topic, exp_table, exp_summary])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=PORT, auth=AUTH)
