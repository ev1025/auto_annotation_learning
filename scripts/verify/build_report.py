# -*- coding: utf-8 -*-
"""scripts/verify/build_report.py - 오토라벨링 검증 리포트(단일 HTML) 생성.

기획(승인본): 조작형 대시보드 대신 "위에서 아래로 읽는 스토리형 리포트".
  구성 = 프로젝트 배경 -> 결과 한눈에(정밀도 막대) -> 방법 1~7 (①데이터 ②증거
  이미지 ③지표·판정, 동일 틀 반복) -> 부록(모델 벤치마크).
원본 = 리포에 커밋된 결과 json + docs/method_previews/ 증거 이미지.
방법 1의 '모델 박스 vs 정답' 비교쌍은 빌드 시 로컬 모델로 자동 생성.

실행:  ./venv/Scripts/python.exe scripts/verify/build_report.py
출력:  docs/report.html  (이미지 임베드 단일 파일 - 브라우저로 열면 끝)
"""
import base64
import json
import random
from pathlib import Path

import cv2

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ 공용(config 등)
import config
import dashboard_core as core
import html as _html

OUT = config.BASE_DIR / "docs" / "report.html"
PREV = config.BASE_DIR / "docs" / "method_previews"
TEST_IMG = config.DATA_DIR / "robo_yolo" / "test" / "images"
TEST_LBL = config.DATA_DIR / "robo_yolo" / "test" / "labels"
GT_CLASSES = ["bearing", "bolt", "gear", "nut"]


def jload(rel):
    p = config.BASE_DIR / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def b64_img(im, max_w=880, q=80):
    h, w = im.shape[:2]
    if w > max_w:
        im = cv2.resize(im, (max_w, int(h * max_w / w)))
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def b64_file(path, max_w=880):
    im = cv2.imread(str(path))
    return b64_img(im, max_w) if im is not None else None


# ---------------- 방법 1 비교쌍 자동 생성 (모델 박스 | 정답) ----------------
def draw_boxes(im, items):
    """items = [(class_name, label, xyxy)] - 박스 + 겹침 회피 라벨(클래스별 고정 색)."""
    dets = [(int(x1), int(y1), int(x2), int(y2), label, core.color_for(name))
            for name, label, (x1, y1, x2, y2) in items]
    core.render_detections(im, dets)
    return im


def gen_selftrain_pairs(n=4, conf=0.6, seed=7):
    """평가셋에서 탐지가 있는 이미지를 골라 (모델박스, 정답) 이미지쌍 생성."""
    if not config.SERVE_MODEL.exists():
        return [], "서빙 모델이 없어 비교쌍 생성을 건너뜀"
    try:
        from ultralytics import YOLO
    except ImportError:
        return [], "ultralytics 미설치로 비교쌍 생성을 건너뜀"
    model = YOLO(str(config.SERVE_MODEL))
    imgs = sorted(TEST_IMG.glob("*.jpg"))
    random.Random(seed).shuffle(imgs)
    pairs = []
    for p in imgs:
        r = model.predict(source=str(p), conf=conf, verbose=False)[0]
        if len(r.boxes) == 0:
            continue
        src = cv2.imread(str(p))
        h, w = src.shape[:2]
        pred = draw_boxes(src.copy(), [
            (model.names[int(c)], f"{model.names[int(c)]} {float(cf):.2f}", tuple(map(int, b)))
            for b, c, cf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf)])
        gt_items = []
        lf = TEST_LBL / f"{p.stem}.txt"
        if lf.exists():
            for i, line in enumerate(lf.read_text().splitlines(), 1):
                f = line.split()
                if len(f) < 5:
                    continue
                c, cx, cy, bw, bh = int(f[0]), *map(float, f[1:5])
                gt_items.append((c, GT_CLASSES[c],
                                 (round((cx - bw / 2) * w), round((cy - bh / 2) * h),
                                  round((cx + bw / 2) * w), round((cy + bh / 2) * h))))
        gt = draw_boxes(src.copy(), gt_items)
        pairs.append((b64_img(pred, 620), b64_img(gt, 620), p.name))
        if len(pairs) == n:
            break
    return pairs, None


# ---------------- 지표 -> HTML 표 ----------------
def table(headers, rows):
    th = "".join(f"<th>{h}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"


def _delta(a, b):
    return f"+{round((b - a) * 100, 1)}%p" if a is not None and b is not None else ""


def autolearn_rows(rel):
    d = jload(rel)
    sp, ps = d["split"], d["pseudo"]
    return [
        ["데이터 구성", f"초기 라벨셋 {sp['seed']}장 / 미라벨 풀 {sp['pool']}장 / 평가셋 {sp['test']}장"],
        ["자동 라벨 생성", f"{ps['labeled_images']}장 (박스 {ps['boxes']}개)"],
        ["자동 라벨 정밀도 / 재현율", f"{ps['precision']} / {ps['recall']}"],
    ]


def perf_rows(rel):
    d = jload(rel)
    r0, r1 = d["round0"], d["round1"]
    return [
        ["mAP50", r0["map50"], r1["map50"], _delta(r0["map50"], r1["map50"])],
        ["mAP50-95", r0["map50_95"], r1["map50_95"], _delta(r0["map50_95"], r1["map50_95"])],
    ]


def per_class_rows(rel):
    d = jload(rel)
    pc0 = d["round0"].get("per_class_map50_95", {})
    pc1 = d["round1"].get("per_class_map50_95", {})
    return [[cls, pc0.get(cls), pc1.get(cls), _delta(pc0.get(cls), pc1.get(cls))] for cls in pc1]


def zeroshot_rows(rel):
    d = jload(rel)
    rows = [[c, s["precision"], s["recall"], s["tp"], s["fp"], s["fn"]]
            for c, s in d["per_class"].items()]
    m = d["micro"]
    rows.append(["<b>전체</b>", f"<b>{m['precision']}</b>", m["recall"], m["tp"], m["fp"], m["fn"]])
    return rows, m["precision"]


def sweep_rows(rel, taus=(0.6, 0.7, 0.8, 0.85)):
    d = jload(rel)
    rows = [[s["tau"], s["precision"], s["recall"], s["f1"]]
            for s in d["sweep"] if s.get("margin", 0.0) == 0.0 and s["tau"] in taus]
    return rows, d.get("best_recall_at_p85")


# ---------------- HTML 부품 ----------------
def badge(kind):
    return {"adopt": '<span class="badge adopt">채택</span>',
            "drop": '<span class="badge drop">탈락</span>',
            "partial": '<span class="badge partial">부분 성공</span>'}[kind]


def figure(src, caption):
    return f'<figure><img src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'


def gallery_html(sub, cols=2):
    d = PREV / sub
    figs = []
    for p in sorted(d.glob("*.jpg")):
        cap = p.stem.split("_", 1)[1].replace("_", " ")
        src = b64_file(p, 620)
        if src:
            figs.append(figure(src, cap))
    return f'<div class="grid c{cols}">' + "".join(figs) + "</div>"


def callout(text):
    return f'<div class="callout">{text}</div>'


def section(anchor, title, badge_html, body):
    return f'<section id="{anchor}"><h2>{title} {badge_html}</h2>{body}</section>'


def sub(t):
    return f'<h3 class="section-h">{t}</h3>'


def verdict(text):
    return f'<p class="verdict">{text}</p>'


def code_html(mid):
    m = core.method_by_id(mid)
    blocks = []
    for sn in (m.get("code") or []):
        blocks.append(
            f'<p class="snip-note">{sn["note"]}</p>'
            f'<div class="snippet"><div class="snip-head">'
            f'<code class="snip-file">{sn["file"]}</code></div>'
            f'<pre><code>{_html.escape(sn["src"])}</code></pre></div>')
    return (sub("실제 코드") + "".join(blocks)) if blocks else ""


def ordered(items):
    return "<ol>" + "".join(f"<li>{i}</li>" for i in items) + "</ol>"


def bullets(items):
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def bar_row(name, value, note="", adopt=None):
    """정밀도 가로 막대 1행 (값 없는 행은 텍스트로)."""
    b = badge(adopt) if adopt else ""
    if value is None:
        return (f'<div class="brow"><div class="bname">{name} {b}</div>'
                f'<div class="btrack"><span class="bnote">{note}</span></div></div>')
    pct = round(value * 100)
    return (f'<div class="brow"><div class="bname">{name} {b}</div>'
            f'<div class="btrack"><div class="bfill" style="width:{pct}%"></div>'
            f'<span class="bval">{value}{note}</span></div></div>')


# ---------------- 본문 구성 ----------------
def build():
    al = jload("exp_results/report_2cls_seed15.json")
    zs_rows, zs_p = zeroshot_rows("exp_results/zeroshot/zeroshot_eval_post.json")
    gs_rows, gs_p = zeroshot_rows("exp_results/zeroshot/zeroshot_eval_gsam_tight.json")
    clip_rows, _ = sweep_rows("exp_results/zeroshot/gallery_eval_clip.json")
    dino_rows, dino_hp = sweep_rows("exp_results/zeroshot/gallery_eval_dinov2.json")
    pairs, pair_note = gen_selftrain_pairs()

    # ---- 결과 한눈에 ----
    overview = "".join([
        bar_row("1. self-training 오토라벨", al["pseudo"]["precision"], " *", "adopt"),
        bar_row("2. 텍스트 제로샷 (Grounding DINO)", zs_p, "", "drop"),
        bar_row("3. Grounded-SAM 타이트박스", gs_p, "", "drop"),
        bar_row("4. SAM + CLIP 갤러리", 0.600, " (임계값 0.85)", "drop"),
        bar_row("5. SAM + DINOv2 갤러리", 0.927, " (임계값 0.85)", "adopt"),
        bar_row("6. 상호 일관성 매칭", None, "사진 묶음 성공 / 영상 실패 (배경 오채택)", "partial"),
        bar_row("7. 1탭 참조 매칭", None, "오채택 0 · 학습 후 미학습 영상 재현율 88% **", "adopt"),
    ])
    overview += ('<p class="fn">* 1번은 자체 실증 데이터(미라벨 풀 647장) 기준, 2~5번은 동일 평가셋 204장 기준. '
                 '** 7번은 영상 등록 실험이라 라벨 정밀도 대신 오채택·최종 탐지 성능으로 평가.</p>')

    # ---- 방법 1 ----
    pair_html = ""
    if pairs:
        pair_html = "".join(
            f'<div class="grid c2"><figure><img src="{a}"><figcaption>모델이 만든 바운딩박스</figcaption></figure>'
            f'<figure><img src="{b}"><figcaption>정답 라벨 · {name}</figcaption></figure></div>'
            for a, b, name in pairs)
        pair_html += ('<p class="fn">비교쌍은 리포트 생성 시 자동 추출 (신뢰도 0.6 = 오토라벨 채택 기준). '
                      '현재 모델은 임시 2클래스(bolt·nut)로, 서버 복구 후 5클래스 모델 기준으로 재생성 예정.</p>')
    else:
        pair_html = f'<p class="fn">{pair_note}</p>'
    m1 = section("m1", "Pseudo-labeling", badge("adopt"),
        '<p class="method-desc">의사 라벨링(Pseudo-labeling)은 라벨이 없는 데이터에 모델이 예측한 값을 '
        '임시 라벨로 지정해 학습에 재활용하는 준지도학습 기법입니다.</p>' +
        sub("실험 순서") + ordered([
            "<b>1차 모델</b>: 순수 YOLO 모델에 기계부품 데이터(bolt·nut·gear·bearing) 10~15%(149장)를 학습시킨다.",
            "<b>임시 학습 데이터 생성</b>: 1차 모델에게 라벨이 없는 데이터 65~70%(647장)를 예측시키고, 신뢰도(confidence)가 0.6 이상인 라벨만 채택.",
            "<b>2차 모델</b>: 순수 YOLO 모델에 1의 학습데이터와 2의 결과(0.6 이상 데이터)를 학습시킨다.",
            "<b>임시 학습 데이터 효과 검증</b>: 1차 모델과 2차 모델의 평가지표를 비교한다."]) +
        code_html("m1") +
        sub("실제 입출력 결과") + pair_html +
        sub("실험 결과") +
        table(["항목", "값"], autolearn_rows("exp_results/report_2cls_seed15.json")) +
        '<p class="subtable-title">모델 성능 (1차 → 2차)</p>' +
        table(["지표", "1차 모델", "2차 모델", "변화율"], perf_rows("exp_results/report_2cls_seed15.json")) +
        '<p class="subtable-title">클래스별 정확도 (mAP50-95)</p>' +
        table(["클래스", "1차 모델", "2차 모델", "변화율"], per_class_rows("exp_results/report_2cls_seed15.json")))

    # ---- 방법 2 ----
    m2 = section("m2", "텍스트 제로샷 (Grounding DINO)", badge("drop"),
        sub("실험 순서") + bullets([
            "<b>방식</b>: 학습 없이 영어 프롬프트 4종('metal hex bolt screw' 등)만으로 박스 생성",
            "<b>데이터</b>: 정답을 숨긴 평가셋 204장, 숨긴 정답과 IoU 0.5 기준 채점",
            "<b>탈락 사유</b>: 유사 금속 부품 간 클래스 혼동 (둥근 접시를 gear 로 오인)"]) +
        sub("실제 입출력 결과") + gallery_html("dino_text") +
        '<p class="fn">초록 = 모델 박스, 빨강 = 정답 라벨 (한 이미지에 겹쳐 표시)</p>' +
        sub("실험 결과") +
        verdict(f"정밀도 {zs_p} → 무검수 라벨 기준(0.87) 미달로 탈락") +
        table(["클래스", "정밀도", "재현율", "맞음", "오탐", "누락"], zs_rows))

    # ---- 방법 3 ----
    m3 = section("m3", "Grounded-SAM 타이트박스", badge("drop"),
        sub("실험 순서") + bullets([
            "<b>방식</b>: 방법 2와 동일 + SAM 마스크로 박스를 픽셀 경계까지 타이트하게 교정",
            "<b>결과</b>: 정밀도 개선 없음. '박스 여백이 문제'라는 가설이 실측으로 기각됨 (맞은 박스 평균 IoU: DINO 원본 0.908 > SAM 0.876)",
            "<b>비고</b>: 증거 이미지는 서버 유실로 미보존"]) +
        sub("실제 입출력 결과") + '<p class="fn">증거 이미지 미보존 - 아래 지표로 확인</p>' +
        sub("실험 결과") +
        verdict(f"정밀도 {gs_p} (방법 2와 동일 수준) → 병목은 박스 여백이 아니라 클래스 혼동으로 판명") +
        table(["클래스", "정밀도", "재현율", "맞음", "오탐", "누락"], gs_rows))

    # ---- 방법 4 ----
    m4 = section("m4", "SAM + CLIP 갤러리", badge("drop"),
        sub("실험 순서") + bullets([
            "<b>방식</b>: SAM 이 물체 후보를 전부 분할 → 각 후보를 참조 갤러리(정답에서 오린 크롭, 클래스당 10장)와 CLIP 임베딩 유사도로 분류",
            "<b>의의</b>: 텍스트 → 시각 매칭 전환으로 정밀도 2.5배 도약 (0.24 → 0.60)",
            "<b>비고</b>: 증거 이미지 미보존"]) +
        sub("실제 입출력 결과") + '<p class="fn">증거 이미지 미보존 - 아래 지표로 확인</p>' +
        sub("실험 결과 (유사도 임계값별)") +
        verdict("최고 정밀도 0.60 → 기준(0.87) 미달. 그러나 '시각 매칭이 텍스트보다 우월'을 입증해 방법 5로 이어짐") +
        table(["유사도 임계값", "정밀도", "재현율", "F1"], clip_rows))

    # ---- 방법 5 ----
    hp = dino_hp or {}
    m5 = section("m5", "SAM + DINOv2 갤러리", badge("adopt"),
        sub("실험 순서") + bullets([
            "<b>방식</b>: 방법 4와 동일하되 임베딩을 CLIP → <b>DINOv2</b>(질감·형상 특징)로 교체",
            f"<b>성과</b>: 임계값 {hp.get('tau')}에서 정밀도 {hp.get('precision')} = 무검수 기준(0.87) 최초 충족",
            "<b>의의</b>: 방법 7(1탭 참조)의 이론적 기반이 됨",
            "<b>비고</b>: 증거 이미지 미보존"]) +
        sub("실제 입출력 결과") + '<p class="fn">증거 이미지 미보존 - 아래 지표로 확인</p>' +
        sub("실험 결과 (유사도 임계값별)") +
        verdict(f"고정밀 운영점 달성: 정밀도 {hp.get('precision')} / 재현율 {hp.get('recall')} (임계값 {hp.get('tau')})") +
        table(["유사도 임계값", "정밀도", "재현율", "F1"], dino_rows))

    # ---- 방법 6 ----
    m6 = section("m6", "상호 일관성 매칭", badge("partial"),
        sub("실험 순서") + bullets([
            "<b>방식</b>: 등록 폴더(부품 1종)의 SAM 후보 중 '다른 모든 사진에도 비슷한 물체가 있는 후보'를 DINOv2 로 식별해 라벨링",
            "<b>적용 ①</b>: 사진 묶음 시뮬 (부품 크롭 2종 × 15장) → 성공",
            "<b>적용 ②</b>: 실사 기어박스 영상 49프레임 → <b>실패</b> (한 장면 영상은 배경도 매 프레임 등장해 매트·드릴·사람까지 오채택)",
            "<b>교훈</b>: 채택률 98%라는 수치만 보면 합격이었으나, 육안 검증이 실패를 적발"]) +
        code_html("m6") +
        sub("실제 입출력 결과") + gallery_html("mutual") +
        sub("실험 결과") +
        verdict("배경이 바뀌는 사진 묶음에서만 유효. 영상 등록에는 부적합 → 방법 7(1탭 참조)로 대체") +
        table(["항목", "결과"], [
            ["사진 묶음 시뮬 (볼트·너트 15장씩)", "채택률 100%, 부품만 정확히 라벨"],
            ["실사 영상 (기어박스 49프레임)", "채택률 98%였으나 배경 오채택 → 폐기"]]))

    # ---- 방법 7 ----
    m7 = section("m7", "1탭 참조 매칭", badge("adopt"),
        sub("실험 순서") + bullets([
            "<b>방식</b>: 등록 화면에서 부품을 <b>한 번 클릭</b> → SAM 포인트 분할로 참조 크롭 확보 → 모든 프레임의 SAM 후보를 참조와 DINOv2 유사도(임계값 0.7)로 매칭",
            "<b>데이터</b>: 실사 기어박스 영상 (1차 33프레임 → 완결 193프레임)",
            "<b>검증</b>: 생성 라벨로 학습 후, 학습에 안 쓴 별도 영상(16프레임)에서 탐지 확인"]) +
        code_html("m7") +
        sub("실제 입출력 결과") + gallery_html("one_tap") +
        sub("실험 결과") +
        verdict("최종 채택. 사람 개입은 탭 1회뿐이며, 라벨 수량이 성능을 직접 좌우 (20장=31% vs 114장=88%) → 등록 영상만 충분히 길면 성능 확보") +
        table(["항목", "결과"], [
            ["라벨 생성 (1차, 33프레임)", "20장 채택, 배경 오채택 0"],
            ["라벨 생성 (완결, 193프레임)", "114장 채택 (59%)"],
            ["학습 후 검증 (라벨 20장)", "미학습 영상 탐지 5/16장 (31%, conf 0.4)"],
            ["학습 후 검증 (라벨 114장)", "미학습 영상 탐지 14/16장 (88%, conf 0.4), 오탐 0"]]))

    # ---- 부록 ----
    bench = jload("bench_results/benchmark.json")
    bench_rows = [[r["model"], r["imgsz"], r["map50"], r["map50_95"],
                   f"{r['latency_ms']}ms", r["fps"], f"{r['weight_MB']}MB"] for r in bench]
    conds = [("2클래스 · 초기 라벨 10%", "exp_results/report_2cls_seed10.json"),
             ("2클래스 · 초기 라벨 15%", "exp_results/report_2cls_seed15.json"),
             ("3클래스", "exp_results/report_3cls.json"),
             ("4클래스", "exp_results/report_4cls.json")]
    cond_rows = []
    for name, rel in conds:
        d = jload(rel)
        cond_rows.append([name, d["split"]["seed"], d["pseudo"]["precision"],
                          d["round0"]["map50"], d["round1"]["map50"],
                          f"+{round(d['delta_map50']*100,1)}%p"])
    appendix = section("appendix", "관련 실험", "",
        sub("배포 모델 벤치마크 (7모델 × 2크기, epochs 100 동일 조건)") +
        callout("선정 = yolo26s@640: 정확도 동급 + 시드 분산 최소 + 지연 최단(NMS-free). 입력 1280은 전 모델에서 이득 없이 지연 1.5~2배") +
        table(["모델", "입력", "mAP50", "mAP50-95", "지연", "FPS", "크기"], bench_rows) +
        sub("오토러닝(방법 1) 조건별 실증") +
        table(["조건", "초기 라벨(장)", "자동 라벨 정밀도", "라운드0 mAP50", "라운드1 mAP50", "효과"], cond_rows))

    toc_items = [("intro", "프로젝트 배경"), ("overview", "결과 한눈에"),
                 ("m1", "self-training [채택]"), ("m2", "텍스트 제로샷"),
                 ("m3", "Grounded-SAM"), ("m4", "SAM+CLIP"),
                 ("m5", "SAM+DINOv2 [채택]"), ("m6", "상호 일관성"),
                 ("m7", "1탭 참조 [채택]"), ("appendix", "부록 · 벤치마크")]
    toc = "".join(f'<a href="#{a}">{t}</a>' for a, t in toc_items)

    intro = section("intro", "프로젝트 배경", "",
        "<p>헬기 정비 부품을 카메라로 자동 인식(탐지)하는 AI 를 만들고 있습니다. AI 학습에는 이미지마다 "
        "부품 위치를 표시한 <b>정답 박스(라벨)</b>가 필요한데, 이를 사람이 일일이 그리는 대신 "
        "<b>자동으로 만드는 방법(오토라벨링)</b> 7가지를 실험하고 비교했습니다.</p>"
        "<p>각 방법마다 같은 틀로 기록했습니다: <b>① 어떤 데이터를 어떻게 사용했나 → "
        "② 만들어진 라벨이 실제로 어떻게 생겼나(정답과 비교) → ③ 수치 성적과 판정.</b></p>" +
        sub("용어") +
        table(["용어", "뜻"], [
            ["라벨 / 바운딩박스", "이미지 속 부품 위치를 표시한 네모 상자. AI 학습의 '정답지'"],
            ["정밀도", "만든 라벨 중 맞은 비율 (오답 라벨이 적을수록 높음)"],
            ["재현율", "실제 부품 중 찾아낸 비율 (놓친 것이 적을수록 높음)"],
            ["mAP50", "탐지 성능 종합 점수 (0~1, 높을수록 좋음)"],
            ["conf (신뢰도)", "모델이 스스로 확신하는 정도. 0.6 이상만 라벨로 채택"],
            ["평가셋", "학습에 쓰지 않고 채점에만 쓰는 별도 문제지"]]))

    overview_sec = section("overview", "결과 한눈에 · 라벨 정밀도", "",
        '<p>7가지 방법의 라벨 품질(정밀도)과 판정. 막대에 마우스를 올리면 값이 보입니다.</p>'
        f'<div class="bars">{overview}</div>')

    css = """
:root{--ink:#1e293b;--muted:#64748b;--accent:#4f46e5;--line:#e2e8f0;--bg:#f8fafc;--card:#fff;}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Noto Sans KR','Malgun Gothic',system-ui,sans-serif;color:var(--ink);background:var(--bg);
     font-size:16px;line-height:1.65}
.layout{display:flex;max-width:1200px;margin:0 auto;gap:28px;padding:32px 20px}
nav{width:220px;flex-shrink:0}
nav .box{position:sticky;top:24px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
nav a{display:block;color:var(--muted);text-decoration:none;font-size:13.5px;padding:5px 8px;border-radius:6px}
nav a:hover{background:var(--bg);color:var(--accent)}
main{flex:1;min-width:0}
h1{font-size:26px;font-weight:700;letter-spacing:-.01em;margin-bottom:4px}
.subtitle{color:var(--muted);margin-bottom:20px}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:24px 26px;margin-bottom:18px}
h2{font-size:19px;font-weight:700;margin-bottom:12px;padding-bottom:10px;border-bottom:1px solid var(--line)}
h3{font-size:15.5px;font-weight:600;margin:18px 0 8px;color:var(--ink)}
ul{padding-left:20px;margin:4px 0}
li{margin:3px 0}
p{margin:6px 0}
table{width:100%;border-collapse:collapse;font-size:14.5px;margin-top:8px;font-variant-numeric:tabular-nums}
th{font-weight:600;text-align:left;border-bottom:2px solid var(--line);padding:7px 10px;background:var(--bg)}
td{border-bottom:1px solid var(--line);padding:6px 10px}
.badge{display:inline-block;font-size:12.5px;font-weight:600;padding:2px 10px;border-radius:20px;vertical-align:2px}
.badge.adopt{background:#eef2ff;color:#3730a3;border:1px solid #c7d2fe}
.badge.drop{background:#f1f5f9;color:#64748b;border:1px solid var(--line)}
.badge.partial{background:#fffbeb;color:#92400e;border:1px solid #fde68a}
.callout{border-left:4px solid var(--accent);background:var(--bg);padding:10px 14px;border-radius:0 8px 8px 0;
         font-weight:600;margin:6px 0 10px}
.method-desc{font-size:16px;font-weight:700;color:var(--ink);margin:6px 0 18px;line-height:1.6}
ol{padding-left:22px}ol li{margin:5px 0}
.subtable-title{font-size:14px;font-weight:600;color:var(--ink);margin:18px 0 6px}
h3.section-h{font-size:16px;font-weight:700;color:var(--ink);border-left:4px solid var(--accent);
             background:var(--bg);padding:9px 14px;border-radius:0 8px 8px 0;margin:30px 0 16px}
.method-desc + h3.section-h{margin-top:10px}
.verdict{font-size:15px;color:var(--ink);margin:4px 0 12px;line-height:1.6}
.snippet{margin:14px 0;border:1px solid var(--line);border-radius:10px;overflow:hidden}
.snip-head{display:flex;flex-wrap:wrap;gap:10px;align-items:baseline;padding:8px 14px;
           background:var(--bg);border-bottom:1px solid var(--line)}
.snip-file{font-family:Consolas,monospace;font-size:12.5px;color:var(--accent);font-weight:600}
.snip-note{display:block;font-size:13.5px;color:var(--muted);margin:16px 0 6px}
.snippet pre{margin:0;padding:12px 16px;background:#0f172a;overflow-x:auto}
.snippet pre code{font-family:Consolas,monospace;font-size:13px;color:#e2e8f0;line-height:1.55;white-space:pre}
.grid{display:grid;gap:14px;margin:8px 0}
.grid.c2{grid-template-columns:1fr 1fr}
figure img{width:100%;border:1px solid var(--line);border-radius:8px;display:block}
figcaption{font-size:13px;color:var(--muted);margin-top:4px;text-align:center}
.fn{font-size:13px;color:var(--muted);margin-top:8px}
.bars{margin-top:10px}
.brow{display:flex;align-items:center;gap:12px;margin:7px 0}
.bname{width:300px;flex-shrink:0;font-size:14.5px;text-align:right}
.btrack{flex:1;background:#eef2f7;border-radius:5px;height:22px;position:relative;display:flex;align-items:center}
.bfill{background:var(--accent);height:100%;border-radius:5px 4px 4px 5px;min-width:2px}
.bval{position:absolute;left:calc(100% + 0px);white-space:nowrap;font-size:13.5px;font-weight:600;
      margin-left:8px;position:static;padding-left:8px;font-variant-numeric:tabular-nums}
.bnote{font-size:13.5px;color:var(--muted);padding-left:10px}
@media(max-width:900px){.layout{flex-direction:column}nav{width:100%}.grid.c2{grid-template-columns:1fr}
.bname{width:150px;font-size:13px}}
"""
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>오토라벨링 검증 리포트</title><style>{css}</style></head><body>
<div class="layout">
<nav><div class="box"><b style="font-size:13px;color:var(--muted)">목차</b>{toc}</div></nav>
<main>
<h1>오토라벨링 검증 리포트</h1>
<p class="subtitle">사람 대신 AI 학습 라벨을 자동 생성하는 7가지 방법의 실험 기록 · XR 오토러닝 프로젝트</p>
{intro}{overview_sec}{m1}{m2}{m3}{m4}{m5}{m6}{m7}{appendix}
<p class="fn" style="text-align:center;margin:10px 0 30px">본 리포트는 scripts/verify/build_report.py 가 실험 결과 파일에서 자동 생성 · 수치 원본: exp_results/, bench_results/, exp_results/zeroshot/</p>
</main></div></body></html>"""
    OUT.write_text(html, encoding="utf-8")
    size = OUT.stat().st_size / 1e6
    print(f"생성 완료: {OUT} ({size:.1f}MB, 비교쌍 {len(pairs)}개)")


if __name__ == "__main__":
    build()
