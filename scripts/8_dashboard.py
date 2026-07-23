"""8_dashboard.py - 실험 결과 대시보드 (Gradio).

핵심 화면 = "지금까지의 학습·실험을 주제별로 고르면 그 결과값을 보여주는" 브라우저.
결과 원본은 리포에 커밋된 json (exp_results/, bench_results/, zeroshot_labeler/eval_out/).

탭 구성:
  ① 실험 결과: 카테고리(무엇을 학습/실험했나) -> 주제 -> 결과 표 + 요약
  ② 데이터셋 라벨: 학습 이미지 + 라벨 박스 열람 (박스마다 #번호로 목록과 1:1 대응)
  ③ 추론 검증: 이미지/영상 업로드 -> 서빙 모델 탐지 결과

실행: python scripts/8_dashboard.py  ->  http://127.0.0.1:7862 (로그인 AUTH 참고)
"""
import json
from pathlib import Path

import cv2
import gradio as gr
import yaml

import config

PORT = 7862
AUTH = ("suri", "suri")
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


# ==================== ① 실험 결과 ====================
def _fmt(v):
    return round(v, 4) if isinstance(v, float) else v


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
        ["라운드0 mAP50 (초기 라벨만 학습)", r0.get("map50")],
        ["라운드1 mAP50 (자동 라벨 추가 학습)", r1.get("map50")],
        ["효과 (mAP50 변화)", f"+{round(d.get('delta_map50', 0) * 100, 1)}%p"],
        ["효과 (mAP50-95 변화)", f"+{round(d.get('delta_map50_95', 0) * 100, 1)}%p"],
    ]
    if ps.get("tta"):
        rows.append(["TTA 필터", f"제거 박스 {ps.get('tta_dropped_boxes')}개, 정밀도 {ps.get('precision')}"])
    up = d.get("delta_map50", 0) > 0
    summary = ("자동 라벨을 추가 학습하자 성능이 " +
               ("올랐다 -> 오토러닝 효과 입증" if up else "오르지 않았다") +
               f" (mAP50 {r0.get('map50')} -> {r1.get('map50')})")
    return ["항목", "값"], rows, summary


def load_bench(_):
    d = jload("bench_results/benchmark.json")
    if not d:
        return None
    rows = [[r["model"], r["imgsz"], r["map50"], r["map50_95"], r["latency_ms"], r["fps"],
             r["weight_MB"], r["train_min"]] for r in d]
    return (["모델", "입력", "mAP50", "mAP50-95", "지연(ms)", "FPS", "크기(MB)", "학습(분)"], rows,
            "선정 = yolo26s@640: 정확도 동급 + 시드 분산 최소 + 지연 최단(7.0ms, NMS-free). "
            "1280은 전 모델에서 정확도 이득 없이 지연 1.5~2배라 탈락")


def load_followup(_):
    d = jload("bench_results/exp_epochs.json")
    if not d:
        return None
    rows = [[r.get("name"), r.get("base"), f"{r.get('epochs_run')}/{r.get('epochs_set')}",
             r.get("map50"), r.get("map50_95"), r.get("train_min")] for r in d]
    return (["실험", "기반 모델", "epochs(실행/설정)", "mAP50", "mAP50-95", "학습(분)"], rows,
            "결론: 300ep 연장은 개선 없음(조기종료 발동) / 26n 공식 레시피는 역효과 / "
            "시드 3반복으로 'v8s 1위(0.905)는 시드 운' 판명 -> 26s 확정의 근거")


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
            f"엔진: {d.get('engine', 'grounding-dino')} / 임계값 {d.get('box_thr')}{extra} / "
            f"판정: 정밀도 {m.get('precision')} -> 무검수 라벨 기준(0.87) " +
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


def load_gearbox(_):
    rows = [
        ["등록 (영상2, 탭 1회)", "1탭 참조 매칭: 33프레임 중 20장 자동 라벨, 배경 오채택 0"],
        ["실패했던 방식 (기록)", "상호 일관성 매칭: 채택률 98%였으나 매트·드릴·사람까지 오채택 -> 폐기"],
        ["학습 1차 (라벨 20장)", "5클래스 재학습 -> 미학습 영상1 탐지 5/16장 (31%, conf 0.4), 오탐 0"],
        ["학습 완결 (라벨 114장)", "샘플링 6배 재등록 -> 탐지 14/16장 (88%, conf 0.4) / 15/16장 (94%, conf 0.25)"],
        ["산입률", "train 100% (1,954장 전량 학습 산입)"],
    ]
    return (["단계", "결과"], rows,
            "결론: '영상 촬영 + 탭 1회 -> 새 부품 탐지' 전체 사이클 실사 증명. "
            "라벨 수량이 성능을 직접 좌우 (20장=31% vs 114장=88%)")


EXPERIMENTS = {
    "오토러닝 실증 (자동 라벨로 재학습하면 성능이 오르나)": {
        "2클래스 · 초기 라벨 10%": ("exp_results/report_2cls_seed10.json", load_autolearn),
        "2클래스 · 초기 라벨 15%": ("exp_results/report_2cls_seed15.json", load_autolearn),
        "2클래스 · TTA 필터 적용": ("exp_results/report_2cls_tta.json", load_autolearn),
        "3클래스": ("exp_results/report_3cls.json", load_autolearn),
        "4클래스": ("exp_results/report_4cls.json", load_autolearn),
    },
    "모델 벤치마크 (배포 모델 선정)": {
        "7모델 × 2크기 = 14조합": ("bench_results/benchmark.json", load_bench),
        "후속 검증 (300ep·레시피·시드 반복)": ("bench_results/exp_epochs.json", load_followup),
    },
    "콜드스타트 자동 라벨 (라벨·모델 없이 라벨 만들기)": {
        "텍스트 제로샷 (Grounding DINO 원시)": ("zeroshot_labeler/eval_out/zeroshot_eval_raw.json", load_zeroshot),
        "+ 후처리 (NMS·거대박스 제거)": ("zeroshot_labeler/eval_out/zeroshot_eval_post.json", load_zeroshot),
        "+ 임계값 0.5 상향": ("zeroshot_labeler/eval_out/zeroshot_eval_thr05.json", load_zeroshot),
        "Grounded-SAM (타이트 박스)": ("zeroshot_labeler/eval_out/zeroshot_eval_gsam_tight.json", load_zeroshot),
        "SAM + CLIP 갤러리": ("zeroshot_labeler/eval_out/gallery_eval_clip.json", load_sweep),
        "SAM + DINOv2 갤러리 (최고 성능)": ("zeroshot_labeler/eval_out/gallery_eval_dinov2.json", load_sweep),
    },
    "기어박스 실사 E2E (등록 -> 학습 -> 탐지)": {
        "전체 사이클 결과": ("", load_gearbox),
    },
}


def topics_of(cat):
    return list(EXPERIMENTS.get(cat, {}).keys())


def show_experiment(cat, topic):
    entry = EXPERIMENTS.get(cat, {}).get(topic)
    if not entry:
        return gr.update(value=[]), "카테고리와 주제를 선택하세요."
    rel, loader = entry
    res = loader(rel)
    if res is None:
        return gr.update(value=[]), f"결과 파일이 없습니다: {rel}"
    headers, rows, summary = res
    return gr.update(value=rows, headers=headers), summary


def on_category(cat):
    ts = topics_of(cat)
    first = ts[0] if ts else None
    table, summary = show_experiment(cat, first) if first else (gr.update(value=[]), "")
    return gr.update(choices=ts, value=first), table, summary


# ==================== ② 데이터셋 라벨 ====================
def list_dataset(split, cls):
    img_dir, lbl_dir = config.IMAGES_DIR / split, config.LABELS_DIR / split
    names = class_names()
    want = None if cls == "전체" else [k for k, v in names.items() if v == cls]
    out = []
    for p in sorted(img_dir.glob("*")):
        if p.suffix.lower() not in config.IMG_EXTS:
            continue
        lf = lbl_dir / f"{p.stem}.txt"
        if want is None:
            out.append(p.name)
        elif lf.exists() and any(line.split() and int(line.split()[0]) in want
                                 for line in lf.read_text().splitlines()):
            out.append(p.name)
    return out


def render_dataset(split, cls, idx):
    files = list_dataset(split, cls)
    if not files:
        return None, "해당 조건의 이미지가 없습니다.", gr.update(value=0, maximum=0)
    idx = int(idx) % len(files)
    name = files[idx]
    im = cv2.imread(str(config.IMAGES_DIR / split / name))
    h, w = im.shape[:2]
    names = class_names()
    lines = []
    lf = config.LABELS_DIR / split / f"{Path(name).stem}.txt"
    if lf.exists():
        for i, line in enumerate(lf.read_text().splitlines(), start=1):
            f = line.split()
            if len(f) < 5:
                continue
            c, cx, cy, bw, bh = int(f[0]), *map(float, f[1:5])
            x1, y1 = round((cx - bw / 2) * w), round((cy - bh / 2) * h)
            x2, y2 = round((cx + bw / 2) * w), round((cy + bh / 2) * h)
            cname = names.get(c, str(c))
            # 박스와 목록을 #번호로 1:1 대응
            color = PALETTE[c % len(PALETTE)]
            cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
            cv2.putText(im, f"#{i} {cname}", (x1, max(y1 - 6, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            lines.append(f"#{i}  {cname}  (가로 {x2 - x1}px x 세로 {y2 - y1}px)")
    info = f"{name}  [{idx + 1}/{len(files)}]\n박스 {len(lines)}개 (이미지의 #번호와 대응)\n" + "\n".join(lines)
    return (cv2.cvtColor(im, cv2.COLOR_BGR2RGB), info,
            gr.update(value=idx, maximum=max(len(files) - 1, 0)))


def nav_dataset(split, cls, idx, step):
    files = list_dataset(split, cls)
    return render_dataset(split, cls, (int(idx) + step) % max(len(files), 1))


def rand_dataset(split, cls):
    import random as _r
    files = list_dataset(split, cls)
    return render_dataset(split, cls, _r.randrange(max(len(files), 1)))


# ==================== ③ 추론 검증 ====================
def get_model():
    from ultralytics import YOLO
    if not config.SERVE_MODEL.exists():
        return None
    if not hasattr(get_model, "m"):
        get_model.m = YOLO(str(config.SERVE_MODEL))
    return get_model.m


NO_MODEL_MSG = "서빙 모델이 없습니다 (models/new_model.pt)."


def infer_image(img_path, conf):
    if not img_path:
        return None, "이미지를 업로드하세요."
    m = get_model()
    if m is None:
        return None, NO_MODEL_MSG
    r = m.predict(source=img_path, conf=conf, verbose=False)[0]
    im = cv2.imread(img_path)
    lines = []
    for i, (b, c, cf) in enumerate(zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf), start=1):
        x1, y1, x2, y2 = map(int, b)
        name = m.names[int(c)]
        color = PALETTE[int(c) % len(PALETTE)]
        cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
        cv2.putText(im, f"#{i} {name} {float(cf):.2f}", (x1, max(y1 - 6, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        lines.append(f"#{i}  {name}  신뢰도 {float(cf):.2f}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB), f"탐지 {len(lines)}개\n" + "\n".join(lines)


def infer_video(video_path, conf, stride):
    if not video_path:
        return None, "영상을 업로드하세요."
    m = get_model()
    if m is None:
        return None, NO_MODEL_MSG
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = str(Path(video_path).with_name(Path(video_path).stem + "_annotated.mp4"))
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    stride = max(int(stride), 1)
    i, last, counts = 0, [], {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            r = m.predict(source=frame, conf=conf, verbose=False)[0]
            last = [(int(c), tuple(map(int, b)), float(cf))
                    for b, c, cf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf)]
            for c, _, _ in last:
                counts[m.names[c]] = counts.get(m.names[c], 0) + 1
        for c, (x1, y1, x2, y2), cf in last:
            color = PALETTE[c % len(PALETTE)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"{m.names[c]} {cf:.2f}", (x1, max(y1 - 8, 24)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        vw.write(frame)
        i += 1
    cap.release()
    vw.release()
    return out_path, f"프레임 {i}개 처리, 클래스별 탐지 누계: {counts if counts else '없음'}"


# ==================== UI ====================
with gr.Blocks(title="XR 오토러닝 실험 결과") as app:
    gr.Markdown("## XR 오토러닝 실험 결과 대시보드")

    with gr.Tab("① 실험 결과"):
        gr.Markdown("지금까지의 **학습·실험을 주제별로 선택하면 결과값**을 보여줍니다. 수치 원본은 리포에 커밋된 결과 파일입니다.")
        with gr.Row():
            cat = gr.Dropdown(list(EXPERIMENTS.keys()), value=list(EXPERIMENTS.keys())[0],
                              label="카테고리 (무엇을 실험했나)", scale=2)
            topic = gr.Dropdown(topics_of(list(EXPERIMENTS.keys())[0]), label="주제", scale=2)
        exp_summary = gr.Textbox(label="요약·판정", lines=2, interactive=False)
        exp_table = gr.Dataframe(label="결과", interactive=False)
        cat.change(on_category, cat, [topic, exp_table, exp_summary])
        topic.change(show_experiment, [cat, topic], [exp_table, exp_summary])
        app.load(on_category, cat, [topic, exp_table, exp_summary])

    with gr.Tab("② 데이터셋 라벨"):
        gr.Markdown("학습 데이터의 라벨을 눈으로 검사합니다. **이미지 박스의 #번호 = 오른쪽 목록의 #번호**입니다.")
        with gr.Row():
            split = gr.Radio(["train", "val"], value="train", label="분할")
            cls = gr.Dropdown(["전체"] + list(class_names().values()), value="전체", label="클래스 필터")
        with gr.Row():
            prev_b = gr.Button("← 이전")
            next_b = gr.Button("다음 →")
            rand_b = gr.Button("랜덤")
            idx = gr.Slider(0, 1, step=1, value=0, label="이미지 번호", scale=3)
        with gr.Row():
            ds_img = gr.Image(label="이미지 + 라벨 박스 (#번호로 목록과 대응)", scale=2)
            ds_info = gr.Textbox(label="박스 목록", lines=10)
        outs = [ds_img, ds_info, idx]
        split.change(render_dataset, [split, cls, idx], outs)
        cls.change(lambda sp, c: render_dataset(sp, c, 0), [split, cls], outs)
        idx.release(render_dataset, [split, cls, idx], outs)
        prev_b.click(lambda sp, c, i: nav_dataset(sp, c, i, -1), [split, cls, idx], outs)
        next_b.click(lambda sp, c, i: nav_dataset(sp, c, i, +1), [split, cls, idx], outs)
        rand_b.click(rand_dataset, [split, cls], outs)
        app.load(lambda: render_dataset("train", "전체", 0), None, outs)

    with gr.Tab("③ 추론 검증"):
        gr.Markdown("현재 서빙 모델에게 사진/영상을 주고 무엇을 탐지하는지 시험합니다. 학습에 안 쓴 자료가 가장 의미 있는 검증입니다.")
        conf = gr.Slider(0.05, 0.9, value=0.4, step=0.05,
                         label="confidence 임계값 (낮추면 많이 잡고 오탐↑, 높이면 확실한 것만)")
        with gr.Row():
            with gr.Column():
                im_in = gr.Image(type="filepath", label="이미지 업로드")
                im_btn = gr.Button("이미지 추론", variant="primary")
            with gr.Column():
                im_out = gr.Image(label="탐지 결과")
                im_txt = gr.Textbox(label="탐지 목록 (#번호 대응)", lines=5)
        im_btn.click(infer_image, [im_in, conf], [im_out, im_txt])
        gr.Markdown("---")
        with gr.Row():
            with gr.Column():
                vid_in = gr.Video(label="영상 업로드")
                stride = gr.Slider(1, 30, value=5, step=1, label="추론 간격 (N프레임마다 1회)")
                vid_btn = gr.Button("영상 추론 → 박스 입힌 mp4", variant="primary")
            with gr.Column():
                vid_out = gr.Video(label="어노테이션 영상")
                vid_txt = gr.Textbox(label="요약", lines=3)
        vid_btn.click(infer_video, [vid_in, conf, stride], [vid_out, vid_txt])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=PORT, auth=AUTH)
