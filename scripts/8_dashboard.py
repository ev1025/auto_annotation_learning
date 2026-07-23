"""8_dashboard.py - 파이프라인 육안 검증 대시보드 (Gradio).

목적: 자동화 파이프라인의 각 단계를 사람이 눈으로 확인한다.
"겉보기 수치는 좋은데 실제는 실패"(상호 일관성 사례)를 잡는 유일한 수단이 육안 검증이라,
로그·라벨·탐지 결과를 이미지/영상 단위 바운딩박스까지 전부 열람 가능하게 한다.

탭 구성:
  ① 데이터셋 라벨: train/val 이미지 + 라벨 박스 열람 (클래스 필터, 파일명 표시)
  ② 등록 결과: 부품 등록(_preview) 미리보기 갤러리 + 참조 확인샷
  ③ 추론 검증: 이미지/영상 업로드 -> 서빙 모델 추론 -> 박스 그려서 반환 (영상은 mp4)
  ④ 학습 이력: 릴리스 타임라인 (일시·상태·mAP·산입률)

실행(서버):
  nohup ./venv/bin/python scripts/8_dashboard.py > dashboard.log 2>&1 &
접속: http://<서버IP>:7862  (외부 접속 가능, 아래 AUTH 계정으로 로그인)
  - 방화벽이 포트를 막으면 SSH 터널로 대체: ssh -L 7862:127.0.0.1:7862 <서버> 후 localhost:7862
"""
import json
from pathlib import Path

import cv2
import gradio as gr
import yaml

import config

PORT = 7862
# 외부 공개용 로그인 (0.0.0.0 바인딩이므로 인증 필수. 계정 변경은 여기서)
AUTH = ("suri", "xr-auto2026")
PALETTE = [(0, 255, 0), (255, 160, 0), (0, 160, 255), (255, 0, 200), (160, 255, 0),
           (0, 255, 255), (255, 80, 80), (180, 120, 255)]


def class_names():
    if not config.DATA_YAML.exists():
        return {}
    raw = yaml.safe_load(config.DATA_YAML.read_text(encoding="utf-8")).get("names", {})
    if isinstance(raw, list):
        return {i: n for i, n in enumerate(raw)}
    return {int(k): v for k, v in raw.items()}


def draw_box(im, cid, x1, y1, x2, y2, label):
    color = PALETTE[cid % len(PALETTE)]
    cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
    cv2.putText(im, label, (x1, max(y1 - 6, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# ---------------- ① 데이터셋 라벨 ----------------
def list_dataset(split, cls):
    """조건(분할·클래스)에 맞는 이미지 파일 목록."""
    img_dir = config.IMAGES_DIR / split
    lbl_dir = config.LABELS_DIR / split
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


def show_dataset(split, cls, idx):
    files = list_dataset(split, cls)
    if not files:
        return None, "해당 조건의 이미지가 없습니다.", gr.update(maximum=0, value=0)
    idx = int(idx) % len(files)
    name = files[idx]
    im = cv2.imread(str(config.IMAGES_DIR / split / name))
    h, w = im.shape[:2]
    names = class_names()
    lines = []
    lf = config.LABELS_DIR / split / f"{Path(name).stem}.txt"
    if lf.exists():
        for line in lf.read_text().splitlines():
            f = line.split()
            if len(f) < 5:
                continue
            c, cx, cy, bw, bh = int(f[0]), *map(float, f[1:5])
            x1, y1 = round((cx - bw / 2) * w), round((cy - bh / 2) * h)
            x2, y2 = round((cx + bw / 2) * w), round((cy + bh / 2) * h)
            draw_box(im, c, x1, y1, x2, y2, names.get(c, str(c)))
            lines.append(f"{names.get(c, c)}  ({x1},{y1})-({x2},{y2})")
    info = f"{name}  [{idx + 1}/{len(files)}]\n박스 {len(lines)}개\n" + "\n".join(lines)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB), info, gr.update(maximum=max(len(files) - 1, 0))


# ---------------- ② 등록 결과 ----------------
def upload_dirs():
    return sorted(str(d.relative_to(config.BASE_DIR)) for pat in ("uploads*",)
                  for u in config.BASE_DIR.glob(pat) if u.is_dir()
                  for d in u.iterdir() if d.is_dir() and not d.name.startswith("_"))


def show_register(part_dir):
    if not part_dir:
        return [], "등록 폴더를 선택하세요."
    prev = config.BASE_DIR / part_dir / "_preview"
    if not prev.exists():
        return [], f"{part_dir}: 미리보기 없음 (0_register_part 를 먼저 실행)"
    imgs = sorted(prev.glob("*.jpg"))
    ref = [p for p in imgs if p.name == "ref_check.jpg"]
    rest = [p for p in imgs if p.name != "ref_check.jpg"]
    gallery = [(str(p), "참조(탭) 확인") for p in ref] + [(str(p), p.name) for p in rest]
    n_src = len([p for p in (config.BASE_DIR / part_dir).glob("*") if p.suffix.lower() in config.IMG_EXTS])
    return gallery, f"{part_dir}: 원본 {n_src}장 / 미리보기 {len(rest)}장 (초록 = 자동 라벨)"


# ---------------- ③ 추론 검증 ----------------
def get_model():
    from ultralytics import YOLO
    if not config.SERVE_MODEL.exists():
        return None
    if not hasattr(get_model, "m"):
        get_model.m = YOLO(str(config.SERVE_MODEL))
    return get_model.m


NO_MODEL_MSG = ("서빙 모델이 없습니다 (models/new_model.pt). "
                "2_train_pipeline 실행 또는 서버에서 모델 회수 후 사용하세요.")


def infer_image(img_path, conf):
    if not img_path:
        return None, "이미지를 업로드하세요."
    m = get_model()
    if m is None:
        return None, NO_MODEL_MSG
    r = m.predict(source=img_path, conf=conf, verbose=False)[0]
    im = cv2.imread(img_path)
    lines = []
    for b, c, cf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
        x1, y1, x2, y2 = map(int, b)
        name = m.names[int(c)]
        draw_box(im, int(c), x1, y1, x2, y2, f"{name} {float(cf):.2f}")
        lines.append(f"{name}  conf {float(cf):.2f}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB), f"탐지 {len(lines)}개\n" + "\n".join(lines)


def infer_video(video_path, conf, stride):
    """영상 전체를 추론해 바운딩박스 입힌 mp4 반환 (stride 프레임 간격으로 추론, 사이는 박스 유지)."""
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
    i = 0
    last = []          # 최근 추론 박스(중간 프레임에도 표시)
    counts = {}
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
            draw_box(frame, c, x1, y1, x2, y2, f"{m.names[c]} {cf:.2f}")
        vw.write(frame)
        i += 1
    cap.release()
    vw.release()
    summary = f"프레임 {i}개 처리 (추론 {i // stride}회, {stride}프레임 간격)\n" \
              f"클래스별 탐지 누계: {counts if counts else '없음'}"
    return out_path, summary


# ---------------- ④ 학습 이력 ----------------
def show_history():
    hist = config.RELEASES_DIR / "history.jsonl"
    rows = []
    if hist.exists():
        for line in hist.read_text(encoding="utf-8").splitlines():
            m = json.loads(line)
            ing = m.get("ingest", {}).get("train", {})
            rows.append([m.get("trained_at", ""), m.get("version", ""), m.get("status", ""),
                         m.get("map50", ""), m.get("map50_95", ""),
                         f"{ing.get('used', '')}/{ing.get('input', '')} ({ing.get('rate_pct', '')}%)",
                         m.get("epochs", "")])
    rows.reverse()
    serving = "서빙 모델: "
    if config.NEW_MODEL_PT.exists():
        import datetime
        ts = datetime.datetime.fromtimestamp(config.NEW_MODEL_PT.stat().st_mtime)
        serving += f"{config.NEW_MODEL_PT.name} (갱신 {ts:%Y-%m-%d %H:%M}) / 클래스: {list(class_names().values())}"
    else:
        serving += "없음"
    return rows, serving


def render_dataset(split, cls, idx):
    """탭① 렌더 + 슬라이더 동기화."""
    files = list_dataset(split, cls)
    if not files:
        return None, "해당 조건의 이미지가 없습니다. 분할/클래스를 바꿔보세요.", gr.update(value=0, maximum=0)
    idx = int(idx) % len(files)
    img, info, _ = show_dataset(split, cls, idx)
    return img, info, gr.update(value=idx, maximum=len(files) - 1)


def nav_dataset(split, cls, idx, step):
    files = list_dataset(split, cls)
    new = (int(idx) + step) % max(len(files), 1)
    return render_dataset(split, cls, new)


def rand_dataset(split, cls):
    import random as _r
    files = list_dataset(split, cls)
    return render_dataset(split, cls, _r.randrange(max(len(files), 1)))


def status_summary():
    m = get_model()
    model_line = (f"서빙 모델: {config.SERVE_MODEL.name} / 클래스: {list(m.names.values())}"
                  if m else "서빙 모델: 없음 (models/new_model.pt 부재) - ③탭 추론 불가")
    tr = len(list_dataset("train", "전체"))
    va = len(list_dataset("val", "전체"))
    ds_line = f"데이터셋: train {tr}장 / val {va}장 / 클래스 정의: {list(class_names().values())}"
    return model_line + chr(10) + ds_line


with gr.Blocks(title="XR 오토러닝 대시보드") as app:
    gr.Markdown("## XR 오토러닝 육안 검증 대시보드")
    status = gr.Textbox(label="현재 상태", lines=2, interactive=False)

    with gr.Tab("① 데이터셋 라벨"):
        gr.Markdown("**무엇을 확인하는 탭인가**: 학습에 들어가는 이미지와 라벨(박스)을 직접 봅니다. "
                    "박스가 부품을 정확히 감싸는지, 엉뚱한 곳(배경·손·공구)에 라벨이 붙지 않았는지가 판정 포인트입니다. "
                    "자동 라벨이 잘못되면 모델이 그대로 잘못 배우므로, 등록·재학습 후 이 탭으로 표본 확인을 권장합니다.")
        with gr.Row():
            split = gr.Radio(["train", "val"], value="train", label="분할 (train=학습용, val=검증용)")
            cls = gr.Dropdown(["전체"] + list(class_names().values()), value="전체",
                              label="클래스 필터 (해당 부품이 포함된 이미지만)")
        with gr.Row():
            prev_b = gr.Button("← 이전")
            next_b = gr.Button("다음 →")
            rand_b = gr.Button("랜덤")
            idx = gr.Slider(0, 1, step=1, value=0, label="이미지 번호 (드래그로 이동)", scale=3)
        with gr.Row():
            ds_img = gr.Image(label="이미지 + 라벨 박스 (색 = 클래스)", scale=2)
            ds_info = gr.Textbox(label="파일명 · 박스 목록 (클래스, 픽셀 좌표)", lines=10)
        outs = [ds_img, ds_info, idx]
        split.change(render_dataset, [split, cls, idx], outs)
        cls.change(lambda sp, c: render_dataset(sp, c, 0), [split, cls], outs)
        idx.release(render_dataset, [split, cls, idx], outs)
        prev_b.click(lambda sp, c, i: nav_dataset(sp, c, i, -1), [split, cls, idx], outs)
        next_b.click(lambda sp, c, i: nav_dataset(sp, c, i, +1), [split, cls, idx], outs)
        rand_b.click(rand_dataset, [split, cls], outs)

    with gr.Tab("② 등록 결과"):
        gr.Markdown("**무엇을 확인하는 탭인가**: 부품 등록(`0_register_part.py`) 시 자동 생성된 라벨의 미리보기를 봅니다. "
                    "'참조(탭) 확인' 이미지에서 노란 박스가 등록 대상 부품을 제대로 잡았는지 먼저 보고, "
                    "이어서 초록 박스(자동 라벨)가 부품에만 붙었는지 확인하세요. 이상하면 반입 전에 재등록해야 합니다.")
        with gr.Row():
            part = gr.Dropdown(upload_dirs(), label="등록 폴더 (uploads*/부품명)", scale=3)
            reload_b = gr.Button("폴더 목록 갱신")
        reg_info = gr.Textbox(label="요약", interactive=False)
        reg_gal = gr.Gallery(label="자동 라벨 미리보기 (초록 = 라벨, 노란 = 참조 분할)", columns=4, height=520)
        part.change(show_register, part, [reg_gal, reg_info])
        reload_b.click(lambda: gr.update(choices=upload_dirs()), None, part)

    with gr.Tab("③ 추론 검증"):
        gr.Markdown("**무엇을 확인하는 탭인가**: 현재 서빙 모델이 실제로 무엇을 탐지하는지 시험합니다. "
                    "학습에 쓰지 않은 사진/영상을 넣어보는 것이 가장 의미 있는 검증입니다. "
                    "confidence 임계값을 낮추면 더 많이 잡고(오탐↑), 높이면 확실한 것만 잡습니다(누락↑).")
        conf = gr.Slider(0.05, 0.9, value=0.4, step=0.05,
                         label="confidence 임계값 (이 값 이상 확신하는 탐지만 표시)")
        with gr.Row():
            with gr.Column():
                im_in = gr.Image(type="filepath", label="이미지 업로드")
                im_btn = gr.Button("이미지 추론", variant="primary")
            with gr.Column():
                im_out = gr.Image(label="탐지 결과 (박스 + 클래스 + 신뢰도)")
                im_txt = gr.Textbox(label="탐지 목록", lines=5)
        im_btn.click(infer_image, [im_in, conf], [im_out, im_txt])
        gr.Markdown("---")
        with gr.Row():
            with gr.Column():
                vid_in = gr.Video(label="영상 업로드")
                stride = gr.Slider(1, 30, value=5, step=1,
                                   label="추론 간격 (N프레임마다 1회 추론. 클수록 처리 빠름, 박스 갱신은 성김)")
                vid_btn = gr.Button("영상 추론 → 박스 입힌 mp4 생성", variant="primary")
            with gr.Column():
                vid_out = gr.Video(label="어노테이션 영상 (다운로드 가능)")
                vid_txt = gr.Textbox(label="요약", lines=4)
        vid_btn.click(infer_video, [vid_in, conf, stride], [vid_out, vid_txt])

    with gr.Tab("④ 학습 이력"):
        gr.Markdown("**무엇을 확인하는 탭인가**: 언제 무엇이 학습·배포됐는지의 타임라인입니다. "
                    "게이트 = promoted(채택되어 서빙 교체) / rejected(성능 하락으로 보류). "
                    "산입(train) = 입력 이미지 중 실제 학습에 들어간 비율(깨진 파일 제외분)로, 100% 미만이면 제외 사유를 학습 로그에서 확인하세요.")
        hist_btn = gr.Button("새로고침")
        serving_txt = gr.Textbox(label="현재 서빙", interactive=False)
        hist_df = gr.Dataframe(headers=["학습 일시", "버전", "게이트", "mAP50", "mAP50-95",
                                        "산입(train)", "epochs"], label="릴리스 타임라인 (최신순)")
        hist_btn.click(show_history, None, [hist_df, serving_txt])

    # 접속 즉시 초기 화면 자동 로드 (빈 화면 방지)
    app.load(status_summary, None, status)
    app.load(lambda: render_dataset("train", "전체", 0), None, outs)
    app.load(show_history, None, [hist_df, serving_txt])

if __name__ == "__main__":
    # 0.0.0.0 바인딩(외부 접속 허용) + 로그인 인증. 무인증 개방 금지.
    app.launch(server_name="0.0.0.0", server_port=PORT, auth=AUTH)
