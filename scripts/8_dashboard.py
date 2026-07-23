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
접속(로컬 PC):
  1) 터미널: ssh -L 7862:127.0.0.1:7862 <서버>   (창 열어둔 채)
  2) 브라우저: http://localhost:7862
"""
import json
from pathlib import Path

import cv2
import gradio as gr
import yaml

import config

PORT = 7862
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
    if not hasattr(get_model, "m"):
        get_model.m = YOLO(str(config.SERVE_MODEL))
    return get_model.m


def infer_image(img_path, conf):
    if not img_path:
        return None, "이미지를 업로드하세요."
    m = get_model()
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


with gr.Blocks(title="XR 오토러닝 대시보드") as app:
    gr.Markdown("## XR 오토러닝 육안 검증 대시보드\n라벨·등록·추론·이력을 이미지/영상 단위로 확인")

    with gr.Tab("① 데이터셋 라벨"):
        with gr.Row():
            split = gr.Radio(["train", "val"], value="train", label="분할")
            cls = gr.Dropdown(["전체"] + list(class_names().values()), value="전체", label="클래스 필터")
            idx = gr.Slider(0, 1, step=1, value=0, label="이미지 번호")
        with gr.Row():
            ds_img = gr.Image(label="이미지 + 라벨 박스", scale=2)
            ds_info = gr.Textbox(label="파일명·라벨", lines=10)
        for comp in (split, cls, idx):
            comp.change(show_dataset, [split, cls, idx], [ds_img, ds_info, idx])

    with gr.Tab("② 등록 결과"):
        part = gr.Dropdown(upload_dirs(), label="등록 폴더 (uploads*/부품명)")
        reg_btn = gr.Button("불러오기")
        reg_info = gr.Textbox(label="요약")
        reg_gal = gr.Gallery(label="자동 라벨 미리보기", columns=4, height=520)
        reg_btn.click(show_register, part, [reg_gal, reg_info])

    with gr.Tab("③ 추론 검증"):
        conf = gr.Slider(0.05, 0.9, value=0.4, step=0.05, label="confidence 임계값")
        with gr.Row():
            with gr.Column():
                im_in = gr.Image(type="filepath", label="이미지 업로드")
                im_btn = gr.Button("이미지 추론", variant="primary")
            with gr.Column():
                im_out = gr.Image(label="탐지 결과")
                im_txt = gr.Textbox(label="탐지 목록", lines=5)
        im_btn.click(infer_image, [im_in, conf], [im_out, im_txt])
        gr.Markdown("---")
        with gr.Row():
            with gr.Column():
                vid_in = gr.Video(label="영상 업로드")
                stride = gr.Slider(1, 30, value=5, step=1, label="추론 간격(프레임)")
                vid_btn = gr.Button("영상 추론 (박스 입힌 mp4 생성)", variant="primary")
            with gr.Column():
                vid_out = gr.Video(label="어노테이션 영상")
                vid_txt = gr.Textbox(label="요약", lines=4)
        vid_btn.click(infer_video, [vid_in, conf, stride], [vid_out, vid_txt])

    with gr.Tab("④ 학습 이력"):
        hist_btn = gr.Button("새로고침")
        serving_txt = gr.Textbox(label="현재 서빙")
        hist_df = gr.Dataframe(headers=["학습 일시", "버전", "게이트", "mAP50", "mAP50-95",
                                        "산입(train)", "epochs"], label="릴리스 타임라인")
        hist_btn.click(show_history, None, [hist_df, serving_txt])

if __name__ == "__main__":
    # 127.0.0.1 바인딩: 공유 서버라 외부 노출 금지, SSH 터널로만 접속
    app.launch(server_name="127.0.0.1", server_port=PORT)
