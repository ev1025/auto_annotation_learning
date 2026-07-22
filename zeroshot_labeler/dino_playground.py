"""dino_playground.py - Grounding DINO 프롬프트 튜닝 웹 UI.

목적: 프롬프트를 바꿔가며 제로샷 탐지 결과를 눈으로 즉시 확인하는 실험대.
왼쪽에서 이미지 선택(테스트셋 랜덤 or 업로드) + 프롬프트 입력 -> 오른쪽에 탐지 결과.
테스트셋 이미지는 정답(빨간 박스) 겹쳐보기 지원.

실행(서버):
  nohup ./venv_zs/bin/python zeroshot_labeler/dino_playground.py > dino_ui.log 2>&1 &
접속(로컬 PC):
  1) 터미널: ssh -L 7861:127.0.0.1:7861 jinwoolee@203.247.46.194   (창 열어둔 채)
  2) 브라우저: http://localhost:7861
"""
import random
from pathlib import Path

import cv2
import gradio as gr
from autodistill.detection import CaptionOntology
from autodistill_grounding_dino import GroundingDINO

BASE = Path(__file__).resolve().parent.parent
TEST_IMG = BASE / "mechanical-parts-yolo" / "test" / "images"
TEST_LBL = BASE / "mechanical-parts-yolo" / "test" / "labels"
GT_CLASSES = ["bearing", "bolt", "gear", "nut"]

# 실험(exp_zeroshot_eval)에서 사용한 프롬프트 = 초기값. 형식: "프롬프트 : 클래스명" (한 줄에 하나)
DEFAULT_PROMPTS = """metal ball bearing ring : bearing
metal hex bolt screw : bolt
metal gear cog wheel : gear
metal hex nut : nut"""

EXP_NOTE = """### Grounding DINO 프롬프트 실험대
- 위 초기 프롬프트로 test 204장 평가 결과: **정밀도 0.24 / 재현율 0.42** (NMS+거대박스 제거 후)
- 주요 실패: 둥근 접시를 gear 로 오인, 유사 금속 부품 혼동 → **프롬프트를 바꿔 개선되는지 확인해보세요**
- 초록 = 예측(신뢰도), 빨강 = 정답(테스트셋 이미지 + 체크박스 켰을 때)
"""

print("[로드] Grounding DINO 모델 로딩 중... (~20초)")
MODEL = GroundingDINO(ontology=CaptionOntology({"object": "object"}),
                      box_threshold=0.35, text_threshold=0.25)
print("[로드] 완료")


def parse_prompts(text):
    """한 줄 = '프롬프트 : 클래스명' (':' 없으면 프롬프트가 곧 클래스명)."""
    mapping = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            p, c = line.split(":", 1)
            mapping[p.strip()] = c.strip()
        else:
            mapping[line] = line
    return mapping


def detect(image_path, prompts_text, box_thr, text_thr, show_gt):
    if not image_path:
        return None, "이미지를 먼저 선택(랜덤 버튼)하거나 업로드하세요."
    mapping = parse_prompts(prompts_text)
    if not mapping:
        return None, "프롬프트를 한 줄 이상 입력하세요."

    # 모델은 1회만 로드, 프롬프트/임계값은 요청마다 교체(재로드 없음 = 빠른 반복)
    MODEL.ontology = CaptionOntology(mapping)
    MODEL.box_threshold = float(box_thr)
    MODEL.text_threshold = float(text_thr)
    det = MODEL.predict(str(image_path))

    im = cv2.imread(str(image_path))
    classes = list(mapping.values())
    lines = []
    for xy, c, cf in zip(det.xyxy, det.class_id, det.confidence):
        x1, y1, x2, y2 = map(int, xy)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(im, f"{classes[int(c)]} {cf:.2f}", (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        lines.append(f"{classes[int(c)]}  conf {cf:.2f}")

    lbl = TEST_LBL / f"{Path(image_path).stem}.txt"
    if show_gt and lbl.exists():
        h, w = im.shape[:2]
        for line in lbl.read_text().splitlines():
            f = line.split()
            if len(f) < 5:
                continue
            c, cx, cy, bw, bh = int(f[0]), *map(float, f[1:5])
            cv2.rectangle(im, (round((cx - bw / 2) * w), round((cy - bh / 2) * h)),
                          (round((cx + bw / 2) * w), round((cy + bh / 2) * h)), (0, 0, 255), 1)
            cv2.putText(im, GT_CLASSES[c], (round((cx - bw / 2) * w), round((cy + bh / 2) * h) + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    summary = f"탐지 박스 {len(lines)}개\n" + "\n".join(lines[:30])
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB), summary


def random_image():
    return str(random.choice(sorted(TEST_IMG.glob("*.jpg"))))


with gr.Blocks(title="DINO 프롬프트 실험대") as app:
    gr.Markdown(EXP_NOTE)
    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(type="filepath", label="입력 이미지 (업로드 가능)")
            rand_btn = gr.Button("테스트셋 랜덤 이미지 불러오기")
            prompts = gr.Textbox(value=DEFAULT_PROMPTS, lines=6,
                                 label="프롬프트 (한 줄에 하나, '영어 프롬프트 : 클래스명')")
            with gr.Row():
                box_thr = gr.Slider(0.1, 0.9, value=0.35, step=0.05, label="box_threshold")
                text_thr = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="text_threshold")
            show_gt = gr.Checkbox(value=True, label="정답(GT) 빨간 박스 겹쳐보기 (테스트셋 이미지만)")
            run_btn = gr.Button("탐지 실행", variant="primary")
        with gr.Column(scale=2):
            image_out = gr.Image(label="탐지 결과")
            result_txt = gr.Textbox(label="탐지 목록", lines=8)

    rand_btn.click(random_image, outputs=image_in)
    run_btn.click(detect, inputs=[image_in, prompts, box_thr, text_thr, show_gt],
                  outputs=[image_out, result_txt])

if __name__ == "__main__":
    # 127.0.0.1 바인딩: 공유 서버라 외부 노출 금지, SSH 터널로만 접속
    app.launch(server_name="127.0.0.1", server_port=7861)
