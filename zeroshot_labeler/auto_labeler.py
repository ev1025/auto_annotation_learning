"""auto_labeler.py - 제로샷 오토 어노테이션 (Grounding DINO + SAM).

라벨이 전혀 없는 원시 이미지 폴더를, 텍스트 프롬프트만으로 YOLO 학습용
데이터셋(images + .txt labels + data.yaml)으로 변환한다.

본 레포의 1_auto_labeling.py 와의 관계:
  - 1_auto_labeling.py : "이미 학습된 모델"이 라벨 생성 (오토러닝 루프용)
  - auto_labeler.py    : 모델이 아예 없는 콜드스타트에서 라벨 생성 (초기 라벨셋 부트스트랩)
  즉 이 스크립트로 초기 라벨셋을 만들고 -> 2_train_pipeline 으로 학습하면
  이후에는 1_auto_labeling 루프가 이어받는 구조.

동작:
  1) CaptionOntology 로 "텍스트 프롬프트 -> YOLO 클래스명" 매핑 정의
  2) Grounding DINO 가 프롬프트로 객체 박스를 제로샷 탐지, SAM 이 경계 정밀화
  3) ./raw_images 의 전체 이미지를 라벨링 -> ./yolov8_dataset 에 YOLO 포맷 저장
     (train/valid 자동 분할 + data.yaml 생성)
  4) 클래스별 박스 통계 출력 + (옵션) 미리보기 이미지 저장

실행:
  python auto_labeler.py                          # 기본: ./raw_images -> ./yolov8_dataset
  python auto_labeler.py --input ./imgs --output ./ds --box-thr 0.3 --preview 10

실행 가이드(요약):
  - 전용 venv 필수: torch 를 CUDA 버전에 맞춰 먼저 설치 후 requirements.txt (파일 상단 주석 참고)
  - 첫 실행 시 가중치 자동 다운로드(Grounding DINO ~0.7GB + SAM ViT-H ~2.4GB, 인터넷 필요)
  - VRAM 8GB 이상 권장. CPU 도 동작하나 장당 수십 초로 느림
  - 프롬프트는 영어 + 구체적 시각 묘사가 잘 됨 ("metal hex bolt" > "bolt")
  - 권장 순서: --preview 로 소량 확인 -> 프롬프트/임계값 튜닝 -> 전량 실행
  - 산출 데이터셋의 클래스 번호는 ONTOLOGY 정의 순서. 본 레포에 반입할 땐
    scripts/0_import_render.py 의 클래스 등록 가드를 거쳐 기존 체계와 충돌 확인
"""
import argparse
import shutil
from pathlib import Path

from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM

# ---------------------------------------------------------------------------
# 탐지 대상 정의: "Grounding DINO 에게 줄 영어 프롬프트" -> "YOLO 클래스명"
# 프롬프트는 시각적 특징을 구체적으로 쓸수록 정확하다. 클래스 번호는 이 순서(0..k-1).
# ---------------------------------------------------------------------------
ONTOLOGY = CaptionOntology({
    "metal aircraft engine valve": "engine_valve",
    "metal wrench hand tool":      "wrench",
    "metal hex bolt":              "bolt",
})

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def label_images(args):
    src = Path(args.input).resolve()
    out = Path(args.output).resolve()
    imgs = [p for p in sorted(src.glob("*")) if p.suffix.lower() in IMG_EXTS]
    if not imgs:
        raise SystemExit(f"[오류] {src} 에 이미지가 없습니다. (지원: {sorted(IMG_EXTS)})")
    exts = {p.suffix.lower() for p in imgs}
    print(f"[입력] {src} : {len(imgs)}장 (확장자 {sorted(exts)})")
    print(f"[클래스] {ONTOLOGY.classes()}  (프롬프트 {len(ONTOLOGY.prompts())}개)")

    # 모델 로드 (첫 실행 시 가중치 자동 다운로드)
    base_model = GroundedSAM(
        ontology=ONTOLOGY,
        box_threshold=args.box_thr,    # 낮출수록 재현율↑ 오탐↑ (0.25~0.40 권장 범위)
        text_threshold=args.text_thr,  # 텍스트-영역 매칭 민감도
    )

    # autodistill 의 label() 은 단일 확장자만 받으므로 확장자별로 순회 실행.
    # 같은 output 폴더에 누적되며 train/valid 분할과 data.yaml 은 자동 생성된다.
    dataset = None
    for ext in sorted(exts):
        print(f"\n[라벨링] *{ext} 처리 중...")
        dataset = base_model.label(
            input_folder=str(src),
            extension=ext,
            output_folder=str(out),
        )

    # 클래스별 통계 (라벨 전수 검사 대신 이 수치로 프롬프트 품질을 판단)
    per_class = {}
    n_boxes = 0
    for _, _, det in dataset:
        for cid in det.class_id:
            name = ONTOLOGY.classes()[int(cid)]
            per_class[name] = per_class.get(name, 0) + 1
            n_boxes += 1
    print(f"\n[결과] 총 박스 {n_boxes}개 / 클래스별 {per_class}")
    print("       한 클래스가 0에 가깝다면: 프롬프트 표현을 바꾸거나 --box-thr 를 낮춰 재시도")
    print(f"[저장] {out} (images + labels + data.yaml, train/valid 자동 분할)")

    if args.preview > 0:
        save_previews(dataset, out, args.preview)


def save_previews(dataset, out, n):
    """어노테이션 결과 미리보기 이미지 저장 (프롬프트 튜닝용 육안 확인)."""
    import cv2
    import supervision as sv

    prev_dir = out / "_preview"
    prev_dir.mkdir(parents=True, exist_ok=True)
    box_ann, label_ann = sv.BoxAnnotator(), sv.LabelAnnotator()
    for i, (img_path, image, det) in enumerate(dataset):
        if i >= n:
            break
        labels = [f"{ONTOLOGY.classes()[int(c)]} {conf:.2f}"
                  for c, conf in zip(det.class_id, det.confidence)]
        vis = box_ann.annotate(scene=image.copy(), detections=det)
        vis = label_ann.annotate(scene=vis, detections=det, labels=labels)
        cv2.imwrite(str(prev_dir / f"preview_{Path(img_path).name}"), vis)
    print(f"[미리보기] {prev_dir} 에 {min(n, len(dataset))}장 저장 - 박스 품질 육안 확인 후 전량 진행 권장")


def parse_args():
    ap = argparse.ArgumentParser(description="Grounding DINO 제로샷 오토 어노테이션")
    ap.add_argument("--input", default="./raw_images", help="원시 이미지 폴더")
    ap.add_argument("--output", default="./yolov8_dataset", help="YOLO 데이터셋 출력 폴더")
    ap.add_argument("--box-thr", type=float, default=0.35, help="박스 confidence 임계값")
    ap.add_argument("--text-thr", type=float, default=0.25, help="텍스트 매칭 임계값")
    ap.add_argument("--preview", type=int, default=0, help="N장 미리보기 이미지 저장(0=끔)")
    return ap.parse_args()


if __name__ == "__main__":
    label_images(parse_args())
