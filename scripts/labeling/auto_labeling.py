"""scripts/labeling/auto_labeling.py - base_model 로 미라벨 이미지를 자동 라벨링(pseudo-labeling).

흐름:
  unlabeled_images/*.jpg ──(base_model.pt 추론, conf>=0.6 만 채택)──┐
                                                                   ├─> datasets/labels/<name>.txt (YOLO 포맷)
                                                                   └─> datasets/images/<name>.jpg (학습이 짝지을 수 있게 복사)

YOLO 라벨 포맷(한 줄당 한 객체):
  <class_id> <x_center> <y_center> <width> <height>
  좌표는 모두 0~1 로 정규화된 값.
  ultralytics 의 boxes.xywhn 이 정확히 이 포맷(정규화 중심좌표+크기)이라 변환 없이 그대로 쓴다.

안전장치 옵션:
  --tta               : TTA 일관성 필터. 원본/좌우반전/0.8배 축소 3뷰에서 모두(같은 클래스,
                        IoU>=0.8) 재현되는 예측만 채택. '자신 있게 틀리는' 예측을 걸러
                        도메인 갭 상황에서 오라벨 유입을 줄인다. (추론 3배, 오프라인이라 무방)
  --conf-per-class    : 클래스별 임계값 덮어쓰기(예: gear=0.45,bolt=0.7).
                        쉬운 클래스만 라벨이 양산되는 '클래스 소외'를 완화.

실행:
  python scripts/labeling/auto_labeling.py
  python scripts/labeling/auto_labeling.py --conf 0.7 --no-copy-images
  python scripts/labeling/auto_labeling.py --tta --conf-per-class "gear=0.45"
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ 공용(config 등)
import config
from pseudo_utils import parse_conf_per_class, predict_boxes


def parse_args():
    ap = argparse.ArgumentParser(description="base_model 로 미라벨 이미지 자동 라벨링")
    ap.add_argument("--weights", default=str(config.BASE_MODEL), help="초기 가중치(.pt)")
    ap.add_argument("--source", default=str(config.UNLABELED_DIR), help="미라벨 이미지 폴더")
    ap.add_argument("--labels-out", default=str(config.LABELS_DIR), help=".txt 출력 폴더")
    ap.add_argument("--images-out", default=str(config.IMAGES_DIR), help="이미지 복사 폴더")
    ap.add_argument("--conf", type=float, default=config.AUTO_LABEL_CONF, help="신뢰 임계값(이 값 이상만 채택)")
    ap.add_argument("--tta", action=argparse.BooleanOptionalAction, default=False,
                    help="TTA 일관성 필터(원본/반전/축소 3뷰 모두 재현되는 예측만 채택)")
    ap.add_argument("--conf-per-class", default=None,
                    help="클래스별 임계값 덮어쓰기 (예: gear=0.45,bolt=0.7)")
    # BooleanOptionalAction: --copy-images / --no-copy-images 로 켜고 끌 수 있다(Python 3.9+).
    ap.add_argument("--copy-images", action=argparse.BooleanOptionalAction, default=True,
                    help="라벨과 짝이 되도록 이미지를 images 폴더로 복사")
    ap.add_argument("--keep-empty", action="store_true",
                    help="검출 0건이어도 빈 .txt 생성(네거티브 샘플로 활용할 때)")
    return ap.parse_args()


def autolabel(args):
    src = Path(args.source)
    labels_out = Path(args.labels_out)
    images_out = Path(args.images_out)
    labels_out.mkdir(parents=True, exist_ok=True)
    images_out.mkdir(parents=True, exist_ok=True)

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"[오류] 초기 가중치가 없습니다: {weights}\n"
                         f"      base_model.pt 를 {config.MODELS_DIR} 에 두고 다시 실행하세요.")

    # 모델은 1회만 로드하고 모든 이미지에 재사용(반복 로드는 느리다).
    model = YOLO(str(weights))
    conf_per_class = parse_conf_per_class(args.conf_per_class, model.names)

    # 처리 대상 이미지 수집(하위 폴더까지 재귀 탐색).
    imgs = [p for p in sorted(src.rglob("*")) if p.suffix.lower() in config.IMG_EXTS]
    if not imgs:
        print(f"[경고] {src} 안에 이미지가 없습니다.")
        return

    n_img, n_box = 0, 0
    # 박스 선택(스트림 추론·TTA·conf 필터)은 pseudo_utils.predict_boxes 단일 구현을 사용.
    for img_path, boxes, _ in predict_boxes(model, imgs, args.conf,
                                            conf_per_class=conf_per_class, tta=args.tta):
        lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                 for c, (cx, cy, w, h), _cf in boxes]

        # 검출이 없고 빈 라벨도 원치 않으면 이 이미지는 건너뛴다.
        if not lines and not args.keep_empty:
            continue

        # 이미지명과 동일한 stem 으로 .txt 생성(YOLO 가 짝을 찾는 규칙).
        txt_path = labels_out / f"{img_path.stem}.txt"
        txt_path.write_text("\n".join(lines), encoding="utf-8")
        if args.copy_images:
            shutil.copy2(img_path, images_out / img_path.name)
        n_img += 1
        n_box += len(lines)
        print(f"[라벨] {img_path.name} -> {len(lines)}건")

    print(f"\n완료: 이미지 {n_img}장 / 박스 {n_box}건 -> {labels_out} (tta={args.tta})")


if __name__ == "__main__":
    autolabel(parse_args())
