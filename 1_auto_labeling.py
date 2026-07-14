"""1_auto_labeling.py - base_model 로 미라벨 이미지를 자동 라벨링(pseudo-labeling).

흐름:
  unlabeled_images/*.jpg ──(base_model.pt 추론, conf>=0.6 만 채택)──┐
                                                                   ├─> datasets/labels/<name>.txt (YOLO 포맷)
                                                                   └─> datasets/images/<name>.jpg (학습이 짝지을 수 있게 복사)

YOLO 라벨 포맷(한 줄당 한 객체):
  <class_id> <x_center> <y_center> <width> <height>
  좌표는 모두 0~1 로 정규화된 값.
  ultralytics 의 boxes.xywhn 이 정확히 이 포맷(정규화 중심좌표+크기)이라 변환 없이 그대로 쓴다.

실행:
  python 1_auto_labeling.py
  python 1_auto_labeling.py --conf 0.7 --no-copy-images
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

import config


def parse_args():
    ap = argparse.ArgumentParser(description="base_model 로 미라벨 이미지 자동 라벨링")
    ap.add_argument("--weights", default=str(config.BASE_MODEL), help="초기 가중치(.pt)")
    ap.add_argument("--source", default=str(config.UNLABELED_DIR), help="미라벨 이미지 폴더")
    ap.add_argument("--labels-out", default=str(config.LABELS_DIR), help=".txt 출력 폴더")
    ap.add_argument("--images-out", default=str(config.IMAGES_DIR), help="이미지 복사 폴더")
    ap.add_argument("--conf", type=float, default=config.AUTO_LABEL_CONF, help="신뢰 임계값(이 값 이상만 채택)")
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

    # 처리 대상 이미지 수집(하위 폴더까지 재귀 탐색).
    imgs = [p for p in sorted(src.rglob("*")) if p.suffix.lower() in config.IMG_EXTS]
    if not imgs:
        print(f"[경고] {src} 안에 이미지가 없습니다.")
        return

    n_img, n_box = 0, 0
    # stream=True: 결과를 제너레이터로 받아 한 장씩 처리 -> 대량 이미지에서도 메모리 누적이 없다.
    # conf 를 모델에 직접 넘겨 임계값 미만은 추론 단계에서 바로 버린다(불필요한 후처리 방지).
    # (주의) 리스트 source 에서 r.path 는 'image0' 같은 가짜 이름이 될 수 있다(ultralytics 8.4).
    # stream 제너레이터는 입력 순서를 보존하므로 입력 리스트와 zip 으로 원본 경로를 짝짓는다.
    results = model.predict(source=[str(p) for p in imgs], conf=args.conf,
                            stream=True, verbose=False)
    for img_path, r in zip(imgs, results):
        boxes = r.boxes
        lines = []
        if boxes is not None and len(boxes) > 0:
            # xywhn = [x_center, y_center, w, h] 를 0~1 로 정규화한 값 = YOLO 라벨 포맷 그대로.
            xywhn = boxes.xywhn.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            for (cx, cy, w, h), c, cf in zip(xywhn, clss, confs):
                if cf < args.conf:   # 안전장치(모델 필터 + 코드 필터 이중화)
                    continue
                lines.append(f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

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

    print(f"\n완료: 이미지 {n_img}장 / 박스 {n_box}건 -> {labels_out}")


if __name__ == "__main__":
    autolabel(parse_args())
