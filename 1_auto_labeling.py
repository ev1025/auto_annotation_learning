"""1_auto_labeling.py - base_model 로 미라벨 이미지를 자동 라벨링(pseudo-labeling).

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
  python 1_auto_labeling.py
  python 1_auto_labeling.py --conf 0.7 --no-copy-images
  python 1_auto_labeling.py --tta --conf-per-class "gear=0.45"
"""
import argparse
import shutil
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

import config


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


def _iou(a, b):
    """정규화 cxcywh 두 박스의 IoU."""
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / union if union > 0 else 0.0


def _boxes_of(r, flip=False):
    """Results -> [(클래스id, (cx,cy,w,h), conf)]. flip=True 면 좌우반전 좌표를 원본 기준으로 복원."""
    bs = []
    if r.boxes is not None and len(r.boxes) > 0:
        for (cx, cy, w, h), c, cf in zip(r.boxes.xywhn.cpu().numpy(),
                                         r.boxes.cls.cpu().numpy().astype(int),
                                         r.boxes.conf.cpu().numpy()):
            if flip:
                cx = 1.0 - cx
            bs.append((int(c), (float(cx), float(cy), float(w), float(h)), float(cf)))
    return bs


def _consistent(cand, others, iou_thr=0.8):
    """후보 박스가 다른 모든 TTA 뷰에서 같은 클래스 + IoU>=0.8 로 재현되는지."""
    c, box, _ = cand
    return all(any(oc == c and _iou(box, ob) >= iou_thr for oc, ob, _ in other)
               for other in others)


def parse_conf_per_class(spec, model_names):
    """'gear=0.45,bolt=0.7' -> {클래스id: 임계값}. model_names = {id: 이름}."""
    if not spec:
        return {}
    name_to_id = {v: k for k, v in model_names.items()}
    out = {}
    for part in spec.split(","):
        name, _, val = part.partition("=")
        name = name.strip()
        if name not in name_to_id:
            raise SystemExit(f"[오류] --conf-per-class 클래스가 모델에 없음: {name} / 보유: {list(name_to_id)}")
        out[name_to_id[name]] = float(val)
    return out


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
    # 추론은 가장 낮은 임계값으로 하고, 채택 단계에서 클래스별 임계값을 적용한다.
    min_conf = min([args.conf, *conf_per_class.values()]) if conf_per_class else args.conf

    # 처리 대상 이미지 수집(하위 폴더까지 재귀 탐색).
    imgs = [p for p in sorted(src.rglob("*")) if p.suffix.lower() in config.IMG_EXTS]
    if not imgs:
        print(f"[경고] {src} 안에 이미지가 없습니다.")
        return

    def predict_one_image_boxes():
        """이미지별 (경로, 채택 박스 목록) 제너레이터. TTA 여부에 따라 경로가 갈린다."""
        if not args.tta:
            # stream=True: 결과를 제너레이터로 받아 한 장씩 처리 -> 대량 이미지에서도 메모리 누적 없음.
            # (주의) 리스트 source 에서 r.path 는 'image0' 같은 가짜 이름이 될 수 있다(ultralytics 8.4).
            # stream 제너레이터는 입력 순서를 보존하므로 입력 리스트와 zip 으로 원본 경로를 짝짓는다.
            results = model.predict(source=[str(p) for p in imgs], conf=min_conf,
                                    stream=True, verbose=False)
            for img_path, r in zip(imgs, results):
                yield img_path, _boxes_of(r)
        else:
            # TTA: 이미지당 3뷰(원본/좌우반전/0.8배 축소)를 한 배치로 추론.
            # 정규화 좌표(xywhn)라 축소 뷰는 좌표 보정이 필요 없고, 반전 뷰만 cx 를 복원한다.
            for img_path in imgs:
                im = Image.open(img_path).convert("RGB")
                variants = [im,
                            im.transpose(Image.FLIP_LEFT_RIGHT),
                            im.resize((max(32, int(im.width * 0.8)),
                                       max(32, int(im.height * 0.8))))]
                rs = model.predict(source=variants, conf=min_conf, verbose=False)
                cands = _boxes_of(rs[0])
                others = [_boxes_of(rs[1], flip=True), _boxes_of(rs[2])]
                yield img_path, [b for b in cands if _consistent(b, others)]

    n_img, n_box = 0, 0
    for img_path, boxes in predict_one_image_boxes():
        lines = []
        for c, (cx, cy, w, h), cf in boxes:
            # 클래스별 임계값(없으면 공통 conf) 적용.
            if cf < conf_per_class.get(c, args.conf):
                continue
            lines.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

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
