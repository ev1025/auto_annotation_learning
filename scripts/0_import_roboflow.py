"""0_import_roboflow.py - Roboflow(YOLOv8/YOLOv11) export 를 이 파이프라인 구조로 변환.

Roboflow 다운로드 zip 을 풀면 보통 이런 구조다:
    <export>/
      ├─ train/images, train/labels
      ├─ valid/images, valid/labels   (또는 val/)
      ├─ test/images,  test/labels    (없을 수도 있음)
      └─ data.yaml                     (names 포함)

이 스크립트는 위를 우리 파이프라인 구조로 옮겨 '오토러닝 리허설'을 바로 시작하게 한다:
  - train(+valid) 이미지/라벨  ->  datasets/images + datasets/labels (flat, '라벨 있는 풀')
  - holdout(기본 test) 이미지  ->  datasets/unlabeled_images (자동 라벨 대상)
  - holdout 라벨              ->  datasets/_gt_holdout (나중에 자동라벨 정확도 채점용 정답 보관)
  - Roboflow data.yaml 의 names -> 우리 data.yaml 의 names 로 동기화

실행:
  python 0_import_roboflow.py --src <풀어놓은 export 폴더>
  python 0_import_roboflow.py --src ./aircraft-component.v1 --holdout valid --no-include-valid
"""
import argparse
import shutil
from pathlib import Path

import yaml  # ultralytics 가 끌어오므로 별도 설치 불필요

import config
from dataset_utils import normalize_names, register_classes


def find_split(src, *names):
    """train/valid/val/test 처럼 이름이 조금씩 다른 split 폴더를 찾아 반환(없으면 None)."""
    for n in names:
        d = src / n
        if (d / "images").is_dir() and (d / "labels").is_dir():
            return d
    return None


def load_names(src):
    """Roboflow data.yaml 의 names 를 {id: name} dict 로 정규화해 반환."""
    y = src / "data.yaml"
    if not y.exists():
        print(f"[경고] {y} 없음 -> 클래스명 동기화 건너뜀")
        return None
    names = normalize_names(yaml.safe_load(y.read_text(encoding="utf-8")).get("names"))
    if not names:
        print("[경고] data.yaml 에서 names 를 해석하지 못함")
    return names


def copy_split(split_dir, images_out, labels_out):
    """split_dir 의 이미지를 images_out 으로, 짝 라벨을 labels_out 으로 복사. 복사 장수 반환.

    labels_out 을 어디로 주느냐로 용도가 갈린다:
    - 라벨풀: datasets/labels (학습이 바로 사용)
    - holdout: datasets/_gt_holdout (이미지는 미라벨 취급, 정답은 채점용으로 숨겨 보관)
    """
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    n = 0
    for img in sorted((split_dir / "images").glob("*")):
        if img.suffix.lower() not in config.IMG_EXTS:
            continue
        shutil.copy2(img, images_out / img.name)
        lbl = split_dir / "labels" / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, labels_out / lbl.name)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description="Roboflow export -> 파이프라인 구조 변환")
    ap.add_argument("--src", required=True, help="풀어놓은 Roboflow export 폴더")
    ap.add_argument("--holdout", default="test", choices=["test", "valid"],
                    help="자동 라벨 대상(unlabeled)으로 뺄 split. 기본 test")
    ap.add_argument("--include-valid", action=argparse.BooleanOptionalAction, default=True,
                    help="valid 를 '라벨 있는 풀'에 함께 넣을지(holdout 이 valid 면 무시)")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    if not src.is_dir():
        raise SystemExit(f"[오류] 폴더가 없습니다: {src}")

    train = find_split(src, "train")
    valid = find_split(src, "valid", "val")
    test = find_split(src, "test")

    # 1) 클래스명 등록(기존 data.yaml 과 클래스가 다르면 안전하게 중단)
    names = load_names(src)
    if names:
        register_classes(names)

    # 2) holdout(미라벨 대상) 결정
    holdout = test if args.holdout == "test" else valid
    if holdout is None:
        raise SystemExit(f"[오류] holdout='{args.holdout}' split 을 찾지 못했습니다.")

    # 3) '라벨 있는 풀' = train (+ valid, holdout 이 valid 가 아니면)
    labeled_splits = [train]
    if args.include_valid and valid is not None and valid != holdout:
        labeled_splits.append(valid)

    total_labeled = 0
    for sp in labeled_splits:
        if sp is None:
            continue
        c = copy_split(sp, config.IMAGES_DIR, config.LABELS_DIR)
        total_labeled += c
        print(f"[라벨풀] {sp.name}: {c}장 -> {config.IMAGES_DIR.name}/{config.LABELS_DIR.name}")

    # 4) holdout -> 이미지는 unlabeled 로, 정답 라벨은 채점용 보관소로
    gt_out = config.DATASETS_DIR / "_gt_holdout"
    c = copy_split(holdout, config.UNLABELED_DIR, gt_out)
    print(f"[미라벨] {holdout.name}: {c}장 -> {config.UNLABELED_DIR.name} (정답은 {gt_out.name} 에 보관)")

    print(f"\n완료: 라벨풀 {total_labeled}장 / 미라벨 {c}장")
    print("다음: python 2_train_pipeline.py  (base_model 만들기)")
    print("      copy models/new_model.pt models/base_model.pt  (첫 생성기로 승격)")
    print("      python 1_auto_labeling.py --weights models/base_model.pt  (자동 라벨)")


if __name__ == "__main__":
    main()
