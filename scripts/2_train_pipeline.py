"""2_train_pipeline.py - 자동 라벨링 결과로 새 모델 학습 후 ONNX 변환.

흐름:
  datasets/images + datasets/labels (flat)
     ──(train/val 분할 복사)──> images/{train,val}, labels/{train,val}
     ──(YOLO.train, data.yaml 참조)──> runs/train/weights/best.pt
     ──(복사)──> models/new_model.pt
     ──(YOLO.export)──> models/new_model.onnx

실행:
  python 2_train_pipeline.py
  python 2_train_pipeline.py --epochs 50 --batch 8 --device 0
"""
import argparse
import random
import shutil
from pathlib import Path

import yaml  # ultralytics 가 의존성으로 끌어오므로 별도 설치 불필요
from ultralytics import YOLO

import config


def build_runtime_data_yaml():
    """data.yaml 의 상대 path 를 절대경로로 고정한 '런타임 사본'을 만들어 그 경로를 반환.

    왜 필요한가:
    - ultralytics 는 data.yaml 의 'path' 가 상대경로이면 이 yaml 파일 위치가 아니라
      전역 설정값(datasets_dir)을 기준으로 해석한다. 버전/환경에 따라
      엉뚱한 곳(예: ~/datasets/...)을 보면서 'Dataset not found' 로 실패할 수 있다.
    - 이를 원천 차단하려고 path 를 절대경로로 박은 data.generated.yaml 을 만들어 학습에 쓴다.
    - 사람이 직접 편집하는 data.yaml(클래스명 정의)은 그대로 보존한다(생성본만 교체).
    """
    cfg = yaml.safe_load(config.DATA_YAML.read_text(encoding="utf-8"))
    cfg["path"] = str(config.DATASETS_DIR.resolve())  # 절대경로로 고정
    out = config.BASE_DIR / "data.generated.yaml"
    out.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"[데이터] 절대경로 적용 -> {out} (path={cfg['path']})")
    return out


def prepare_split(images_dir, labels_dir, val_ratio, seed=0):
    """flat 한 images / labels 를 train / val 하위로 분할 복사한다.

    - ultralytics 는 이미지 경로의 마지막 'images' 를 'labels' 로 치환해 라벨을 찾는다.
      따라서 images/train 의 짝은 labels/train 이어야 하고, 둘 다 만들어 줘야 한다.
    - 이미 분할돼 있으면(이전 실행 흔적) 건너뛴다(멱등성 보장).
    - 원본은 보존(copy)한다. train == val 로 평가하는 잘못된 관행을 피하려고 실제로 나눈다.
    """
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    train_img = images_dir / "train"
    if train_img.exists() and any(train_img.iterdir()):
        print("[분할] 기존 train/val 구조를 재사용(분할 건너뜀)")
        return

    # 라벨(.txt)이 실제로 존재하는 이미지만 학습 대상으로 삼는다(짝이 맞는 데이터만).
    pairs = []
    for img in sorted(images_dir.glob("*")):
        if img.suffix.lower() not in config.IMG_EXTS:
            continue
        lbl = labels_dir / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))
    if not pairs:
        raise SystemExit(
            f"[오류] {images_dir} / {labels_dir} 에 짝이 맞는 데이터가 없습니다.\n"
            f"      먼저 1_auto_labeling.py 를 실행해 라벨을 생성하세요."
        )

    # 고정 seed 로 셔플 -> 재현 가능한 분할.
    random.Random(seed).shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio))  # 최소 1장은 검증에 둔다.
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}

    for split, items in splits.items():
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)
        for img, lbl in items:
            shutil.copy2(img, images_dir / split / img.name)
            shutil.copy2(lbl, labels_dir / split / lbl.name)
    print(f"[분할] train {len(splits['train'])}장 / val {len(splits['val'])}장")


def train(args):
    # 1) 데이터셋을 train/val 구조로 정리(멱등).
    prepare_split(config.IMAGES_DIR, config.LABELS_DIR, args.val_ratio)

    # 2) 전이학습 시작점 결정.
    #    base_model 이 있으면 거기서 이어 학습(도메인 지식 유지), 없으면 공개 사전학습 가중치.
    start = args.weights
    if start is None:
        start = str(config.BASE_MODEL) if config.BASE_MODEL.exists() else config.PRETRAINED
    print(f"[학습] 시작 가중치: {start}")

    model = YOLO(start)
    # data.yaml 의 path 를 절대경로로 고정한 런타임 사본을 참조해 학습.
    data_yaml = build_runtime_data_yaml()
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,                 # None 이면 ultralytics 가 GPU/CPU 자동 선택
        project=str(config.BASE_DIR / "runs"),
        name="train",
        exist_ok=True,
        verbose=True,
    )

    # 3) 최고 성능 가중치(best.pt)를 약속된 위치로 복사.
    #    model.trainer.best 는 버전에 관계없이 안정적으로 best.pt 경로를 가리킨다.
    best = Path(model.trainer.best)
    if not best.exists():
        raise SystemExit(f"[오류] best.pt 를 찾지 못했습니다: {best}")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, config.NEW_MODEL_PT)
    print(f"[학습] best.pt -> {config.NEW_MODEL_PT}")

    # 4) ONNX 변환.
    #    Unity/C# 등 비파이썬 런타임에서 onnxruntime 으로 바로 구동할 수 있게 한다.
    export_model = YOLO(str(config.NEW_MODEL_PT))
    onnx_path = export_model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=12,        # 넓은 런타임 호환성을 위한 보수적 opset
        dynamic=False,   # 고정 입력 크기 -> 엣지/임베디드에서 단순하고 안정적
        simplify=True,   # 그래프 단순화로 추론 속도/호환성 개선
    )
    # export() 는 생성된 .onnx 경로(str)를 반환한다. 약속된 위치로 이동.
    onnx_path = Path(onnx_path)
    if onnx_path.resolve() != config.NEW_MODEL_ONNX.resolve():
        shutil.move(str(onnx_path), str(config.NEW_MODEL_ONNX))
    print(f"[변환] ONNX -> {config.NEW_MODEL_ONNX}")


def parse_args():
    ap = argparse.ArgumentParser(description="YOLO 학습 + ONNX 변환 파이프라인")
    ap.add_argument("--weights", default=None, help="시작 가중치(미지정 시 base_model 또는 사전학습)")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--imgsz", type=int, default=config.IMG_SIZE)
    ap.add_argument("--batch", type=int, default=config.BATCH)
    ap.add_argument("--val-ratio", type=float, default=config.VAL_RATIO)
    ap.add_argument("--device", default=None, help="'0'(GPU) / 'cpu'. 미지정 시 자동 선택")
    return ap.parse_args()


if __name__ == "__main__":
    # Windows 의 멀티프로세싱 데이터로더는 반드시 __main__ 가드 안에서 시작돼야 안전하다.
    train(parse_args())
