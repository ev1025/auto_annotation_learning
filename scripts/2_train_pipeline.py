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
from pseudo_utils import free_cuda


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
    """flat 한 images / labels 를 train / val 하위로 반영한다(누적 학습 지원).

    - ultralytics 는 이미지 경로의 마지막 'images' 를 'labels' 로 치환해 라벨을 찾는다.
      따라서 images/train 의 짝은 labels/train 이어야 하고, 둘 다 만들어 줘야 한다.
    - 최초 실행: 고정 seed 로 train/val 분할을 생성한다.
    - 이후 실행(증분): val 은 최초 분할 그대로 고정하고, 아직 train/val 에 없는
      새 flat 데이터(자동 라벨링으로 추가된 분)만 train 에 추가한다.
      val 을 고정하는 이유: 매번 다시 섞으면 라운드 간 성능 비교가 불가능해지고,
      검증 안 된 자동 라벨이 val 로 흘러들어 평가 자체가 오염되기 때문.
    - 원본 flat 은 보존(copy)한다.
    """
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)

    # 라벨(.txt)이 실제로 존재하는 이미지만 학습 대상으로 삼는다(짝이 맞는 데이터만).
    pairs = []
    for img in sorted(images_dir.glob("*")):
        if img.suffix.lower() not in config.IMG_EXTS:
            continue
        lbl = labels_dir / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))

    train_img = images_dir / "train"
    if train_img.exists() and any(train_img.iterdir()):
        # 증분 모드: 기존 분할(train/val)에 없는 새 데이터만 train 에 추가한다.
        existing = {p.stem for p in (images_dir / "train").glob("*")} | \
                   {p.stem for p in (images_dir / "val").glob("*")}
        new_pairs = [(i, l) for i, l in pairs if i.stem not in existing]
        for img, lbl in new_pairs:
            shutil.copy2(img, images_dir / "train" / img.name)
            shutil.copy2(lbl, labels_dir / "train" / lbl.name)
        n_train = len(list((images_dir / "train").glob("*")))
        n_val = len(list((images_dir / "val").glob("*")))
        print(f"[분할] 기존 val 고정, 신규 {len(new_pairs)}장을 train 에 추가 "
              f"(train {n_train} / val {n_val})")
        return

    if not pairs:
        raise SystemExit(
            f"[오류] {images_dir} / {labels_dir} 에 짝이 맞는 데이터가 없습니다.\n"
            f"      먼저 1_auto_labeling.py 를 실행해 라벨을 생성하세요."
        )

    # 최초 분할: 고정 seed 로 셔플 -> 재현 가능한 분할.
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

    # 학습 직후 GPU 메모리를 회수하고 다음 단계(변환용 모델 로드)로 넘어간다(OOM 방지).
    del model
    free_cuda()

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
