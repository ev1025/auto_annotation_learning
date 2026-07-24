"""scripts/training/train_pipeline.py - 자동 라벨링 결과로 새 모델 학습 후 ONNX 변환.

흐름:
  training_pool/images + training_pool/labels (flat)
     ──(train/val 분할 복사)──> images/{train,val}, labels/{train,val}
     ──(YOLO.train, data.yaml 참조)──> runs/train/weights/best.pt
     ──(복사)──> models/new_model.pt
     ──(YOLO.export)──> models/new_model.onnx

실행:
  python scripts/training/train_pipeline.py
  python scripts/training/train_pipeline.py --epochs 50 --batch 8 --device 0
"""
import argparse
import csv
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml  # ultralytics 가 의존성으로 끌어오므로 별도 설치 불필요
from ultralytics import YOLO

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ 공용(config 등)
import config
from gpu_utils import free_cuda
from release_utils import finalize_metrics


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
    cfg["path"] = str(config.TRAINING_POOL_DIR.resolve())  # 절대경로로 고정
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
            f"      먼저 scripts/labeling/auto_labeling.py 를 실행해 라벨을 생성하세요."
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


class _Tee:
    """stdout/stderr 를 화면과 로그 파일에 동시에 기록(학습 터미널 로그 보존용)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def _last_metrics(results_csv):
    """ultralytics results.csv 마지막 행에서 val 지표를 추출."""
    rows = list(csv.DictReader(open(results_csv, encoding="utf-8")))
    last = rows[-1]
    pick = {}
    for k, v in last.items():
        k = k.strip()
        if "mAP50(B)" in k:
            pick["map50"] = round(float(v), 4)
        elif "mAP50-95(B)" in k:
            pick["map50_95"] = round(float(v), 4)
        elif "precision(B)" in k:
            pick["precision"] = round(float(v), 4)
        elif "recall(B)" in k:
            pick["recall"] = round(float(v), 4)
    return pick


def _latest_promoted():
    """releases/ 에서 가장 최근에 '채택(promoted)'된 릴리스의 metrics 를 반환(없으면 None)."""
    if not config.RELEASES_DIR.exists():
        return None
    for d in sorted(config.RELEASES_DIR.iterdir(), reverse=True):
        mfile = d / "metrics.json"
        if mfile.exists():
            m = json.loads(mfile.read_text(encoding="utf-8"))
            if m.get("status") == "promoted":
                return m
    return None


def _prune_releases(keep):
    """오래된 릴리스를 keep 개만 남기고 삭제(디스크 관리)."""
    if not config.RELEASES_DIR.exists():
        return
    dirs = sorted([d for d in config.RELEASES_DIR.iterdir() if d.is_dir()])
    for d in dirs[:-keep] if len(dirs) > keep else []:
        shutil.rmtree(d, ignore_errors=True)
        print(f"[정리] 오래된 릴리스 삭제: {d.name}")


def train(args):
    # 0) 릴리스 폴더 준비 + 학습 터미널 로그 tee 시작.
    #    지속 배포에서 "언제 무엇을 어떻게 학습했나"는 사고 조사·롤백 판단의 근거라 전문 보존한다.
    version = "v" + datetime.now().strftime("%Y%m%d_%H%M%S")
    rel_dir = config.RELEASES_DIR / version
    rel_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(rel_dir / "train.log", "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_f)
    sys.stderr = _Tee(sys.__stderr__, log_f)
    print(f"[릴리스] {version} (로그: {rel_dir / 'train.log'})")

    # 1) 데이터셋을 train/val 구조로 정리(멱등).
    prepare_split(config.IMAGES_DIR, config.LABELS_DIR, args.val_ratio)

    # 2) 전이학습 시작점 결정.
    #    base_model 이 있으면 거기서 이어 학습(도메인 지식 유지), 없으면 공개 사전학습 가중치.
    start = args.weights
    if start is None:
        start = str(config.BASE_MODEL) if config.BASE_MODEL.exists() else config.PRETRAINED
    print(f"[학습] 시작 가중치: {start}")

    model = YOLO(start)

    # 학습률(데이터 산입률) 콜백: 입력 대비 실제 학습에 산입된 이미지 비율.
    # ultralytics 는 학습 전 스캔에서 깨진 이미지/라벨을 제외하고 진행하므로
    # (전량 불량이 아닌 한 중단되지 않음) 입력-산입 차이가 곧 '가공 실패분'이다.
    # 시험항목 '학습데이터 학습률' 의 증빙 로그 겸 release metrics 에 기록.
    n_input = {s: len([p for p in (config.IMAGES_DIR / s).glob("*")
                       if p.suffix.lower() in config.IMG_EXTS]) for s in ("train", "val")}
    ingest = {}

    def report_ingest_rate(trainer):
        used = {"train": len(trainer.train_loader.dataset.im_files)}
        val_loader = getattr(trainer, "test_loader", None)
        if val_loader is not None:
            used["val"] = len(val_loader.dataset.im_files)
        for split, n_used in used.items():
            total = n_input.get(split, 0)
            rate = (n_used / total * 100) if total else 0.0
            ingest[split] = {"input": total, "used": n_used, "rate_pct": round(rate, 1)}
            print(f"[학습률] {split}: 입력 {total}장 중 {n_used}장 산입 = {rate:.1f}%")
            if n_used < total:  # 제외된 파일명을 로그에 남겨 원인 추적 가능하게
                folder = {p.name for p in (config.IMAGES_DIR / split).glob("*")
                          if p.suffix.lower() in config.IMG_EXTS}
                loaded = {Path(f).name for f in
                          (trainer.train_loader if split == "train" else val_loader).dataset.im_files}
                print(f"[학습률] {split} 제외 파일: {sorted(folder - loaded)[:20]}")

    model.add_callback("on_train_start", report_ingest_rate)

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

    # 3) 산출물을 릴리스 폴더에 보관(best.pt / 지표 / 학습곡선 csv).
    best = Path(model.trainer.best)
    if not best.exists():
        raise SystemExit(f"[오류] best.pt 를 찾지 못했습니다: {best}")
    run_dir = best.parent.parent  # runs/train
    shutil.copy2(best, rel_dir / "best.pt")
    if (run_dir / "results.csv").exists():
        shutil.copy2(run_dir / "results.csv", rel_dir / "results.csv")

    metrics = _last_metrics(rel_dir / "results.csv") if (rel_dir / "results.csv").exists() else {}
    metrics.update({"version": version, "epochs": args.epochs, "imgsz": args.imgsz,
                    "start_weights": str(start), "ingest": ingest})

    del model
    free_cuda()  # 학습 직후 GPU 메모리 회수(다음 단계 OOM 방지)

    # 4) 배포 게이트: 직전 채택본보다 mAP50 이 떨어지면 서빙 모델을 갈아끼우지 않는다.
    #    나빠진 모델의 자동 배포(오토러닝 오류 증폭)를 차단하는 운영 안전장치.
    prev = _latest_promoted()
    drop_ok = True
    if prev and "map50" in metrics and not args.force_promote:
        drop_ok = metrics["map50"] >= prev["map50"] - config.PROMOTE_MIN_DROP
        print(f"[게이트] 직전 채택본 mAP50 {prev['map50']} vs 신규 {metrics['map50']} "
              f"-> {'통과' if drop_ok else '미달(배포 보류)'}")

    if not drop_ok:
        metrics["status"] = "rejected"
        finalize_metrics(rel_dir, metrics)
        print(f"[보류] {version} 은 보관만 하고 서빙 모델은 유지합니다. "
              f"강제 배포는 --force-promote, 복원은 scripts/training/model_registry.py rollback")
        _prune_releases(config.KEEP_RELEASES)
        return

    # 5) 채택: 서빙 위치 갱신 + ONNX 변환(릴리스에도 보관).
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rel_dir / "best.pt", config.NEW_MODEL_PT)
    print(f"[학습] best.pt -> {config.NEW_MODEL_PT}")

    export_model = YOLO(str(config.NEW_MODEL_PT))
    onnx_path = Path(export_model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=12,        # 넓은 런타임 호환성을 위한 보수적 opset
        dynamic=False,   # 고정 입력 크기 -> 엣지/임베디드에서 단순하고 안정적
        simplify=True,   # 그래프 단순화로 추론 속도/호환성 개선
    ))
    if onnx_path.resolve() != config.NEW_MODEL_ONNX.resolve():
        shutil.move(str(onnx_path), str(config.NEW_MODEL_ONNX))
    shutil.copy2(config.NEW_MODEL_ONNX, rel_dir / "best.onnx")
    print(f"[변환] ONNX -> {config.NEW_MODEL_ONNX}")

    metrics["status"] = "promoted"
    finalize_metrics(rel_dir, metrics)
    print(f"[배포] {version} 채택 완료 (보관: {rel_dir})")
    _prune_releases(config.KEEP_RELEASES)


def parse_args():
    ap = argparse.ArgumentParser(description="YOLO 학습 + 릴리스 보관 + 배포 게이트 + ONNX 변환")
    ap.add_argument("--weights", default=None, help="시작 가중치(미지정 시 base_model 또는 사전학습)")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--imgsz", type=int, default=config.IMG_SIZE)
    ap.add_argument("--batch", type=int, default=config.BATCH)
    ap.add_argument("--val-ratio", type=float, default=config.VAL_RATIO)
    ap.add_argument("--device", default=None, help="'0'(GPU) / 'cpu'. 미지정 시 자동 선택")
    ap.add_argument("--force-promote", action="store_true",
                    help="배포 게이트 무시하고 무조건 서빙 모델 갱신")
    return ap.parse_args()


if __name__ == "__main__":
    # Windows 의 멀티프로세싱 데이터로더는 반드시 __main__ 가드 안에서 시작돼야 안전하다.
    train(parse_args())
