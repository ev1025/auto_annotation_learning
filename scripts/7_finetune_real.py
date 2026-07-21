"""7_finetune_real.py - 실사 소량 파인튜닝 (하이브리드 학습).

배경: 합성(렌더) 데이터로 학습한 모델은 실사에서 도메인 갭으로 성능이 떨어진다.
문헌 실측(README 참고자료 [1]) 기준, 합성 사전학습 모델을 실사 50~100장으로만
파인튜닝해도 저품질 합성 + 실사 400장 수준을 회복한다(라벨 비용 약 8배 절감).

절차(2단계 프로토콜, catastrophic forgetting 방지):
  0) 실사 데이터(images/ + labels/, YOLO 포맷)를 train/val 분할(고정 seed)
  1) 기준선: 시작 가중치를 실사 val 로 평가 (파인튜닝 전 성능)
  2) 1단계: backbone 동결(freeze=10) + lr0 0.01 로 헤드만 적응 (기본 20ep)
  3) 2단계: 전층 해제 + lr0 0.001 + cos_lr + 파인튜닝 특화 증강
     (mosaic 1.0 / mixup 0.4 / hsv 강화: 렌더의 완벽 조명을 현실처럼 왜곡) (기본 40ep)
  4) 게이트: 실사 val 에서 전/후 비교 -> 개선 시에만 서빙 모델(new_model.pt) 교체
  5) 결과를 models/releases/ 에 버전 보관 (2_train_pipeline 과 동일 규약)

실행:
  python scripts/7_finetune_real.py --src ./real_photos --device 0
  # real_photos/images/*.jpg + real_photos/labels/*.txt (클래스 번호는 data.yaml 과 동일 체계)
"""
import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics import YOLO

import config
from dataset_utils import normalize_names, write_yaml
from pseudo_utils import free_cuda

WORK_DIR = config.BASE_DIR / "finetune_real"


def prepare_real_split(src, val_ratio, seed=0):
    """실사 (이미지, 라벨) 짝을 train/val 로 분할해 작업 폴더에 복사."""
    images = sorted(p for p in (src / "images").glob("*") if p.suffix.lower() in config.IMG_EXTS)
    pairs = [(im, src / "labels" / f"{im.stem}.txt") for im in images
             if (src / "labels" / f"{im.stem}.txt").exists()]
    if len(pairs) < 10:
        raise SystemExit(f"[오류] 라벨 짝이 {len(pairs)}건. 최소 10건 필요 "
                         f"(권장 50~100장, 균등 랜덤 샘플링이 hard mining 보다 우수)")
    random.Random(seed).shuffle(pairs)
    n_val = max(2, int(len(pairs) * val_ratio))
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}

    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    for split, items in splits.items():
        (WORK_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (WORK_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        for im, lb in items:
            shutil.copy2(im, WORK_DIR / "images" / split / im.name)
            shutil.copy2(lb, WORK_DIR / "labels" / split / lb.name)
    print(f"[분할] 실사 train {len(splits['train'])}장 / val {len(splits['val'])}장")

    names = normalize_names(yaml.safe_load(config.DATA_YAML.read_text(encoding="utf-8")).get("names"))
    data_yaml = WORK_DIR / "data.yaml"
    write_yaml(data_yaml, {"path": str(WORK_DIR.resolve()),
                           "train": "images/train", "val": "images/val", "names": names})
    return data_yaml


def eval_map50(weights, data_yaml, imgsz, device):
    """실사 val 기준 mAP50 평가."""
    m = YOLO(str(weights))
    res = m.val(data=str(data_yaml), split="val", imgsz=imgsz, device=device, verbose=False)
    out = {"map50": round(float(res.box.map50), 4), "map50_95": round(float(res.box.map), 4)}
    del m, res
    free_cuda()
    return out


def finetune(args):
    src = Path(args.src).resolve()
    data_yaml = prepare_real_split(src, args.val_ratio)

    start = Path(args.weights) if args.weights else config.NEW_MODEL_PT
    if not start.exists():
        raise SystemExit(f"[오류] 시작 가중치가 없습니다: {start} (2_train_pipeline 먼저 실행)")

    # 1) 파인튜닝 전 기준선 (같은 실사 val 로 전/후 공정 비교)
    before = eval_map50(start, data_yaml, args.imgsz, args.device)
    print(f"[기준선] 파인튜닝 전 실사 val mAP50 {before['map50']}")

    common = dict(data=str(data_yaml), imgsz=args.imgsz, batch=args.batch,
                  device=args.device, project=str(WORK_DIR / "runs"), exist_ok=True, verbose=False)

    # 2) 1단계: backbone 동결, 헤드만 실사에 적응
    print(f"\n[1단계] freeze=10, lr0 0.01, {args.epochs1}ep")
    m1 = YOLO(str(start))
    m1.train(epochs=args.epochs1, freeze=10, lr0=0.01, lrf=0.01, name="stage1", **common)
    best1 = Path(m1.trainer.best)
    del m1
    free_cuda()

    # 3) 2단계: 전층 해제, 낮은 lr + 파인튜닝 특화 증강 (기억 파괴 없이 미세 적응)
    print(f"\n[2단계] 전층 해제, lr0 0.001, cos_lr, mosaic1.0/mixup0.4/hsv 강화, {args.epochs2}ep")
    m2 = YOLO(str(best1))
    m2.train(epochs=args.epochs2, lr0=0.001, lrf=0.1, cos_lr=True,
             mosaic=1.0, mixup=0.4, hsv_h=0.02, hsv_s=0.8, hsv_v=0.5,
             name="stage2", **common)
    best2 = Path(m2.trainer.best)
    del m2
    free_cuda()

    # 4) 전/후 비교 게이트 (실사 val 동일 기준)
    after = eval_map50(best2, data_yaml, args.imgsz, args.device)
    improved = after["map50"] > before["map50"]
    print(f"\n[게이트] 실사 val mAP50 {before['map50']} -> {after['map50']} "
          f"({'개선, 채택' if improved else '미개선, 서빙 유지'})")

    # 5) 릴리스 보관 (2_train_pipeline 과 동일 규약: metrics.json + best.pt)
    version = "v" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_finetune"
    rel_dir = config.RELEASES_DIR / version
    rel_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best2, rel_dir / "best.pt")
    metrics = {"version": version, "kind": "finetune_real",
               "real_val_before": before, "real_val_after": after,
               "epochs": [args.epochs1, args.epochs2], "imgsz": args.imgsz,
               "start_weights": str(start), "n_real": len(list((WORK_DIR / 'images' / 'train').iterdir())),
               "status": "promoted" if (improved or args.force_promote) else "rejected"}
    (rel_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                          encoding="utf-8")

    if improved or args.force_promote:
        shutil.copy2(best2, config.NEW_MODEL_PT)
        print(f"[배포] {version} -> {config.NEW_MODEL_PT} (롤백: 6_model_registry.py)")
    else:
        print(f"[보류] {version} 은 보관만. 강제 배포는 --force-promote")


def parse_args():
    ap = argparse.ArgumentParser(description="실사 소량 파인튜닝 (합성 사전학습 -> 2단계 미세조정)")
    ap.add_argument("--src", required=True, help="실사 데이터 폴더 (images/ + labels/)")
    ap.add_argument("--weights", default=None, help="시작 가중치(기본: 서빙 중 new_model.pt)")
    ap.add_argument("--epochs1", type=int, default=20, help="1단계(동결) epochs")
    ap.add_argument("--epochs2", type=int, default=40, help="2단계(해제) epochs")
    ap.add_argument("--imgsz", type=int, default=config.IMG_SIZE)
    ap.add_argument("--batch", type=int, default=config.BATCH)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--force-promote", action="store_true", help="게이트 무시하고 배포")
    return ap.parse_args()


if __name__ == "__main__":
    finetune(parse_args())
