"""5_benchmark.py - 모델 x 입력크기 매트릭스 벤치마크.

목적: 배포 모델 선정 근거 만들기. 어떤 YOLO 계열/크기, 어떤 입력 해상도가
      정확도-속도 균형이 좋은지 같은 데이터로 정량 비교한다.

설계 원칙:
  - 데이터만 바꿔 끼우면 바로 재실행: --src 에 Roboflow YOLO 레이아웃
    (train/valid/test + data.yaml) 폴더를 주면 어떤 데이터셋이든 동일하게 돈다.
  - 모델/입력크기는 리스트 인자: --models ... --imgsz 640 1280
  - 결과는 조합마다 즉시 bench_results/benchmark.json 에 누적 저장(중단돼도 부분 결과 보존),
    종료 시 markdown 비교표 출력.

측정 항목(조합당):
  - 정확도: test split 기준 mAP50 / mAP50-95
  - 속도: 1장 단위 추론 지연(ms)과 FPS (배포 시나리오인 실시간 단건 추론 기준)
  - 규모: 파라미터 수, best.pt 파일 크기, 학습 소요 시간
  ※ 지연/FPS 는 "이 스크립트를 실행한 장비" 기준. Jetson Thor 실측은 Thor 에서
    같은 명령을 다시 실행하면 된다(스크립트는 장비 독립적).

실행:
  python scripts/5_benchmark.py --src ./mechanical-parts-yolo \
      --models yolov8n.pt yolov8s.pt yolov8m.pt yolo11n.pt yolo26n.pt yolo26s.pt yolo26m.pt \
      --imgsz 640 1280 --epochs 100 --device 0

참고(YOLO26 레시피): 크기별 권장 epochs 가 다르다(26n=245, 26s=70, 26m=80).
동일 epochs 벤치는 '같은 학습 비용' 비교라 26n 은 잠재력보다 낮게 나올 수 있음.
26 계열이 유력하면 레시피 조건으로 추가 검증할 것.
"""
import argparse
import json
import time
from pathlib import Path

import yaml
from ultralytics import YOLO

import config
from dataset_utils import normalize_names, write_yaml
from pseudo_utils import free_cuda

BENCH_DIR = config.BASE_DIR / "bench_results"


def build_data_yaml(src):
    """--src 데이터셋으로 벤치마크용 data yaml 생성(절대경로, test split 포함)."""
    names = normalize_names(yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8")).get("names"))
    if not names:
        raise SystemExit(f"[오류] {src / 'data.yaml'} 에서 names 를 해석하지 못했습니다.")
    valid = "valid" if (src / "valid").is_dir() else "val"
    cfg = {
        "path": str(src.resolve()),
        "train": "train/images",
        "val": f"{valid}/images",
        "test": "test/images",
        "names": names,
    }
    out = BENCH_DIR / "bench_data.yaml"
    write_yaml(out, cfg)
    return out, names


def measure_latency(weights, imgs, imgsz, device, n_warmup=10, n_measure=100):
    """단건(batch=1) 추론 지연 측정. 실시간 서빙 시나리오 기준."""
    model = YOLO(str(weights))
    targets = imgs[: n_warmup + n_measure]
    for p in targets[:n_warmup]:
        model.predict(source=str(p), imgsz=imgsz, device=device, verbose=False)
    t0 = time.perf_counter()
    for p in targets[n_warmup:]:
        model.predict(source=str(p), imgsz=imgsz, device=device, verbose=False)
    dt = time.perf_counter() - t0
    n = len(targets) - n_warmup
    del model
    free_cuda()
    return (dt / n) * 1000.0, n / dt  # ms/img, fps


def run_combo(model_name, imgsz, data_yaml, test_imgs, args):
    """모델 x imgsz 1개 조합: 학습 -> test 평가 -> 지연 측정."""
    combo = f"{Path(model_name).stem}_{imgsz}"
    print(f"\n===== [{combo}] 학습 시작 =====")
    model = YOLO(model_name)
    t0 = time.perf_counter()
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=imgsz,
        batch=args.batch,          # -1 = GPU 메모리에 맞춰 자동(조합마다 요구량이 달라서)
        device=args.device,
        project=str(BENCH_DIR / "runs"),
        name=combo,
        exist_ok=True,
        verbose=False,
    )
    train_min = (time.perf_counter() - t0) / 60.0
    best = Path(model.trainer.best)
    n_params = sum(p.numel() for p in model.model.parameters())
    del model
    free_cuda()

    m = YOLO(str(best))
    res = m.val(data=str(data_yaml), split="test", imgsz=imgsz, verbose=False)
    row = {
        "model": Path(model_name).stem,
        "imgsz": imgsz,
        "map50": round(float(res.box.map50), 4),
        "map50_95": round(float(res.box.map), 4),
        "params_M": round(n_params / 1e6, 2),
        "weight_MB": round(best.stat().st_size / 1e6, 1),
        "train_min": round(train_min, 1),
        "best": str(best),
    }
    del m, res
    free_cuda()

    ms, fps = measure_latency(best, test_imgs, imgsz, args.device)
    row["latency_ms"] = round(ms, 1)
    row["fps"] = round(fps, 1)
    return row


def save_and_print(rows, out="benchmark"):
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    (BENCH_DIR / f"{out}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    # markdown 비교표
    cols = ["model", "imgsz", "map50", "map50_95", "latency_ms", "fps",
            "params_M", "weight_MB", "train_min"]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    table = "\n".join(lines)
    (BENCH_DIR / f"{out}.md").write_text(table, encoding="utf-8")
    print("\n" + table)


def main():
    ap = argparse.ArgumentParser(description="모델 x 입력크기 벤치마크")
    ap.add_argument("--src", required=True, help="Roboflow YOLO 레이아웃 데이터셋 폴더")
    ap.add_argument("--models", nargs="+",
                    default=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
                    help="비교할 사전학습 가중치들(자동 다운로드)")
    ap.add_argument("--imgsz", nargs="+", type=int, default=[640],
                    help="비교할 입력 크기들 (예: 640 1280)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = GPU 메모리 자동")
    ap.add_argument("--device", default=0)
    ap.add_argument("--out", default="benchmark",
                    help="결과 파일 이름(확장자 제외). 여러 GPU 병렬 실행 시 서로 다르게 지정")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    data_yaml, names = build_data_yaml(src)
    test_imgs = [p for p in sorted((src / "test" / "images").glob("*"))
                 if p.suffix.lower() in config.IMG_EXTS]
    print(f"[준비] 클래스 {list(names.values())} / test {len(test_imgs)}장")
    print(f"[매트릭스] {[Path(m).stem for m in args.models]} x {args.imgsz} "
          f"= {len(args.models) * len(args.imgsz)}조합, epochs {args.epochs}")

    rows = []
    for model_name in args.models:
        for imgsz in args.imgsz:
            try:
                rows.append(run_combo(model_name, imgsz, data_yaml, test_imgs, args))
            except Exception as e:  # 한 조합이 죽어도 나머지는 계속(부분 결과 보존)
                print(f"[실패] {model_name} x {imgsz}: {e}")
                rows.append({"model": Path(model_name).stem, "imgsz": imgsz,
                             "error": str(e)[:200]})
            save_and_print(rows, args.out)  # 조합마다 누적 저장

    print(f"\n완료: {BENCH_DIR / 'benchmark.json'} / benchmark.md")


if __name__ == "__main__":
    main()
