"""release_utils.py - 릴리스 기록 공용 (학습 일시 + 누적 이력).

2_train_pipeline / 7_finetune_real 이 학습을 마칠 때마다 호출한다.
- 릴리스 폴더의 metrics.json 에 학습 일시(trained_at)를 명시적으로 기록
- models/releases/history.jsonl 에 1줄씩 누적 (한 줄 = 학습 1회)
  -> 릴리스 폴더를 일일이 열지 않아도 전체 학습 타임라인을 한 파일로 조회.
     릴리스 폴더가 보관 개수 초과로 정리돼도 이력은 남는다(감사 추적).
"""
import json
from datetime import datetime

import config

HISTORY = config.RELEASES_DIR / "history.jsonl"


def finalize_metrics(rel_dir, metrics):
    """학습 일시를 찍어 metrics.json 저장 + 전체 이력(history.jsonl)에 추가."""
    metrics["trained_at"] = datetime.now().isoformat(timespec="seconds")
    (rel_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
    return metrics
