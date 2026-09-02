# -*- coding: utf-8 -*-
"""학습 전용 프로세스. 대시보드가 이 파일을 subprocess 로 띄운다.

대시보드와 프로세스를 나눈 이유
  전에는 학습이 대시보드와 같은 프로세스의 스레드였다. 램 캐시가 35,276장을 올리는 동안
  프로세스가 I/O 대기(D 상태)로 멈춰 API 가 통째로 죽었고(정적 파일도 20초 타임아웃),
  그 상태에서는 중단 버튼을 눌러도 요청이 서버에 닿지 못했다.

주고받는 방법
  진행 상황  자식 -> results/_jobs/<job>.json      (JobIO 가 값이 바뀔 때마다 흘린다)
  중단 요청  부모 -> results/_jobs/<job>.json.cancel (JobIO.get("cancel") 이 이 파일을 본다)
  종료 처리  부모의 _reap 스레드가 프로세스 종료를 기다렸다가 잠금을 푼다

인자: 실행 명세 JSON 경로 (job · session · epochs · classes · augment · job_file)
"""
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                      # sam2_autolabel, autolabel
sys.path.insert(0, str(HERE.parents[1] / "scripts"))   # config

import sam2_autolabel as sa  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print("사용법: train_proc.py <spec.json>", file=sys.stderr)
        return 2
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    job = spec["job"]
    j = sa.JobIO(spec["job_file"],
                 {"stage": "start", "running": True, "error": None, "log": [], "curve": []})
    sa.JOBS[job] = j                 # _run_multiclass 는 JOBS[job] 을 통해 상태를 쓴다
    sa._BUSY["on"] = True            # 이 프로세스 안의 다른 작업(합성 등)이 보는 잠금
    sa._ACTIVE.update(job=job, kind="multiclass", session=spec["session"])
    try:
        sa._run_multiclass(job, spec["session"], int(spec["epochs"]),
                           spec.get("classes"), bool(spec.get("augment")))
    except Exception:                # noqa: BLE001 - 무엇이 터졌는지 파일에 남겨야 부모가 안다
        j.update(stage="error", running=False, error=traceback.format_exc()[-400:])
    finally:
        j.update(running=False)
        j.flush(force=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
