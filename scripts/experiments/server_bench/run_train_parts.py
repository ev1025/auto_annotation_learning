# -*- coding: utf-8 -*-
"""서버 GPU2 다중 부품 학습.

학습 트리   XR_AUTOLABELS  기본 autolabels (밝기보정판은 autolabels_tm)
에폭        XR_EPOCHS      기본 100
제외 부품   XR_EXCLUDE     쉼표 구분. 기본 medicine (프레임이 없어 라벨만 남은 부품)
            `_test` 로 끝나는 부품은 항상 제외
개수 확인   XR_EXPECT      숫자를 주면 그 수와 다를 때 학습 전에 멈춘다. 안 주면 확인 없음.

XR_EXPECT 는 그 실행에서만 쓰는 안전장치다. 예전에 a_test·medicine 이 섞여
37종으로 6시간을 돌린 일이 있어, 원하는 개수를 알고 있을 때만 걸 수 있게 뒀다.
부품이 늘거나 줄면 값을 바꾸거나 아예 주지 않으면 된다.
"""
import os
import sys
import time
from pathlib import Path

W = Path("/workspace/data2/jinwoolee/xr_dash_260901")
sys.path.insert(0, str(W / "scripts"))
sys.path.insert(0, str(W / "backend" / "autolearning"))
import sam2_autolabel as sa      # noqa: E402
import autolabel                 # noqa: E402

EPOCHS = int(os.environ.get("XR_EPOCHS", "100"))
EXCLUDE = {x.strip() for x in os.environ.get("XR_EXCLUDE", "medicine").split(",") if x.strip()}
EXPECT = os.environ.get("XR_EXPECT", "").strip()

AL = autolabel.AUTOLABELS
print("학습 트리: %s" % AL, flush=True)
have = {p.parents[1].name for p in AL.glob("*/labels/*.txt")}
parts = sorted(p for p in have
               if not p.endswith("_test") and not p.startswith("_") and p not in EXCLUDE)
print("학습 부품 %d종: %s" % (len(parts), ", ".join(parts)), flush=True)
print("제외: %s" % (", ".join(sorted(have - set(parts))) or "없음"), flush=True)
if EXPECT:
    if len(parts) != int(EXPECT):
        print("중단: XR_EXPECT=%s 인데 %d종이다. 라벨 상태를 확인하라."
              % (EXPECT, len(parts)), flush=True)
        sys.exit(2)
    print("개수 확인 통과(XR_EXPECT=%s)" % EXPECT, flush=True)

print("라벨 파일 %d장 · epochs %d" % (sum(1 for _ in AL.glob("*/labels/*.txt")), EPOCHS), flush=True)
r = sa.start_multiclass(None, EPOCHS, parts, True)
if "job" not in r:
    print("시작 실패:", r, flush=True)
    sys.exit(1)
job = r["job"]
print("job", job, flush=True)
seen = 0
while True:
    j = sa.job_status(job) if hasattr(sa, "job_status") else sa.JOBS.get(job, {})
    for e in (j.get("log") or [])[seen:]:
        print("  [%s] %s" % (e.get("level"), e.get("msg")), flush=True)
    seen = len(j.get("log") or [])
    print("  ... stage=%s epoch=%s/%s frac=%s %s"
          % (j.get("stage"), j.get("epoch"), j.get("total_epochs"),
             j.get("train_frac"), time.strftime("%H:%M:%S")), flush=True)
    if not j.get("running"):
        print("종료 stage=%s error=%s" % (j.get("stage"), j.get("error")), flush=True)
        break
    time.sleep(60)
