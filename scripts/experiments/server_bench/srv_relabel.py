# -*- coding: utf-8 -*-
"""서버: 이전 라벨 전부 삭제 후 참조샷 기준으로 재생성. 대시보드가 없어 모듈을 직접 쓴다."""
import json
import shutil
import sys
import time
from pathlib import Path

W = Path("/workspace/data2/jinwoolee/xr_dash_260901")
sys.path.insert(0, str(W / "scripts"))
sys.path.insert(0, str(W / "backend" / "autolearning"))
import sam2_autolabel as sa   # noqa: E402

AL = W / "results" / "autolabels"

# 1) 이전 라벨 제거(labels·boxs). 캐시(_cuts·_synth_pool)는 라벨 좌표로 지문을 만들어
#    라벨이 바뀌면 자동으로 새 항목이 되므로 그대로 둔다.
gone = 0
for part in sorted(AL.iterdir()):
    if not part.is_dir() or part.name.startswith("_"):
        continue
    for sub in ("labels", "boxs"):
        d = part / sub
        if d.exists():
            gone += sum(1 for _ in d.glob("*"))
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
print("이전 라벨·박스 파일 %d개 삭제" % gone, flush=True)

# 2) 참조샷에서 작업 목록 구성
items = []
for sj in sorted(AL.glob("*/shots.json")):
    try:
        data = json.loads(sj.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        continue
    for video, frames in sorted(data.items()):
        pairs = [[int(f), [[float(p[0]), float(p[1]), int(p[2])] for p in arr]]
                 for f, arr in sorted((frames or {}).items(), key=lambda kv: int(kv[0])) if arr]
        if pairs:
            items.append({"video": video, "shots": pairs})
print("영상 %d개 · 탭 프레임 %d개" % (len(items), sum(len(i["shots"]) for i in items)), flush=True)

r = sa.start_parts_label_batch("autolabel", items)
if "job" not in r:
    print("시작 실패:", r, flush=True)
    sys.exit(1)
job = r["job"]
print("job", job, flush=True)
t0 = time.time()
last = ""
while True:
    j = sa.JOBS.get(job, {})
    cur = "%s %s/%s %s" % (j.get("stage"), j.get("done"), j.get("total"), (j.get("note") or "")[:44])
    if cur != last:
        print("  %5.1f분 %s" % ((time.time() - t0) / 60, cur), flush=True)
        last = cur
    if not j.get("running"):
        print("종료 stage=%s error=%s" % (j.get("stage"), j.get("error")), flush=True)
        res = j.get("results") or []
        print("영상 %d개 · 라벨 %d장" % (len(res), sum(x.get("labels", 0) for x in res)), flush=True)
        break
    time.sleep(20)
n = sum(1 for _ in AL.glob("*/labels/*.txt"))
print("최종 라벨 파일 %d장" % n, flush=True)
