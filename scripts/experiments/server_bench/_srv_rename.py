# -*- coding: utf-8 -*-
"""서버에서 실행: data/bell412 · results/autolabels · part_codes.json · 3d 파일명 개명."""
import json, io, sys, os
from pathlib import Path

P = Path("/workspace/data2/jinwoolee/xr_dash_260901")
MAP = json.loads(io.open(sys.argv[1], encoding="utf-8").read())
APPLY = "--apply" in sys.argv
SPECIAL = {"mirror": "inspection-mirror"}

def newname(k):
    if k in MAP:
        return MAP[k]
    return MAP.get(SPECIAL.get(k, k.replace("_", "-")), None)

for sub in ("data/bell412", "results/autolabels"):
    root = P / sub
    moves = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        n = newname(d.name)
        if n and n != d.name:
            moves.append((d, root / n))
    print("[%s] %d개 이동, 제외 %s" % (sub, len(moves),
          [d.name for d in sorted(root.iterdir()) if d.is_dir() and not d.name.startswith('_')
           and not newname(d.name)]))
    for src, dst in moves:
        if APPLY:
            assert not dst.exists(), dst
            src.rename(dst)
    if moves:
        print("   예: %s -> %s ... (총 %d)" % (moves[0][0].name, moves[0][1].name, len(moves)))

# 3d 파일명
n3 = 0
for pd in sorted((P / "data/bell412").iterdir()):
    d3 = pd / "3d"
    if not d3.is_dir():
        continue
    for f in sorted(d3.iterdir()):
        if f.stem != pd.name:
            print("   3d: %s -> %s%s" % (f.name, pd.name, f.suffix))
            n3 += 1
            if APPLY:
                f.rename(d3 / (pd.name + f.suffix))
print("[3d] %d개" % n3)

# part_codes.json (코드 번호 보존)
pc = P / "data/bell412/parts/part_codes.json"
d = json.loads(io.open(pc, encoding="utf-8").read())
out, hit = {}, 0
for k, v in d.items():
    n = newname(k)
    out[n or k] = v
    hit += 1 if n and n != k else 0
print("[part_codes] %d개 중 %d개 키 개명" % (len(d), hit))
if APPLY:
    io.open(pc, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=2))

# GT_PARTS (서버는 채점 안 하지만 옛 이름이 남으면 헷갈린다)
gp = P / "backend/autolearning/gt_eval.py"
s = io.open(gp, encoding="utf-8").read()
old = [l for l in s.splitlines() if l.startswith("GT_PARTS = ")]
new = ('GT_PARTS = ["Igb_seal-support", "Igb_r-ring", "Igb_split-gearbox", "Igb_bracket",\n'
       '             "Igb_gear-assembly"]   # 2026-09-07 카테고리 접두(Igb_) 통일')
print("[GT_PARTS] %s" % old)
if APPLY and old:
    io.open(gp, "w", encoding="utf-8").write(s.replace(old[0], new))

print("완료" if APPLY else "미리보기")
