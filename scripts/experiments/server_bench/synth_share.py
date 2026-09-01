"""이번 학습셋에서 부품별 합성 장수를 센다(gearbox 몫 확인)."""
import collections
from pathlib import Path
L = Path("/workspace/data2/jinwoolee/benchset_panel400/labels/train")
names = [l.split(": ", 1)[1].strip() for l in
         Path("/workspace/data2/jinwoolee/benchset_panel400/data.yaml").read_text(encoding="utf-8").splitlines()
         if l.startswith("  ") and ": " in l]
real = collections.Counter(); syn = collections.Counter()
for p in L.glob("*.txt"):
    tgt = syn if p.name.startswith("syn_") else real
    for line in p.read_text(encoding="utf-8").splitlines():
        f = line.split()
        if len(f) == 5:
            tgt[names[int(f[0])]] += 1
for k in ("gearbox", "split_gearbox", "bracket", "nut"):
    print(f"  {k:<16} 실촬 {real[k]:>5}  합성 {syn[k]:>5}  합성/실촬 {syn[k]/max(1,real[k]):.2f}배")
print(f"  합계             실촬 {sum(real.values()):>5}  합성 {sum(syn.values()):>5}")
