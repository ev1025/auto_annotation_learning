"""run_bench.py 에 7/30 레시피(mixup+erasing) 모드를 추가한다."""
from pathlib import Path
p = Path("/workspace/data2/jinwoolee/run_bench.py")
s = p.read_text(encoding="utf-8")
old = '''if len(sys.argv) > 8 and sys.argv[8] == "strong":
    AUG = {"degrees": 45.0, "flipud": 0.5, "perspective": 0.0005, "shear": 5.0, "scale": 0.7}'''
new = '''if len(sys.argv) > 8 and sys.argv[8] == "strong":
    AUG = {"degrees": 45.0, "flipud": 0.5, "perspective": 0.0005, "shear": 5.0, "scale": 0.7}
# occl = 7/30 에 장착 도메인(test3·4)을 뚫은 레시피. 가림 시뮬레이션(erasing) + 배경 교란(mixup).
# 그때 test3·4 가 0·3% -> 99·80% 로 올라간 조합이라 그대로 되살린다.
elif len(sys.argv) > 8 and sys.argv[8] == "occl":
    AUG = {"degrees": 15.0, "mixup": 0.2, "erasing": 0.5}'''
if "occl" in s:
    print("이미 있음")
else:
    assert s.count(old) == 1
    p.write_text(s.replace(old, new), encoding="utf-8")
    print("occl 모드 추가")
