# -*- coding: utf-8 -*-
"""drop_nested 자체 점검. 실행: python backend/yolo_server/test_drop_nested.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import drop_nested   # noqa: E402


def det(cls, x, y, w, h, conf):
    return {"detection_class": cls, "confidence": conf, "bbox": {"x": x, "y": y, "w": w, "h": h}}


# 1) 같은 부품, 좁은 박스가 넓은 박스 안에 있음 -> 신뢰도 높은 하나만
wide, narrow = det("a_test", 0, 0, 100, 100, 0.9), det("a_test", 20, 20, 40, 40, 0.8)
assert drop_nested([wide, narrow]) == [wide]
# 신뢰도가 뒤집혀도 높은 쪽이 남는다(좁은 게 더 확실하면 좁은 게 남음)
assert drop_nested([det("a_test", 0, 0, 100, 100, 0.7), det("a_test", 20, 20, 40, 40, 0.95)])[0]["bbox"]["w"] == 40

# 2) 같은 부품이지만 떨어져 있으면 둘 다 남는다(부품 2개가 화면에 있는 경우)
a, b = det("nut", 0, 0, 50, 50, 0.9), det("nut", 200, 200, 50, 50, 0.9)
assert len(drop_nested([a, b])) == 2

# 3) 다른 부품이 겹쳐 있으면 지우지 않는다(클래스 혼동은 여기서 판단하지 않는다)
assert len(drop_nested([det("nut", 0, 0, 100, 100, 0.9), det("washer", 20, 20, 40, 40, 0.8)])) == 2

# 4) 절반만 겹치면(8할 미만) 남긴다
assert len(drop_nested([det("nut", 0, 0, 100, 100, 0.9), det("nut", 80, 0, 40, 100, 0.8)])) == 2

# 5) 빈 입력
assert drop_nested([]) == []

print("drop_nested 점검 5건 통과")
