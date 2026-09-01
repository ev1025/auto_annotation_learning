# -*- coding: utf-8 -*-
"""사무실 배경(숫자 이름)을 _negatives/images -> _negatives/office 로 옮겨 학습에서 제외한다.

격납고 배포에 사무실 배경은 이득 없이 재현율만 깎는다(실측: 배경 417->984 에서 GT 소폭 하락).
삭제하지 않는 이유: 사무실 폰 데모용 학습에는 다시 필요할 수 있다(모니터->gearbox 오검 방지용).
"""
import sys
from pathlib import Path
neg = Path(sys.argv[1])
img, office = neg / "images", neg / "office"
office.mkdir(exist_ok=True)
moved = 0
for p in sorted(img.glob("*.jpg")):
    if p.name[0].isdigit():                 # 숫자 이름 = 사무실(영상 프레임·폰 수집)
        p.replace(office / p.name); moved += 1
print(f"  이동 {moved}장 · 학습 배경 {len(list(img.glob('*.jpg')))}장 · office 보관 {len(list(office.glob('*.jpg')))}장")
