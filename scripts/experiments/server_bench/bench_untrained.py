# -*- coding: utf-8 -*-
"""미학습 영상 프레임에서 모델별 '인식률'을 재고 비교한다.

왜 필요한가
  학습 곡선의 mAP 는 같은 영상(뒤 15% 블록)에서 오토라벨과의 일치도를 잰 값이다.
  전혀 학습하지 않은 영상에서 부품을 찾는지는 별개 문제라, 정답 박스가 없어도 낼 수 있는
  지표(정답 클래스가 검출된 프레임 비율)를 따로 재서 모델 선택의 근거로 쓴다.

지표
  인식률(hit)      정답 부품 클래스가 검출된 프레임 비율
  오검율(wrong)    정답이 없고 다른 부품만 검출된 프레임 비율
  미검율(none)     아무것도 검출 못한 프레임 비율
  평균신뢰도       정답 클래스 검출의 conf 평균
"""
import json
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(sys.argv[1])          # untrained/<부품>/<영상>/*.jpg
OUT = Path(sys.argv[2])           # bench 결과 폴더(bench.json 이 있는 곳)
CONF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4

bench = json.loads((OUT / "bench.json").read_text(encoding="utf-8"))
res = []
for row in bench:
    w = row["weights"]
    model = YOLO(w)
    names = model.names
    per, agg = {}, {"n": 0, "hit": 0, "wrong": 0, "none": 0, "confs": []}
    for pd in sorted(ROOT.iterdir()):
        if not pd.is_dir():
            continue
        part = pd.name
        for vd in sorted(pd.iterdir()):
            fs = sorted(vd.glob("*.jpg"))
            if not fs:
                continue
            hit = wrong = none = 0
            cf = []
            for i in range(0, len(fs), 8):          # 8장씩 배치 추론
                chunk = [str(p) for p in fs[i:i + 8]]
                for r in model.predict(source=chunk, conf=CONF, imgsz=640, device=0, verbose=False):
                    got = [(names[int(c)], float(v)) for c, v in
                           zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]
                    ok = [v for n, v in got if n == part]
                    if ok:
                        hit += 1; cf.append(max(ok))
                    elif got:
                        wrong += 1
                    else:
                        none += 1
            n = hit + wrong + none
            per[f"{part}/{vd.name}"] = {
                "n": n, "hit": round(hit / n, 3), "wrong": round(wrong / n, 3), "none": round(none / n, 3),
                "conf": round(sum(cf) / len(cf), 3) if cf else None}
            agg["n"] += n; agg["hit"] += hit; agg["wrong"] += wrong; agg["none"] += none; agg["confs"] += cf
    res.append({"model": row["model"],
                "hit": round(agg["hit"] / agg["n"], 3), "wrong": round(agg["wrong"] / agg["n"], 3),
                "none": round(agg["none"] / agg["n"], 3), "n": agg["n"],
                "conf": round(sum(agg["confs"]) / len(agg["confs"]), 3) if agg["confs"] else None,
                "per_video": per})
    print(f"{row['model']:10} 인식률 {res[-1]['hit'] * 100:5.1f}% · 오검 {res[-1]['wrong'] * 100:5.1f}%"
          f" · 미검 {res[-1]['none'] * 100:5.1f}% · conf {res[-1]['conf']}", flush=True)
    del model

(OUT / "bench_untrained.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("\n영상별 인식률")
hdr = "영상"
print(f"{hdr:34} " + " ".join(f"{r['model']:>10}" for r in res))
for k in res[0]["per_video"]:
    print(f"{k:34} " + " ".join(f"{r['per_video'][k]['hit'] * 100:9.1f}%" for r in res))
