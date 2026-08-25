# -*- coding: utf-8 -*-
"""GT 채점 결과에 부트스트랩 신뢰구간을 붙인다.

GT 가 70장뿐이라 적중 장수 43 과 48 의 차이가 실력인지 우연인지 눈으로는 알 수 없다.
70장에서 복원추출(중복 허용)로 새 70장을 2000번 만들어 적중 장수의 분포를 보면,
그 차이가 0 을 포함하는지 알 수 있다. 0 을 포함하면 "차이 없음" 으로 읽어야 한다.

  python scripts/experiments/gt_bootstrap.py                 conf 0.25(뽑을 때 기준) 로 전 모델
  python scripts/experiments/gt_bootstrap.py 0.7             운영 임계(conf 0.7) 로 다시 채점
  python scripts/experiments/gt_bootstrap.py 0.7 c36_3x:yolo11m c36_panel400:yolo26s
                                                             두 모델을 짝지어 차이의 구간까지

모델을 두 개 지정하면 "같은 사진에서의 차이" 를 부트스트랩한다(짝지은 비교).
사진마다 난이도가 달라서, 각각의 구간을 따로 보는 것보다 차이를 보는 쪽이 훨씬 좁고 정확하다.
"""
import json
import random
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
PREDS = BASE / "scripts" / "gt_preds.js"
N_BOOT = 2000
SEED = 0


def load():
    """gt_preds.js 에서 window.D 의 JSON 만 떼어 읽는다."""
    s = PREDS.read_text(encoding="utf-8")
    return json.loads(s[s.index("{"):s.rstrip().rstrip(";").rindex("}") + 1])


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    i = (x2 - x1) * (y2 - y1)
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i)


def hit_vector(data, exp_id, model, conf):
    """사진 순서대로 적중(1)/미검출(0). conf 이상인 예측만 센다."""
    exp = next(e for e in data["exps"] if e["id"] == exp_id)
    frames = exp["models"][model]
    out = []
    for im in data["images"]:
        best = 0.0
        for b in frames[im["file"]]["p"]:
            if b[4] < conf or b[5] != im["part"]:      # 신뢰도 미달이거나 다른 부품이면 제외
                continue
            for g in im["gt"]:
                best = max(best, iou(b[:4], g))
        out.append(1 if best >= 0.5 else 0)
    return out


def boot_index(n):
    """모든 모델에 같은 재표본을 쓴다. 그래야 모델 간 비교가 짝지어진다."""
    rnd = random.Random(SEED)
    return [[rnd.randrange(n) for _ in range(n)] for _ in range(N_BOOT)]


def ci(values):
    """2000개 표본에서 95% 구간(2.5% ~ 97.5% 분위수)."""
    v = sorted(values)
    return v[int(N_BOOT * 0.025)], v[int(N_BOOT * 0.975)]


def main():
    args = [a for a in sys.argv[1:]]
    conf = 0.25
    if args and args[0].replace(".", "", 1).isdigit():
        conf = float(args.pop(0))
    data = load()
    n = len(data["images"])
    idx = boot_index(n)
    print(f"GT {n}장 · conf >= {conf} · 부트스트랩 {N_BOOT}회")

    if len(args) == 2:                                  # 두 모델 짝지은 비교
        vs = []
        for spec in args:
            e, m = spec.split(":")
            vs.append(hit_vector(data, e, m, conf))
        diff = [vs[0][i] - vs[1][i] for i in range(n)]
        lo, hi = ci([sum(diff[i] for i in ix) for ix in idx])
        print(f"  {args[0]} {sum(vs[0])}/{n}   {args[1]} {sum(vs[1])}/{n}")
        print(f"  차이 {sum(diff):+d}  95% CI [{lo:+d}, {hi:+d}]  -> "
              + ("유의한 차이" if (lo > 0 or hi < 0) else "차이 없음(구간이 0을 포함)"))
        return

    rows = []                                           # 지정 없으면 전 모델 나열
    for e in data["exps"]:
        for m in e["models"]:
            v = hit_vector(data, e["id"], m, conf)
            lo, hi = ci([sum(v[i] for i in ix) for ix in idx])
            rows.append((sum(v), e["id"], m, lo, hi))
    for h, eid, m, lo, hi in sorted(rows, reverse=True):
        print(f"  {eid:15s} {m:9s} {h:2d}/{n}  95% CI [{lo:2d}, {hi:2d}]")


if __name__ == "__main__":
    main()
