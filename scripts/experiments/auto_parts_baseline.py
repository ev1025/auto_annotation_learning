# -*- coding: utf-8 -*-
"""자동 탭(센터포인트) 멀티클래스 baseline: 사람 탭 없이 파이프라인을 끝까지 돌려 기준선을 만든다.

데이터 = data/bell412/parts/<부품>/videos/*.mp4 (폴더=부품, 폴더 안 영상=그 부품의 train/test 테이크).
테스트 규칙: 부품에 "2" 테이크(끝수 최대)나 test* 가 있으면 그걸 테스트로, 없으면(단일) 학습영상 자체로 테스트.
부품 촬영은 '부품을 어두운 매트 위 중앙에 놓고 카메라가 도는' 구도라, 영상 중앙부 프레임에 화면
중앙점(0.5,0.5)을 찍으면 SAM2가 부품을 잡는다(육안 검증). 이 자동 탭으로 학습 테이크마다 라벨을 만들고
→ 34클래스 통합 학습 → 부품별 테스트 테이크 검출 평가.

목적: '자동 탭 baseline' 대비 '사람이 직접 탭'이 성능을 얼마나 올리는지 비교하는 기준선.
대시보드 백엔드(sam2_autolabel)의 워커 함수를 그대로 재사용 = 사람 탭과 동일한 경로.

사용: python scripts/experiments/auto_parts_baseline.py [--session auto_baseline] [--epochs 100]
      [--smoke] (부품 3개·5epoch) [--no-train] (라벨만)
윈도우 multiprocessing 대비 __main__ 가드 필수."""
import os
import re
import sys
import glob
import argparse

BASE = os.environ.get("XR_BASE") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, BASE + "/scripts")
sys.path.insert(0, BASE + "/scripts/verify")
PARTS = BASE + "/data/bell412"          # 각 부품 = bell412/<부품>/videos (평탄 구조). '_' 접두 폴더는 제외

CENTER_PTS = [[0.5, 0.5, 1]]     # 화면 중앙 포함점(rx, ry, lab=1)
REF_FRACS = [0.35, 0.5, 0.65]    # 영상 중앙부 3프레임을 참조샷으로


def trail_num(s):
    m = re.search(r"(\d+)\s*$", s)
    return int(m.group(1)) if m else 0


def take_roles(stems):
    """부품 테이크 목록 → (학습 테이크들, 테스트 테이크). test* 우선, 없고 2개+면 끝수 최대가 테스트, 단일이면 학습영상 자체."""
    if not stems:
        return [], None
    explicit = [n for n in stems if "test" in n.lower()]
    if explicit:
        return [n for n in stems if "test" not in n.lower()], explicit[-1]
    if len(stems) >= 2:
        s = sorted(stems, key=trail_num)
        return s[:-1], s[-1]
    return stems, stems[0]


def scan_parts():
    """data/bell412/<부품>/videos/*.mp4 → {부품: [영상 stem, ...]}. '_' 접두 폴더(_gearbox 등 보관용) 제외."""
    parts = {}
    for vp in glob.glob(PARTS + "/*/videos/*.mp4"):
        vp2 = vp.replace("\\", "/")
        part = vp2.split("/")[-3]
        if part.startswith("_"):
            continue
        stem = os.path.splitext(os.path.basename(vp2))[0]
        parts.setdefault(part, []).append(stem)
    return parts


def center_shots(video, autolabel):
    """영상 중앙부 3프레임 각각에 화면 중앙 포함점 → [[프레임, [[rx,ry,lab]]], ...]. 프레임 없으면 []."""
    fs = autolabel._frames(video)     # 캐시 없으면 자동 컷
    if not fs:
        return []
    n = len(fs)
    idxs = sorted(set(min(n - 1, max(0, int(n * r))) for r in REF_FRACS))
    return [[i, [list(pt) for pt in CENTER_PTS]] for i in idxs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="auto_baseline", help="results/parts/<세션>/ 이름")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--smoke", action="store_true", help="부품 3개·5epoch 빠른 검증")
    ap.add_argument("--no-train", action="store_true", help="라벨만 생성(학습 생략)")
    a = ap.parse_args()

    import sam2_autolabel as sa
    import autolabel

    parts = scan_parts()
    if not parts:
        print(f"[중단] 부품 폴더 없음: {PARTS}/*/videos/*.mp4"); sys.exit(1)
    part_names = sorted(parts)
    if a.smoke:
        part_names, a.epochs = part_names[:3], 5

    # 부품별 학습/테스트 테이크 결정
    train_takes, test_takes = [], []
    for part in part_names:
        tr, te = take_roles(parts[part])
        train_takes += tr
        if te and te not in test_takes:
            test_takes.append(te)
    print(f"[시작] BASE={BASE}\n  세션={a.session}  부품={len(part_names)}  학습테이크={len(train_takes)}  테스트테이크={len(test_takes)}  epochs={a.epochs}", flush=True)

    # 1) 학습 테이크마다 자동 탭 → SAM2 전파 → 세션 폴더 누적 (대시보드 워커 재사용)
    total = 0
    for k, v in enumerate(train_takes, 1):
        shots = center_shots(v, autolabel)
        if not shots:
            print(f"  [{k}/{len(train_takes)}] {v}: 프레임 없음, 건너뜀", flush=True); continue
        jid = f"auto{k}"
        sa.JOBS[jid] = {"stage": "start", "running": True, "error": None}
        sa._run_parts_label(jid, a.session, v, shots)     # 동기 실행
        st = sa.JOBS[jid]
        if st.get("error"):
            print(f"  [{k}/{len(train_takes)}] {v}: 오류 {st['error']}", flush=True)
        else:
            n = st.get("labels", 0); total += n
            print(f"  [{k}/{len(train_takes)}] {v}: 라벨 {n}장 / {st.get('frames', 0)}프레임", flush=True)
    print(f"[라벨 완료] 누적 {total}장 → results/parts/{a.session}/train", flush=True)

    if a.no_train:
        print("DONE(라벨만)"); return

    # 2) 클래스 통합 학습 → 부품별 테스트 테이크 검출 평가 (대시보드 멀티클래스 워커 재사용)
    jid = "mc"
    sa.JOBS[jid] = {"stage": "start", "running": True, "error": None}
    sa._run_multiclass(jid, a.session, a.epochs, test_takes)
    st = sa.JOBS[jid]
    if st.get("error"):
        print(f"[학습 오류] {st['error']}", flush=True); sys.exit(1)
    print(f"[학습 완료] 통합 {st.get('n_images')}장 / {st.get('n_classes')}클래스", flush=True)
    print(f"  가중치: {st.get('weights')}", flush=True)
    if st.get("miss"):
        print(f"  ⚠️ 미매핑(classes.txt 없음): {st['miss']}", flush=True)
    for e in st.get("eval", []):
        top = ", ".join(f"{c}({n})" for c, n in (e.get("top_classes") or []))
        print(f"  [test {e['src']}] 검출률 {e['rate']*100:.0f}% ({e['detected']}/{e['frames']})  "
              f"평균신뢰도 {e['mean_conf']}  주요:{top}", flush=True)
    print(f"  결과: results/parts/{a.session}/multiclass/  (meta.json·eval/)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
