# -*- coding: utf-8 -*-
"""자동 탭(센터포인트) 멀티클래스 baseline: 사람 탭 없이 파이프라인을 끝까지 돌려 기준선을 만든다.

부품 촬영 영상은 '부품을 어두운 매트 위 중앙에 놓고 카메라가 도는' 구도라, 영상 중앙부
프레임에 화면 중앙점(0.5,0.5)을 찍으면 SAM2가 부품을 잡는다(육안 검증 완료). 이 자동 탭으로
train 영상마다 라벨을 만들고 → 34클래스 통합 학습 → test 영상 검출 평가까지 자동 수행.

목적: '자동 탭 baseline' 대비 '사람이 직접 탭'이 성능을 얼마나 올리는지 비교하는 기준선.
대시보드 백엔드(sam2_autolabel)의 워커 함수를 그대로 재사용 = 사람 탭과 동일한 경로.

사용: python scripts/experiments/auto_parts_baseline.py [--session auto_baseline] [--epochs 100]
      [--smoke] (train 3개·5epoch 빠른 검증) [--no-train] (라벨만)
윈도우 multiprocessing 대비 __main__ 가드 필수(내부 YOLO 학습)."""
import os
import sys
import argparse
from pathlib import Path

BASE = os.environ.get("XR_BASE") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, BASE + "/scripts")
sys.path.insert(0, BASE + "/scripts/verify")
SPLIT = BASE + "/data/bell412/parts/TRAIN_TEST_SPLIT.txt"

# 화면 중앙 포함점. 여러 참조 프레임(영상 중앙부)에 찍어 SAM2 전파 앵커로.
CENTER_PTS = [[0.5, 0.5, 1]]     # (rx, ry, lab=1)
REF_FRACS = [0.35, 0.5, 0.65]    # 영상 중앙부 3프레임을 참조샷으로


def read_split():
    """TRAIN_TEST_SPLIT.txt → (train 영상 stem 목록, test 영상 stem 목록). stem = 파일명(확장자 제거)."""
    train, test = [], []
    for line in open(SPLIT, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) != 2:
            continue
        tag, fn = p[0].strip().lower(), p[1].strip()
        if not fn.lower().endswith(".mp4"):
            continue
        stem = fn[:-4]
        (test if tag == "test" else train).append(stem)
    return train, test


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
    ap.add_argument("--smoke", action="store_true", help="train 3개·5epoch 빠른 검증")
    ap.add_argument("--no-train", action="store_true", help="라벨만 생성(학습 생략)")
    a = ap.parse_args()

    import sam2_autolabel as sa
    import autolabel

    train, test = read_split()
    if a.smoke:
        train, a.epochs = train[:3], 5
    print(f"[시작] BASE={BASE}\n  세션={a.session}  train={len(train)}  test={len(test)}  epochs={a.epochs}", flush=True)

    # 1) train 영상마다 자동 탭 → SAM2 전파 → 세션 폴더 누적 (대시보드 워커 재사용)
    total = 0
    for k, v in enumerate(train, 1):
        shots = center_shots(v, autolabel)
        if not shots:
            print(f"  [{k}/{len(train)}] {v}: 프레임 없음, 건너뜀", flush=True); continue
        jid = f"auto{k}"
        sa.JOBS[jid] = {"stage": "start", "running": True, "error": None}
        sa._run_parts_label(jid, a.session, v, shots)     # 동기 실행
        st = sa.JOBS[jid]
        if st.get("error"):
            print(f"  [{k}/{len(train)}] {v}: 오류 {st['error']}", flush=True)
        else:
            n = st.get("labels", 0); total += n
            print(f"  [{k}/{len(train)}] {v}: 라벨 {n}장 / {st.get('frames', 0)}프레임", flush=True)
    print(f"[라벨 완료] 누적 {total}장 → results/parts/{a.session}/train", flush=True)

    if a.no_train:
        print("DONE(라벨만)"); return

    # 2) 34클래스 통합 학습 → test 검출 평가 (대시보드 멀티클래스 워커 재사용)
    jid = "mc"
    sa.JOBS[jid] = {"stage": "start", "running": True, "error": None}
    sa._run_multiclass(jid, a.session, a.epochs, test)
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
