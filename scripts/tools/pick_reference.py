# -*- coding: utf-8 -*-
"""pick_reference.py - 등록 영상 프레임을 눈으로 넘겨보며 직접 클릭해서 참조점을 지정하는 도구.

좌표를 코드에 때려박는 대신(=BELL처럼 배경에 잘못 찍히는 사고 방지), 실제 프레임을 보며
부품 위를 클릭해 참조를 만든다. 실제 배포의 '화면 탭'과 같은 방식.

출력: reference_shots.json  = [[프레임번호, [[x비율, y비율, label], ...]], ...]
       (label 1=부품(전경) / 0=제외(배경). point_ref_lib 의 SHOTS 형식과 동일)

사용:
    ./venv/Scripts/python.exe scripts/tools/pick_reference.py <프레임폴더 또는 영상.mp4> [출력.json]
예:
    ... pick_reference.py data/bell412/gearbox/register2/gearbox

조작:
    a / d           이전 / 다음 프레임
    , / .           10프레임 뒤로 / 앞으로
    마우스 좌클릭    부품 점(초록, label 1)
    마우스 우클릭    제외 점(빨강, label 0)  ← 옆의 방해물(드릴 등) 뺄 때
    z               이 프레임의 마지막 점 취소
    c               이 프레임 점 전부 지우기
    SPACE           이 프레임을 '참조 샷'으로 저장(현재 점들과 함께)
    x               이 프레임의 참조 샷 저장 취소
    ENTER           저장하고 종료 (json 기록)
    ESC             기록 없이 종료
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

DISP_W = 1280          # 화면 표시 최대 가로폭(원본이 더 크면 축소해서 보여줌)
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def imread_u(path):
    """비ASCII(한글) 경로도 안전하게 읽기: 바이트로 읽어 디코드."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_frames(src):
    """폴더면 jpg 정렬 목록(경로), 영상이면 프레임을 메모리로 읽어 반환. (frames, is_paths)"""
    src = Path(src)
    if src.is_dir():
        paths = sorted(src.glob("*.jpg"))
        if not paths:
            paths = sorted(src.glob("*.png"))
        return paths, True
    if src.suffix.lower() in VIDEO_EXT:
        cap = cv2.VideoCapture(str(src))
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        return frames, False
    raise SystemExit(f"입력을 못 읽음: {src} (폴더 또는 영상 파일이어야 함)")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    src = sys.argv[1]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("reference_shots.json")
    frames, is_paths = load_frames(src)
    n = len(frames)
    if n == 0:
        raise SystemExit("프레임이 없음")

    def get_frame(i):
        return imread_u(frames[i]) if is_paths else frames[i]

    shots = {}          # {프레임번호: [(rx, ry, label), ...]}
    idx = 0
    win = "pick_reference  (a/d 이동  ,/. ±10  L클릭=부품 R클릭=제외  SPACE 저장  ENTER 완료)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    state = {"scale": 1.0, "ow": 1, "oh": 1}

    def on_mouse(event, x, y, flags, param):
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        rx = (x / state["scale"]) / state["ow"]
        ry = (y / state["scale"]) / state["oh"]
        rx = min(max(rx, 0.0), 1.0); ry = min(max(ry, 0.0), 1.0)
        lab = 1 if event == cv2.EVENT_LBUTTONDOWN else 0
        shots.setdefault(idx, []).append((rx, ry, lab))

    cv2.setMouseCallback(win, on_mouse)

    while True:
        im = get_frame(idx)
        oh, ow = im.shape[:2]
        scale = DISP_W / ow if ow > DISP_W else 1.0
        disp = cv2.resize(im, (int(ow * scale), int(oh * scale))) if scale != 1.0 else im.copy()
        state.update(scale=scale, ow=ow, oh=oh)

        pts = shots.get(idx, [])
        for rx, ry, lab in pts:
            px, py = int(rx * ow * scale), int(ry * oh * scale)
            color = (0, 255, 0) if lab == 1 else (0, 0, 255)
            cv2.circle(disp, (px, py), 7, color, -1)
            cv2.circle(disp, (px, py), 8, (255, 255, 255), 1)

        is_saved = idx in state.get("saved_set", set())    # 이 프레임이 참조 샷으로 저장됐나
        npos = sum(1 for *_, l in pts if l == 1)
        nneg = len(pts) - npos
        bar = f"frame {idx+1}/{n}  |  shots {len(state.get('saved_set', set()))}  |  pts {len(pts)} (+{npos}/-{nneg})"
        if is_saved:
            bar += "   [SAVED]"
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(disp, bar, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        help1 = "L=part(green) R=exclude(red)  z=undo c=clear  SPACE=save x=unsave  ENTER=done ESC=cancel"
        cv2.rectangle(disp, (0, disp.shape[0] - 24), (disp.shape[1], disp.shape[0]), (0, 0, 0), -1)
        cv2.putText(disp, help1, (8, disp.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(win, disp)
        k = cv2.waitKeyEx(20)
        # 창 X버튼으로 닫으면 저장하고 종료 (안 그러면 루프가 창을 다시 띄움)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if k == -1:
            continue
        saved_set = state.setdefault("saved_set", set())
        if k in (ord('d'), 2555904):          # 다음
            idx = min(idx + 1, n - 1)
        elif k in (ord('a'), 2424832):         # 이전
            idx = max(idx - 1, 0)
        elif k == ord('.'):
            idx = min(idx + 10, n - 1)
        elif k == ord(','):
            idx = max(idx - 10, 0)
        elif k == ord('z'):                    # 점 취소
            if pts:
                pts.pop()
        elif k == ord('c'):                    # 이 프레임 점 지움
            shots[idx] = []
        elif k == 32:                          # SPACE = 샷 저장
            if pts:
                saved_set.add(idx)
        elif k == ord('x'):                    # 샷 저장 취소
            saved_set.discard(idx)
        elif k == 13:                          # ENTER = 완료
            break
        elif k == 27:                          # ESC = 취소
            cv2.destroyAllWindows()
            print("취소됨(기록 안 함)")
            return

    cv2.destroyAllWindows()
    saved_set = state.get("saved_set", set())
    spec = [[int(i), [[round(rx, 4), round(ry, 4), lab] for rx, ry, lab in shots[i]]]
            for i in sorted(saved_set) if shots.get(i)]
    out.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장 완료: {out}  (참조 샷 {len(spec)}개)")
    # point_ref_lib 에 붙여넣을 SHOTS 형식도 출력
    py = "[" + ", ".join(f"({i}, {[list(p) for p in pl]})" for i, pl in spec) + "]"
    print("SHOTS =", py)


if __name__ == "__main__":
    main()
