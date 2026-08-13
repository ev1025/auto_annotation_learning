# -*- coding: utf-8 -*-
"""scripts/verify/autolabel.py - 대시보드용 '포인트 참조 오토라벨' 백엔드 로직.

브라우저에서 프레임을 넘겨보며 부품을 클릭(참조점)하면, 그 참조로 영상 전체 프레임에
자동으로 라벨을 붙인다(방법 7 = SAM 후보 + DINOv2 유사도 매칭). 좌표를 코드에 박던 걸
사용자 클릭으로 대체 = 실제 배포의 '화면 탭'과 동일.

무거운 라벨 생성은 백그라운드 스레드로 돌리고 진행률을 폴링으로 노출한다.
SAM/DINOv2 는 한 번만 로드해 캐시. GPU 는 학습과 공유하므로 동시에 돌면 OOM 가능 → 예외 보고.
"""
import base64
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))  # 공용 config·experiments
import cv2
import numpy as np
import torch

import config
from experiments.point_ref_lib import (load_img, embed, candidates, nms, write_label,
                                             REF_TAU, SAM_CKPT)

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# 참조 소스 = data/ 하위 어디든(주제별 폴더 예: bell412/gearbox/videos/)의 영상. 새 영상 넣으면 자동 컷.
RESULTS = config.BASE_DIR / "results"
AUTOLABELS = RESULTS / "autolabels"       # results/autolabels/<부품>/{images/<stem>/, labels, boxs} (영속 = 라벨됨)
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
TARGET_FRAMES = 200                       # 영상당 목표 프레임 수(서브샘플, 과다·과부하 방지)

_MODELS = {}          # SAM/DINO 캐시
_LOCK = threading.Lock()
_CUT_LOCK = threading.Lock()   # 프레임 컷 동시 실행 방지
JOBS = {}             # job_id -> 진행상태 dict
_BUSY = {"on": False}  # GPU 라벨생성 동시 1건만

_VIDX = {"t": 0.0, "vids": None}   # _videos() 결과 짧은 TTL 캐시(프레임 요청마다 rglob 반복 방지)

def _videos(force=False):
    """data/ 하위 모든 영상(부품 폴더 재귀). '_' 로 시작하는 폴더(_frame_cache·_trash 등)는 제외.
    프레임 요청마다 호출되므로 3초 TTL로 결과를 재사용(새 영상은 몇 초 내 반영)."""
    now = time.time()
    if not force and _VIDX["vids"] is not None and now - _VIDX["t"] < 3.0:
        return _VIDX["vids"]
    if not config.DATA_DIR.exists():
        _VIDX.update(t=now, vids=[]); return []
    vids = []
    for ext in VIDEO_EXT:
        vids += config.DATA_DIR.rglob(f"*{ext}")   # data/ 어느 주제 폴더든 재귀 탐색
    base = config.DATA_DIR.resolve()
    # data/ 아래에서 '_' 로 시작하는 폴더(_frame_cache·보관용 _gearbox 등)는 제외
    out = sorted(p for p in vids if p.is_file()
                 and not any(part.startswith("_") for part in p.resolve().relative_to(base).parts[:-1]))
    _VIDX.update(t=now, vids=out)
    return out

def part_root_of(vp):
    """영상 Path → 그 부품 루트 폴더(data/bell412/<부품>). 'videos' 폴더 아래면 상위, 아니면 부모."""
    return vp.parent.parent if vp.parent.name == "videos" else vp.parent

def video_key(vp):
    """영상 Path → 부품경로 기반 유니크 키 = data/ 기준 상대경로(확장자 제거).
    예: data/bell412/gearbox/videos/test1.mp4 → 'bell412/gearbox/videos/test1'."""
    return vp.resolve().relative_to(config.DATA_DIR.resolve()).with_suffix("").as_posix()

def cache_dir_of(vp):
    """영상 Path → 그 영상의 프레임 저장소(results/autolabels/<부품>/images/<stem>).
    _frame_cache 폐지: autolabels/images 가 프레임(전 프레임) 저장소를 겸한다(SAM2 init_state·스크럽·학습이미지 공용)."""
    return AUTOLABELS / part_root_of(vp).name / "images" / vp.stem

def resolve_video(src):
    """src → 실제 영상 Path. 부품경로 기반 유니크 식별(같은 stem 이 다른 부품에 있어도 안 꼬임).
    허용 형태:
      - 부품경로 상대: 'bell412/<부품>/videos/<stem>' (확장자 유무 무관)  ← 프론트 기본
      - '<부품>/<stem>' 축약
      - bare stem (레거시, 전역 유니크 가정) — 하위호환
    """
    if not src:
        return None
    s = str(src).replace("\\", "/").strip("/")
    base = config.DATA_DIR.resolve()
    vids = _videos()
    # 1) 부품경로 상대(전체/확장자무시) 정확 매칭
    for vp in vids:
        rel = vp.resolve().relative_to(base).as_posix()          # bell412/<부품>/videos/<stem>.mp4
        if s == rel or s == rel.rsplit(".", 1)[0]:
            return vp
    # 2) '<부품>/<stem>' 축약 매칭
    if "/" in s:
        for vp in vids:
            if s == f"{part_root_of(vp).name}/{vp.stem}":
                return vp
    # 3) bare stem (레거시)
    return next((vp for vp in vids if vp.stem == s), None)

def _cached_frames(d):
    """디렉토리 d 안의 프레임(jpg) 정렬 목록. 없으면 빈 리스트."""
    return sorted(d.glob("*.jpg")) if d.exists() else []

def _frames_dir(vp):
    """이 영상의 프레임 저장소(results/autolabels/<부품>/images/<stem>). 없으면 새로 컷.

    _frame_cache(부품별·중앙) 폴백은 제거했다. 잔여 프레임을 이 저장소로 이사 완료(2026-08-13)했고,
    폴백을 남겨두면 '어디에 있느냐에 따라 동작이 갈리는' 상태가 계속된다(프레임 수 누락의 원인이었음).
    """
    part_cache = cache_dir_of(vp)
    if _cached_frames(part_cache):
        return part_cache
    _extract(vp, part_cache)
    return part_cache

def frames_dir_for(src):
    """src(부품경로/stem) → 프레임 디렉토리(SAM2 init_state 의 video_path 용). 없으면 컷."""
    vp = resolve_video(src)
    return _frames_dir(vp) if vp else None

def _estimate_count(vp):
    cap = cv2.VideoCapture(str(vp))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if total <= 0:
        return 0
    stride = max(1, total // TARGET_FRAMES)
    return (total + stride - 1) // stride

def _extract(vp, dest):
    """영상 -> 서브샘플 프레임 컷(부품별 캐시 dest 로). 이미 있으면 그대로 반환."""
    if _cached_frames(dest):
        return _cached_frames(dest)
    with _CUT_LOCK:
        if _cached_frames(dest):          # 락 안에서 재확인(중복 컷 방지)
            return _cached_frames(dest)
        dest.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(vp))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        stride = max(1, total // TARGET_FRAMES) if total else 1
        i = out = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % stride == 0:
                cv2.imwrite(str(dest / f"{out:05d}.jpg"), fr)
                out += 1
            i += 1
        cap.release()
    return _cached_frames(dest)

def _frames(src):
    """소스(부품경로/stem)의 프레임 목록. 부품별 캐시 우선, 레거시 중앙 캐시 폴백, 없으면 컷."""
    vp = resolve_video(src)
    if not vp:
        return []
    return _cached_frames(_frames_dir(vp))

def _source_info(vp):
    cached = _cached_frames(cache_dir_of(vp))   # 프레임 저장소는 한 곳뿐(레거시 _frame_cache 폴백 제거)
    return {"name": vp.stem, "key": video_key(vp),
            "count": len(cached) if cached else _estimate_count(vp),
            "ready": bool(cached)}

def list_folders():
    """영상이 있는 폴더별로 묶기. [{folder, label, videos:[{name,count,ready}]}].

    folder = data/ 기준 상대경로(예: bell412/gearbox/videos).
    label  = 사람이 볼 이름(끝이 videos면 상위 = 부품명, 예: bell412/gearbox).
    """
    groups = {}
    for vp in _videos():
        rel = vp.parent.relative_to(config.DATA_DIR).as_posix()
        groups.setdefault(rel, []).append(vp)
    out = []
    for rel in sorted(groups):
        parts = rel.split("/")
        label = "/".join(parts[:-1]) if parts[-1] == "videos" and len(parts) > 1 else rel
        out.append({"folder": rel, "label": label,
                    "videos": [_source_info(vp) for vp in sorted(groups[rel])]})
    return out

def prepare(src):
    """소스 프레임을 컷(없으면 생성)하고 프레임 수 반환. 소스 선택 시 호출."""
    fs = _frames(src)
    return {"name": src, "count": len(fs), "ready": True}

def _b64(img, w=None, q=80):
    if w and img.shape[1] > w:
        img = cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

def frame_jpeg(src, idx, w=960):
    """프레임 한 장을 JPEG 바이트로 (뷰어 표시용, 다운스케일)."""
    fs = _frames(src)
    if not fs or idx < 0 or idx >= len(fs):
        return None
    im = load_img(fs[idx])          # 비ASCII 경로 안전 로더(내부 cv2.imread) — load_img는 exp에서 처리
    if im is None:
        return None
    if im.shape[1] > w:
        im = cv2.resize(im, (w, int(im.shape[0] * w / im.shape[1])))
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()
