# -*- coding: utf-8 -*-
"""sam2_autolabel.py - SAM2 기반 단계형 오토라벨 백엔드.

흐름: 사람이 train 영상에 점 탭 → (1) 마스크 확인 → (2) 영상 전파로 라벨 생성 (영상마다 반복)
      → (3) train 영상 라벨을 합쳐 YOLO 학습 → test 영상(미학습)으로 평가.
      자동라벨(automask) 대신 SAM2 전파라 부품 전체를 일관되게 잡음.

산출물: results/<부품>/<실행시각>/ 한 폴더에 직접(실행마다 폴더 따로, 덮어쓰지 않음, 안에 또 시각 없음)
  tap/ 탭확인(점+마스크+박스) · train/ 학습라벨(images·labels) · train_box/ 박스표기 · meta.json
  model/  : runs/model/weights/best.pt · eval/<test영상>/ 예측이미지 · meta.json(검출률·신뢰도)
  map_eval/ : (별도 mAP 하네스 실행 시) GT 대비 실측 mAP·오버레이
  ※ test 정답 라벨이 없어 mAP 아님 = 검출률(프레임당 검출 여부)+신뢰도+육안 확인.

프레임 소스·캐시는 verify/autolabel.py 재사용. GPU 8GB: SAM2는 학습 전 반드시 해제.
"""
import base64
import gc
import json
import logging
import logging.handlers
import re
import shutil
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))  # 공용 config·experiments
sys.path.insert(0, str(Path(__file__).resolve().parent))                   # backend/autolearning (autolabel 등)
import cv2
import numpy as np
import torch

import config
import autolabel   # _frames, cache_dir_of (프레임 저장소 공용)


# ---- DB 동기화(선택적) ----
# 파일이 원본이고 DB 는 그 색인이다. 파일에 쓴 직후 아래 함수로 DB 를 최신화한다.
# DB 가 없거나 꺼져 있어도 라벨 생성·학습은 그대로 돌아야 하므로 import 실패는 무시한다.
def _db():
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # backend (db 패키지)
        from db import migrate_from_files as m   # noqa: PLC0415
        return m
    except Exception as e:   # noqa: BLE001
        print(f"[DB] 동기화 모듈 없음(파일만 기록): {type(e).__name__}: {e}", flush=True)
        return None


def _db_sync_part(video_or_part):
    """라벨 생성 직후 그 부품을 DB 에 동기화. video 경로/키를 주면 부품명을 뽑아낸다."""
    m = _db()
    if not m:
        return
    part = None
    try:
        vp = autolabel.resolve_video(video_or_part)
        part = autolabel.part_root_of(vp).name if vp else str(video_or_part).split("/")[-2:-1][0]
    except Exception:   # noqa: BLE001
        part = None
    if part:
        st = m.sync_part(part)
        if st:
            print(f"[DB] {part} 동기화: 프레임 {st['frames']}(신규 {st['frames_new']}) "
                  f"주석 {st['ann']}(참조샷 {st['ann_ref']})", flush=True)


def _db_sync_runs():
    """학습 완료·모델 적용·롤백 직후 학습이력/서비스모델을 DB 에 동기화."""
    m = _db()
    if not m:
        return
    st = m.sync_runs()
    if st:
        print(f"[DB] 학습이력 동기화: {st['runs']}건(신규 {st['runs_new']}), 서비스모델 {st['active']}", flush=True)


DEV = "cuda" if torch.cuda.is_available() else "cpu"
CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
CKPT = config.BASE_DIR / "models" / "sam2" / "sam2.1_hiera_base_plus.pt"
RESULTS = config.BASE_DIR / "results"
CUTS_DIR = config.BASE_DIR / "results" / "autolabels" / "_cuts"   # 누끼 캐시(라벨이 그대로면 재사용)
CUT_PER_CLASS = 60          # 부품당 누끼 상한. 인접 프레임은 사실상 같은 그림이라 더 뽑아도 다양성이 안 늘어난다
VAL_CONF = 0.4
# 비교 화면에서 한 프레임에 표시할 박스 상한(신뢰도 상위). 덜 학습된 모델이
# 300개까지 뱉어 사진이 박스로 덮이는 것을 막는다. 전체 개수는 따로 표시한다.
BOX_CAP = 12
EPOCHS = 100

_IMG = {}          # SAM2 이미지 예측기 캐시(마스크 확인용)
_LOCK = threading.Lock()
JOBS = {}
_BUSY = {"on": False}

def _rd(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)

def _b64(img, w=640, q=80):
    if w and img.shape[1] > w:
        img = cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

def _overlay(im, mask, color=(0, 200, 0), alpha=0.45):
    ov = im.copy(); ov[mask] = color
    return cv2.addWeighted(ov, alpha, im, 1 - alpha, 0)

def _ov_sizes(w, h):
    """오버레이(점·박스) 크기. 이미지의 긴 변에 비례시킨다.

    예전에는 세 군데가 서로 다른 기준을 썼다.
      - 마스크 미리보기: 가로 폭 기준(w/82, w/95)
      - 라벨 확인 이미지(boxs): 3px·반지름 10px 고정
      - 화면 탭 점(CSS): 8px 고정
    그래서 세로영상(1080x1920)과 가로영상(1920x1080), 미리보기와 확인 이미지에서
    점·박스 굵기가 제각각으로 보였다. 긴 변 기준 한 가지로 통일한다.
    """
    L = max(w, h)
    return (max(5, round(L / 110)),     # 점 반지름
            max(3, round(L / 220)),     # 박스 두께
            max(2, round(L / 550)))     # 점 흰 테두리


def _bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

# ==================== (1) 마스크 확인: 단일 프레임 ====================
def _img_predictor():
    if "p" not in _IMG:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        _IMG["p"] = SAM2ImagePredictor(build_sam2(CFG, str(CKPT), device=DEV))
    return _IMG["p"]

def mask_preview(src, frame_idx, points):
    """탭 점으로 그 프레임 SAM2 마스크 생성 → 포인트·마스크·박스를 한 이미지에 합쳐 반환 + 면적/박스. (학습 무관, 가벼움)"""
    fs = autolabel._frames(src)
    if not fs or frame_idx < 0 or frame_idx >= len(fs):
        return {"error": "프레임 범위 밖"}
    if not points:
        return {"error": "점이 없습니다"}
    # 화면 좌표는 0~1 비율이다. 범위를 벗어난 값이 그대로 SAM2 로 가면 이미지 밖을 가리킨다.
    points = [[min(max(float(x), 0.0), 1.0), min(max(float(y), 0.0), 1.0), int(l)] for x, y, l in points]
    im = _rd(fs[frame_idx]); h, w = im.shape[:2]
    pred = _img_predictor()
    with torch.inference_mode(), torch.autocast(DEV, dtype=torch.bfloat16):
        pred.set_image(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        pts = np.array([[rx * w, ry * h] for rx, ry, _ in points], dtype=np.float32)
        lbls = np.array([int(lab) for *_, lab in points], dtype=np.int32)
        masks, scores, _ = pred.predict(point_coords=pts, point_labels=lbls, multimask_output=True)
    m = masks[int(np.argmax(scores))].astype(bool)
    bb = _bbox(m)
    # 한 이미지에 전부: 마스크(초록 오버레이) + 박스(주황) + 포인트(초록=부품/빨강=제외)
    # 점·박스는 원본 해상도라 화면 표시크기에 맞춰 이미지 폭에 비례(왼쪽 CSS 점 8px 과 통일감)
    r, bt, ew = _ov_sizes(w, h)          # 점·박스·테두리 크기(긴 변 기준 공용 규칙)
    vis = _overlay(im, m)
    if bb:
        cv2.rectangle(vis, (bb[0], bb[1]), (bb[2], bb[3]), (0, 165, 255), bt)
    for rx, ry, lab in points:   # 마스크가 초록이라 부품점은 파랑(BGR), 제외점은 빨강
        cv2.circle(vis, (int(rx * w), int(ry * h)), r, (255, 60, 0) if lab else (0, 0, 255), -1)
        cv2.circle(vis, (int(rx * w), int(ry * h)), r, (255, 255, 255), ew)
    # 과포착 판정: 마스크가 프레임의 절반 넘게 덮으면 배경까지 먹은 것. 탭 점 간격은 신뢰불가
    # (사람은 부품 안쪽에 점 몇 개만 가까이 찍으므로) → 프레임 대비 마스크 '면적'으로만 판단.
    area_frac = float(m.sum()) / (w * h)
    if not bb:
        verdict = "empty"
    elif area_frac > 0.5:                               # 마스크가 프레임 절반 초과 = 배경까지 먹음(과포착)
        verdict = "over"
    else:
        verdict = "ok"
    gc.collect(); torch.cuda.empty_cache()
    return {"combo": _b64(vis), "area_frac": round(float(m.sum()) / (w * h), 4), "bbox": bb, "verdict": verdict}

def free_sam2():
    _IMG.clear()
    gc.collect(); torch.cuda.empty_cache()

# ==================== (2) 영상 전파 → 라벨 생성 ====================

# ==================== (3) 학습 ====================

def _propagate_into(video, shots, labels_dir, boxs_dir, progress=None):
    """영상 하나를 SAM2 전파 → labels_dir 에 <stem>_<frame>.txt(YOLO 라벨),
    boxs_dir 에 탭점+박스 확인 이미지. 프레임 이미지는 이미 autolabels/images/<stem>/ 캐시에 있으므로
    복사하지 않는다(_frame_cache 폐지·중복 제거). 라벨수·전체프레임수 반환.
    파일명 접두어는 항상 stem(슬래시 방지). 라벨 <stem>_<frame> ↔ 이미지 images/<stem>/<frame>.jpg 로 짝.

    progress(단계, 진행, 전체) 를 주면 진행 상황을 알려준다(화면의 tqdm 식 표시용).
    전파는 프레임 수만큼 돌기 때문에, 그동안 아무 숫자도 안 보이면 멈춘 것처럼 보였다."""
    from sam2.build_sam import build_sam2_video_predictor
    vp = autolabel.resolve_video(video)
    if not vp:
        raise RuntimeError(f"영상을 찾지 못함: {video}")
    stem = vp.stem
    fs = autolabel._frames(video)
    cache_dir = autolabel.frames_dir_for(video)        # = autolabels/<부품>/images/<stem> (전 프레임)
    predictor = build_sam2_video_predictor(CFG, str(CKPT), device=DEV)
    h, w = _rd(fs[0]).shape[:2]
    with torch.inference_mode(), torch.autocast(DEV, dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(cache_dir),
                                     offload_video_to_cpu=True, offload_state_to_cpu=True)
        for fi, pts in shots:
            p = np.array([[rx * w, ry * h] for rx, ry, _ in pts], dtype=np.float32)
            l = np.array([int(lab) for *_, lab in pts], dtype=np.int32)
            predictor.add_new_points_or_box(inference_state=state, frame_idx=int(fi), obj_id=1, points=p, labels=l)
        masks = {}
        for fidx, _ids, logits in predictor.propagate_in_video(state):
            m = logits[0].cpu().numpy(); masks[fidx] = (m[0] if m.ndim == 3 else m) > 0.0
            if progress:
                progress("propagate", len(masks), len(fs))     # 전파: 프레임 단위 진행
    del predictor, state; free_sam2()

    n = 0
    for i, p in enumerate(fs):
        if progress and (i % 10 == 0 or i == len(fs) - 1):
            progress("write", i + 1, len(fs))                  # 라벨 쓰기: 10장마다 갱신(과도한 갱신 방지)
        mk = masks.get(i)
        if mk is not None and mk.shape != (h, w):
            mk = cv2.resize(mk.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        bb = _bbox(mk) if mk is not None else None
        nm = f"{stem}_{p.stem}"   # 영상stem_번호 → 라벨/boxs 한 폴더에 통합해도 안 겹침
        if bb and (bb[2] - bb[0]) > 10 and (bb[3] - bb[1]) > 10:
            cx, cy = (bb[0] + bb[2]) / 2 / w, (bb[1] + bb[3]) / 2 / h
            bw, bh = (bb[2] - bb[0]) / w, (bb[3] - bb[1]) / h
            (labels_dir / f"{nm}.txt").write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")
            boxed = _rd(p).copy()                       # 박스 확인 이미지(라벨된 전 프레임)
            _r, _bt, _ew = _ov_sizes(w, h)
            cv2.rectangle(boxed, (bb[0], bb[1]), (bb[2], bb[3]), (0, 165, 255), _bt)
            cv2.putText(boxed, "part", (bb[0], max(bb[1] - 6, 16)), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.6, w / 1600), (0, 165, 255), _ew + 1)
            cv2.imwrite(str(boxs_dir / f"{nm}.jpg"), boxed)
            n += 1

    for fi, pts in shots:   # 탭 프레임: 점+마스크+박스 더 자세히(박스뷰 위에 덮어써 확인성↑)
        mk = masks.get(int(fi)); base = _rd(fs[int(fi)]); has = mk is not None and mk.shape == (h, w)
        vis = _overlay(base, mk) if has else base
        bb = _bbox(mk) if has else None
        _r, _bt, _ew = _ov_sizes(w, h)
        if bb:
            cv2.rectangle(vis, (bb[0], bb[1]), (bb[2], bb[3]), (0, 165, 255), _bt)
        for rx, ry, lab in pts:
            cv2.circle(vis, (int(rx * w), int(ry * h)), _r, (255, 60, 0) if lab else (0, 0, 255), -1)
            cv2.circle(vis, (int(rx * w), int(ry * h)), _r, (255, 255, 255), _ew)
        cv2.imwrite(str(boxs_dir / f"{stem}_{Path(fs[int(fi)]).stem}.jpg"), vis)
    return n, len(fs)

def job_status(jid):
    return JOBS.get(jid, {"error": "unknown job"})

# ==================== 멀티클래스 부품 라벨링 (장비별 위저드) ====================
# 여러 부품 영상을 한 세션 폴더(results/parts/<세션>/)에 누적. 파일명=<영상>_<프레임>.
# 부품 하나씩: 탭 → 라벨 생성(전파) → 다음 부품. 다 모으면 34클래스로 통합 학습.
# 학습 run = results/<시각>/ (model/·pred/·<부품>/·meta.json). 옛 results/parts/{_models,_active} 폐지.
# 개발자용 상세 로그(백그라운드 작업 추적): 회전 파일 + 타임스탬프 + 전체 트레이스백.
# 사용자용 친절 로그(job["log"], 프론트 터미널)와 분리 저장.
LOG_DIR = RESULTS / "_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("xr.parts")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _fh = logging.handlers.RotatingFileHandler(
        LOG_DIR / "pipeline.log", maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_fh)
    logger.propagate = False

# ==================== 부품별 영속 오토라벨 저장소 ====================
# 오토라벨 산출(입력마스크+학습라벨)을 실험결과(results/)와 분리해 부품 폴더 아래 영속 보관한다.
#   data/bell412/<부품>/autolabel/{images,labels,masks}  (파일명=<영상>_<프레임>)
# 프론트는 여전히 '세션' 개념으로 동작하므로, 이 영속 저장소 전체를 단일 세션(PERSIST)처럼 합성해 노출한다.
DATA_BELL = config.DATA_DIR / "bell412"
PERSIST = "autolabel"          # 프론트에 노출하는 단일 영속 세션 이름

def _video_stem(name):
    """파일명/stem에서 뒤 프레임번호 제거 → 영상 stem. 'train_00012'→'train', 'Gearbox_gearbox1_00007'→'Gearbox_gearbox1'."""
    return re.sub(r"_\d+$", "", Path(name).stem)

def _part_root_for_video(video):
    """영상(부품경로 키 또는 stem) → 그 부품 폴더(data/bell412/<부품>).
    resolve_video 로 부품경로 기반 유니크 해석(같은 stem 이 다른 부품에 있어도 안 꼬임).
    못 찾으면 안전 폴백(data/bell412/_unmapped)."""
    vp = autolabel.resolve_video(video)
    if vp is not None:
        return autolabel.part_root_of(vp)
    return DATA_BELL / "_unmapped"

def _autolabel_store(video):
    """영상(부품경로 키/stem) → 그 부품 영속 오토라벨 저장소 dict.
    results/autolabels/<부품>/{images/<stem>/, labels, boxs} (영속 = '라벨됨' 유지).
    images 는 프레임 저장소(전 프레임, autolabel.cache_dir_of 와 동일 트리)를 겸한다 = _frame_cache 폐지."""
    base = autolabel.AUTOLABELS / _part_root_for_video(video).name
    return {"root": base, "images": base / "images", "labels": base / "labels", "boxs": base / "boxs"}

def _store_for_part(part):
    """부품 폴더명 → 그 부품 영속 저장소 dict. (검수 썸네일/삭제에서 part 를 알 때 유니크)."""
    base = autolabel.AUTOLABELS / part
    return {"root": base, "images": base / "images", "labels": base / "labels", "boxs": base / "boxs"}

def _store_for_frame(name):
    """프레임 파일명(<영상>_<프레임>.jpg) → 그 부품 저장소."""
    return _autolabel_store(_video_stem(name))

def _persist_videos():
    """부품별 영속 저장소를 스캔 → {영상stem: {labels, frames}}. 프론트 labeledMap 재료(세션 무관 통합)."""
    videos = {}
    if autolabel.AUTOLABELS.exists():
        for lbl_dir in autolabel.AUTOLABELS.glob("*/labels"):
            for tp in lbl_dir.glob("*.txt"):
                v = _video_stem(tp.name)
                d = videos.setdefault(v, {"labels": 0, "frames": 0})
                d["labels"] += 1
                d["frames"] += 1
    return videos

def parts_sessions():
    """부품별 영속 저장소(data/bell412/<부품>/autolabel)를 단일 세션(PERSIST)으로 합성해 반환(프론트 호환).
    videos={영상stem:{labels,frames}}, trained=현재 서비스 모델이 이미 가진 부품의 영상."""
    videos = _persist_videos()
    trained = []
    sv = served_model()          # 현재 서비스 모델이 보유한 부품 → 그 부품의 영상은 '학습됨'으로 표시
    if sv:
        classes = set(sv.get("classes", []))
        try:
            sys.path.insert(0, str(config.BASE_DIR / "scripts" / "experiments"))
            import build_multiclass as bm
            trained = [v for v in videos if bm.stem_to_class(f"{v}_0") in classes]
        except Exception:
            trained = []
    return [{"session": PERSIST, "videos": videos, "n_videos": len(videos),
             "total_labels": sum(v["labels"] for v in videos.values()),
             "trained": trained, "updated": ""}]

def _clear_video(store, video):
    """재탭 반영: 그 영상(stem)의 기존 라벨·boxs 만 제거. 프레임 이미지(images/<stem> 캐시)는 재사용하므로 보존."""
    for g in (store["labels"].glob(f"{video}_*.txt"),
              store["boxs"].glob(f"{video}_*.jpg")):
        for f in list(g):
            f.unlink()

# ==================== 참조샷(탭 포인트) 영속 저장/로드 ====================
# 라벨은 서버(autolabel/labels)에 영속하지만, 참조샷(사람이 찍은 탭 포인트)은 여태 프론트
# localStorage 에만 있어 이관·타 브라우저에서 안 떴다. → 부품 폴더에 shots.json 으로 durable 보관.
#   data/bell412/<부품>/autolabel/shots.json = {"<영상stem>": {"<프레임>": [[rx,ry,lab], ...]}}
def _save_shots(video, shots):
    """이 영상의 참조샷(탭 포인트)을 그 부품 shots.json 에 병합 저장.
    shots=[[frame,[[rx,ry,lab],...]],...] (전파에 쓴 그 값). 그 영상(stem) 키만 새로 쓰고
    부품 안 다른 영상 키는 보존한다."""
    vp = autolabel.resolve_video(video)
    if not vp:
        return
    stem = vp.stem
    store = _autolabel_store(video)
    store["root"].mkdir(parents=True, exist_ok=True)
    fp = store["root"] / "shots.json"
    data = {}
    if fp.exists():
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    frames = {}
    for fi, pts in (shots or []):
        frames[str(int(fi))] = [[float(rx), float(ry), int(lab)] for rx, ry, lab in pts]
    data[stem] = frames
    fp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def delete_shot(video, frame):
    """참조샷 1개(그 프레임의 탭 포인트) 삭제 → shots.json 에 즉시 반영.
    화면에서만 지우면 새로고침 때 서버 값으로 되살아난다."""
    vp = autolabel.resolve_video(video)
    if not vp:
        return {"error": f"영상을 찾지 못함: {video}"}
    fp = _autolabel_store(video)["root"] / "shots.json"
    if not fp.exists():
        return {"ok": True, "removed": False}
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return {"error": "shots.json 을 읽을 수 없습니다"}
    frames = data.get(vp.stem) or {}
    removed = frames.pop(str(int(frame)), None) is not None
    removed_labels = 0
    if frames:
        data[vp.stem] = frames
    else:
        data.pop(vp.stem, None)      # 남은 참조샷이 없으면 그 영상 키도 지운다
        # 참조샷이 하나도 없으면 그 라벨은 근거가 없어진다 -> 이 영상 라벨·박스 미리보기 삭제.
        # (라벨은 참조샷 여러 개가 합쳐진 전파 결과라 참조샷 1개분만 골라 지울 수는 없다)
        store = _store_for_part(autolabel.part_root_of(vp).name)
        for g in (store["labels"].glob(f"{vp.stem}_*.txt"), store["boxs"].glob(f"{vp.stem}_*.jpg")):
            for f in list(g):
                f.unlink(missing_ok=True); removed_labels += 1
    fp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    _db_sync_part(video)             # DB 참조샷·라벨 색인도 맞춘다
    logger.info(f"[delete_shot] {vp.stem} #{frame} removed={removed} labels={removed_labels}")
    return {"ok": True, "removed": removed, "removed_labels": removed_labels}


def load_shots():
    """모든 부품 shots.json 취합 → {"<영상stem>": {"<프레임>": [[rx,ry,lab],...]}}. 프론트 참조샷 복원용."""
    out = {}
    if not autolabel.AUTOLABELS.exists():
        return out
    for fp in sorted(autolabel.AUTOLABELS.glob("*/shots.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            for video, frames in data.items():
                if isinstance(frames, dict):
                    out.setdefault(video, {}).update(frames)
    return out

def _run_parts_label(job_id, session, video, shots):
    """부품 영상 하나 → SAM2 전파 → 그 부품 영속 저장소(data/bell412/<부품>/autolabel)에 <stem>_<프레임>로 누적.
    video = 부품경로 키 또는 stem. 저장소는 부품경로 기반으로 유니크 해석, 파일명 접두어·반환 식별자는 stem."""
    j = JOBS[job_id]
    try:
        vp = autolabel.resolve_video(video)
        if not vp:
            raise RuntimeError(f"영상을 찾지 못함: {video}")
        stem = vp.stem
        store = _autolabel_store(video)
        pl, boxs = store["labels"], store["boxs"]
        for d in (pl, boxs, store["images"]):
            d.mkdir(parents=True, exist_ok=True)
        j.update(stage="propagate", note=f"{stem} 라벨 생성 중", prog_done=0, prog_total=0, prog_step="propagate")

        def _prog(step, done, total):   # 화면에 tqdm 처럼 개수를 보여준다
            j.update(prog_step=step, prog_done=done, prog_total=total)

        _clear_video(store, stem)                  # 재탭이면 이 영상분(stem_*)만 새로(라벨·boxs)
        n, tot = _propagate_into(video, shots, pl, boxs, progress=_prog)
        _save_shots(video, shots)                  # 참조샷(탭 포인트) 영속 저장 → 재접속·타 브라우저 복원
        _db_sync_part(video)                       # 파일에 쓴 직후 DB 색인 최신화(조회가 DB 라서 필수)
        taps = [_b64(_rd(p), w=520) for p in sorted(boxs.glob(f"{stem}_*.jpg"))][:6]
        j.update(stage="done", running=False, video=stem, labels=n, frames=tot,
                 session=PERSIST, taps=taps)
    except Exception as e:
        logger.exception("백그라운드 작업 오류")
        j.update(stage="error", error=f"{type(e).__name__}: {e}", running=False)
    finally:
        _BUSY["on"] = False
        gc.collect(); torch.cuda.empty_cache()

def start_parts_label(session, video, shots):
    with _LOCK:
        if _BUSY["on"]:
            return {"error": "이미 실행 중입니다. 끝난 뒤 다시 시도하세요."}
        if not video or not shots:
            return {"error": "이 부품에 점(참조샷)이 없습니다."}
        _BUSY["on"] = True
    jid = uuid.uuid4().hex[:8]
    _vp = autolabel.resolve_video(video)
    JOBS[jid] = {"stage": "start", "running": True, "error": None, "session": PERSIST,
                 "video": _vp.stem if _vp else video}
    threading.Thread(target=_run_parts_label, args=(jid, PERSIST, video, shots), daemon=True).start()
    return {"job": jid, "session": PERSIST}

def _run_parts_label_batch(job_id, session, items):
    """탭한 부품 여러 개를 한 번에: 각 영상 SAM2 전파 → 각 부품 영속 저장소 누적. items=[{video,shots},...]."""
    j = JOBS[job_id]
    try:
        results = []
        total = len(items)
        for k, it in enumerate(items, 1):
            video, shots = it.get("video"), it.get("shots")
            if not video or not shots:
                continue
            vp = autolabel.resolve_video(video)
            if not vp:
                continue
            stem = vp.stem
            store = _autolabel_store(video)
            pl, boxs = store["labels"], store["boxs"]
            for d in (pl, boxs, store["images"]):
                d.mkdir(parents=True, exist_ok=True)
            j.update(stage="propagate", note=f"{k}/{total} · {stem}", done=k - 1, total=total,
                     prog_step="propagate", prog_done=0, prog_total=0)

            def _prog(step, done, tot_, _stem=stem, _k=k):
                j.update(prog_step=step, prog_done=done, prog_total=tot_, note=f"{_k}/{total} · {_stem}")

            _clear_video(store, stem)
            n, tot = _propagate_into(video, shots, pl, boxs, progress=_prog)
            _save_shots(video, shots)              # 참조샷 영속 저장(부품별 shots.json)
            _db_sync_part(video)                   # 부품마다 DB 색인 최신화
            results.append({"video": stem, "labels": n, "frames": tot})
        j.update(stage="done", running=False, session=PERSIST, done=total, total=total, results=results)
    except Exception as e:
        logger.exception("백그라운드 작업 오류")
        j.update(stage="error", error=f"{type(e).__name__}: {e}", running=False)
    finally:
        _BUSY["on"] = False
        gc.collect(); torch.cuda.empty_cache()

def start_parts_label_batch(session, items):
    with _LOCK:
        if _BUSY["on"]:
            return {"error": "이미 실행 중입니다. 끝난 뒤 다시 시도하세요."}
        items = [it for it in (items or []) if it.get("video") and it.get("shots")]
        if not items:
            return {"error": "탭한 부품이 없습니다."}
        _BUSY["on"] = True
    jid = uuid.uuid4().hex[:8]
    JOBS[jid] = {"stage": "start", "running": True, "error": None, "session": PERSIST}
    threading.Thread(target=_run_parts_label_batch, args=(jid, PERSIST, items), daemon=True).start()
    return {"job": jid, "session": PERSIST}

OCCL_P = 0.5              # 합성 객체 중 절반은 판넬로 부분 가림(7/30 레시피)
OCCL_FRAC = (0.35, 0.8)   # 가림 판넬 폭 = 부품 박스 폭의 35~80%


def _synth_augment(oi, ol, logln, n_syn=400, cancelled=None):
    """멀티클래스 학습셋(oi/ol)의 객체를 SAM2로 누끼 → 실배경에 copy-paste 합성 n_syn장 추가.
    배경 과적합을 줄이는 opt-in 증강. 라벨의 원래 클래스 idx를 유지한다. (run_augtrain 로직을 멀티클래스로 일반화)"""
    import random
    random.seed(0)
    # 합성에 쓰는 자원은 한 폴더에 모아 둔다.
    #   data/bell412/_synth/backgrounds/  누끼를 붙일 바닥(공장·격납고)
    #   data/bell412/_synth/occluders/    부품을 부분적으로 가릴 누끼(금속판넬 등, 알파 PNG)
    synth_dir = config.BASE_DIR / "data" / "bell412" / "_synth"
    bgdir = synth_dir / "backgrounds"
    bgs = [str(p) for p in bgdir.rglob("*.jpg")] if bgdir.exists() else []
    ocdir = synth_dir / "occluders"
    occs = []
    for op in sorted(ocdir.glob("*.png")) if ocdir.is_dir() else []:
        oi_img = cv2.imread(str(op), cv2.IMREAD_UNCHANGED)
        if oi_img is not None and oi_img.ndim == 3 and oi_img.shape[2] == 4:
            occs.append(oi_img)
    if not bgs:
        logln("배경 이미지 없음(data/bell412/_synth/backgrounds) → 배경 합성 증강 생략", "info"); return 0

    def _prep_bg(path):
        bg = _rd(path); h, w = bg.shape[:2]
        cw, ch = int(w * random.uniform(0.7, 1.0)), int(h * random.uniform(0.7, 1.0))
        x, y = random.randint(0, w - cw), random.randint(0, h - ch); bg = bg[y:y + ch, x:x + cw]
        s = 960 / max(bg.shape[:2])
        if s < 1:
            bg = cv2.resize(bg, (int(bg.shape[1] * s), int(bg.shape[0] * s)))
        if random.random() < 0.5:
            bg = cv2.flip(bg, 1)
        return np.clip(bg.astype(np.float32) * random.uniform(0.7, 1.25), 0, 255).astype(np.uint8)

    def _paste(bg, fg, x, y):
        H, W = bg.shape[:2]; h, w = fg.shape[:2]
        x0, y0 = max(0, x), max(0, y); x1, y1 = min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return None
        sub = fg[y0 - y:y1 - y, x0 - x:x1 - x]; al = sub[:, :, 3:4].astype(np.float32) / 255
        bg[y0:y1, x0:x1] = (sub[:, :, :3].astype(np.float32) * al + bg[y0:y1, x0:x1].astype(np.float32) * (1 - al)).astype(np.uint8)
        ys, xs = np.where(sub[:, :, 3] > 15)
        return None if len(xs) == 0 else (x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()), y0 + int(ys.max()))

    # 1) 누끼: 객체를 SAM2 마스크로 오려 RGBA 로 보관.
    #    비용이 큰 단계라 두 가지로 줄인다.
    #      (a) 부품당 CUT_PER_CLASS 장만 고르게 뽑는다. 인접 프레임은 거의 같은 그림이라
    #          495장을 다 돌려도 다양성이 늘지 않고 시간만 든다(실측 495장 = 2분 13초).
    #      (b) 잘라낸 조각을 캐시한다. 라벨이 그대로면 다음 학습부터 SAM2 를 안 돈다.
    CUTS_DIR.mkdir(parents=True, exist_ok=True)
    by_cls_src = {}                        # 클래스 -> [(이미지, 라벨)]
    for ip in sorted(oi.glob("*.jpg")):
        if ip.stem.startswith("syn_"):
            continue
        lp = ol / f"{ip.stem}.txt"
        if not lp.exists():
            continue
        head = lp.read_text(encoding="utf-8").split()
        if len(head) < 5:
            continue
        by_cls_src.setdefault(int(head[0]), []).append((ip, lp))
    picked = []
    for ci, items in sorted(by_cls_src.items()):
        step = max(1, len(items) // CUT_PER_CLASS)
        picked += [(ip, lp, ci) for ip, lp in items[::step][:CUT_PER_CLASS]]
    if not picked:
        logln("배경 합성 증강 생략(대상 없음)", "info"); return 0

    logln("배경 합성 증강 중...", "info")
    logger.info(f"[synth] 누끼 대상 {len(picked)}장 선별(부품당 최대 {CUT_PER_CLASS})")
    cuts, hit, miss = [], 0, 0
    pred = None
    for _k, (ip, lp, ci) in enumerate(picked):
        if cancelled and _k % 20 == 0 and cancelled():
            raise _Cancelled
        cache = CUTS_DIR / f"{ip.stem}.png"
        if cache.exists() and cache.stat().st_mtime >= lp.stat().st_mtime:
            rgba = cv2.imread(str(cache), cv2.IMREAD_UNCHANGED)
            if rgba is not None and rgba.ndim == 3 and rgba.shape[2] == 4:
                cuts.append((rgba, ci)); hit += 1; continue
        if pred is None:                   # 캐시 미스가 하나라도 있을 때만 SAM2 를 올린다
            pred = _img_predictor()
        im = _rd(ip); h, w = im.shape[:2]
        try:
            pred.set_image(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        except Exception:
            continue
        for line in lp.read_text(encoding="utf-8").splitlines():
            f = line.split()
            if len(f) != 5:
                continue
            cx, cy, bw, bh = map(float, f[1:])
            box = np.array([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], np.float32)
            try:
                with torch.inference_mode(), torch.autocast(DEV, dtype=torch.bfloat16):
                    masks, _, _ = pred.predict(box=box, multimask_output=False)
                m = masks[0]; m = (m[0] if m.ndim == 3 else m) > 0.0
            except Exception:
                continue
            bb = _bbox(m)
            if not bb or bb[2] - bb[0] < 12 or bb[3] - bb[1] < 12:
                continue
            xa, ya, xb, yb = bb
            a = cv2.GaussianBlur(m[ya:yb + 1, xa:xb + 1].astype(np.uint8) * 255, (3, 3), 0)
            rgba = np.dstack([im[ya:yb + 1, xa:xb + 1], a])
            cuts.append((rgba, int(f[0]))); miss += 1
            cv2.imwrite(str(cache), rgba)   # 다음 학습에서 재사용
    if pred is not None:
        free_sam2()
    if not cuts:
        logln("배경 합성 증강 생략(대상 없음)", "info"); return 0
    # 클래스별로 나눠 담는다. 한 통에서 뽑으면 라벨 많은 부품이 합성까지 독점한다.
    by_cls = {}
    for rgba, ci in cuts:
        by_cls.setdefault(ci, []).append(rgba)
    logger.info(f"[synth] 누끼 완료: 캐시 재사용 {hit} · 신규 {miss} · 총 {len(cuts)}개({len(by_cls)}종)")

    # 2) 합성: 배경에 1~2개 랜덤 붙여넣기(회전·스케일), 멀티클래스 라벨 기록
    made = 0
    for k in range(n_syn):
        if cancelled and k % 50 == 0 and cancelled():
            raise _Cancelled
        bg = _prep_bg(random.choice(bgs)); H, W = bg.shape[:2]; labels = []
        for _ in range(1 if random.random() > 0.25 else 2):
            ci = random.choice(list(by_cls))          # 클래스 먼저 균등하게, 그 안에서 객체
            rgba = random.choice(by_cls[ci])
            hh, ww = rgba.shape[:2]
            M = cv2.getRotationMatrix2D((ww / 2, hh / 2), random.uniform(-20, 20), 1.0)
            c, s = abs(M[0, 0]), abs(M[0, 1]); nw, nh = int(hh * s + ww * c), int(hh * c + ww * s)
            M[0, 2] += (nw - ww) / 2; M[1, 2] += (nh - hh) / 2
            cut = cv2.warpAffine(rgba, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
            tw = int(W * random.uniform(0.18, 0.42)); sc = tw / cut.shape[1]
            cut = cv2.resize(cut, (max(1, int(cut.shape[1] * sc)), max(1, int(cut.shape[0] * sc))))
            ch2, cw2 = cut.shape[:2]
            if cw2 >= W or ch2 >= H:
                continue
            bb = _paste(bg, cut, random.randint(0, W - cw2), random.randint(0, H - ch2))
            if not bb:
                continue
            # 부분 가림: 기체에 장착된 부품은 배관·판넬에 일부가 가려져 보인다. 판넬 누끼를 부품 위에
            # 얹고 라벨 박스는 그대로 둬서 "일부만 보이는 부품"을 배우게 한다(7/30 실험의 주 지렛대).
            if occs and random.random() < OCCL_P:
                oc = occs[random.randrange(len(occs))]
                bw2, bh2 = bb[2] - bb[0], bb[3] - bb[1]
                ow = max(8, int(bw2 * random.uniform(*OCCL_FRAC)))
                osc = ow / oc.shape[1]
                oc2 = cv2.resize(oc, (ow, max(8, int(oc.shape[0] * osc))))
                if random.random() < 0.5:
                    oc2 = cv2.flip(oc2, 1)
                ox = bb[0] + random.randint(-ow // 3, max(0, bw2 - ow // 2))
                oy = bb[1] + random.randint(-oc2.shape[0] // 3, max(0, bh2 - oc2.shape[0] // 2))
                _paste(bg, oc2, ox, oy)      # 라벨은 갱신하지 않는다
            cx = (bb[0] + bb[2]) / 2 / W; cy = (bb[1] + bb[3]) / 2 / H
            labels.append(f"{ci} {cx:.6f} {cy:.6f} {(bb[2] - bb[0]) / W:.6f} {(bb[3] - bb[1]) / H:.6f}")
        if labels:
            cv2.imwrite(str(oi / f"syn_{k:05d}.jpg"), bg)
            (ol / f"syn_{k:05d}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
            made += 1
    logln(f"배경 합성 증강 완료 · {made}장 추가"
          + (f" (판넬 가림 {len(occs)}종 적용)" if occs else " (가림 누끼 없음)"), "ok")
    return made

# ---- 내부 평가용 영상 ----
# data/bell412/<부품>/eval/*.mp4 : 성능 확인용 미학습 영상. 부품 폴더 안에 두어 그 부품의 자료가
# 한자리에 모이게 한다. autolabel._videos() 가 'eval' 폴더를 제외하므로 등록·목록·학습 화면에는
# 보이지 않는다. 프레임 저장소 규칙은 부품 영상과 같다(results/autolabels/<부품>/images/<stem>).
def eval_videos(part):
    d = config.DATA_DIR / "bell412" / part / "eval"
    return sorted(f for f in d.glob("*") if f.suffix.lower() in autolabel.VIDEO_EXT) if d.is_dir() else []


def _csv_epochs(runs_dir):
    """가장 최근 results.csv 를 [{epoch,box,cls,dfl,map50,map5095}, ...] 로 읽는다."""
    files = sorted(Path(runs_dir).glob("*/results.csv"), key=lambda f: f.stat().st_mtime)
    if not files:
        return []
    txt = files[-1].read_text(encoding="utf-8", errors="replace").strip().splitlines()
    if len(txt) < 2:
        return []
    head = [h.strip() for h in txt[0].split(",")]

    def pick(row, *names):
        for n in names:
            if n in head:
                v = row[head.index(n)].strip()
                if v:
                    try:
                        return round(float(v), 4)
                    except ValueError:
                        return None
        return None

    rows = []
    for ln in txt[1:]:
        row = ln.split(",")
        if len(row) < len(head):
            continue
        ep = pick(row, "epoch")
        if ep is None:
            continue
        rows.append({"epoch": int(ep),
                     "box": pick(row, "train/box_loss"), "cls": pick(row, "train/cls_loss"),
                     "dfl": pick(row, "train/dfl_loss"),
                     "map50": pick(row, "metrics/mAP50(B)", "metrics/mAP50"),
                     "map5095": pick(row, "metrics/mAP50-95(B)", "metrics/mAP50-95"),
                     # 화면 곡선은 mAP 대신 이 둘을 쓴다.
                     #   r(재현율)  = 있는 부품을 실제로 인지한 비율 = "인지했다"
                     #   p(정밀도)  = 인지했다고 한 것 중 맞은 비율 = "헛인지 안 했다"
                     # mAP 는 이 학습이 train 과 val 에 같은 폴더를 쓰므로 0.99 로 포화돼 판단에 못 쓴다.
                     "p": pick(row, "metrics/precision(B)", "metrics/precision"),
                     "r": pick(row, "metrics/recall(B)", "metrics/recall")})
        # 탐지에는 '정확도(accuracy)' 라는 지표가 없다(맞힌 칸/전체 칸을 셀 수 없어서).
        # 대신 Precision·Recall 을 하나로 합친 F1 을 함께 그린다. F1 = 2PR/(P+R)
        _p, _r = rows[-1]["p"], rows[-1]["r"]
        rows[-1]["f1"] = round(2 * _p * _r / (_p + _r), 4) if (_p and _r) else None
    return rows


class _Cancelled(Exception):
    """중단 요청. ultralytics 의 trainer.stop 은 에폭 경계에서만 확인되므로(100 에폭이면 몇 분씩
    안 멈춘다) 배치 콜백에서 이 예외를 올려 학습 루프를 즉시 빠져나온다."""


def _run_multiclass(job_id, session, epochs, only_classes=None, augment=False, replay_served=False):
    """세션에 누적된 per-part 라벨(class 0) → 영상명→부품→클래스 remap → YOLO 학습 → 검출 평가.
    only_classes 지정 시 그 클래스만 학습. augment=True면 학습 전 배경 합성 증강 추가."""
    j = JOBS[job_id]

    # 로그는 세 곳으로 나간다.
    #   1) job["log"]                  프론트 터미널(폴링)
    #   2) results/_logs/pipeline.log  모든 작업이 섞이는 회전 로그(개발자용, 최근 것만)
    #   3) results/<시각>/train.log    이 학습만의 기록(모델·meta.json 과 같은 폴더에 남는다)
    # 3번이 없으면 학습이 끝난 뒤 "그때 무슨 일이 있었나"를 run 폴더만 보고는 알 수 없다.
    _runlog = {"path": None}

    def logln(msg, level="info"):
        j.setdefault("log", []).append({"level": level, "msg": msg})
        logger.log(logging.ERROR if level == "err" else logging.INFO, f"[multiclass:{job_id}] {msg}")
        p = _runlog["path"]
        if p:
            try:
                with open(p, "a", encoding="utf-8") as f:
                    f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} [{level}] {msg}\n")
            except Exception:   # noqa: BLE001 - 로그 기록 실패가 학습을 멈추면 안 된다
                pass

    try:
        runid = datetime.now().strftime("%y%m%d_%H%M%S")   # 이 학습 run = results/<시각>/
        session = runid
        sess = RESULTS / runid
        sess.mkdir(parents=True, exist_ok=True)
        _runlog["path"] = sess / "train.log"
        logln("학습 시작")
        logger.info(f"[multiclass:{job_id}] run {runid} 시작")   # run·job 은 서버 로그에만
        sys.path.insert(0, str(config.BASE_DIR / "scripts" / "experiments"))
        import build_multiclass as bm
        names, name2idx = bm.load_classes()
        # 학습 입력 = 부품별 영속 오토라벨(results/autolabels/<부품>/labels/<stem>_<frame>.txt).
        # 이미지는 results/autolabels/<부품>/images/<stem>/<frame>.jpg (프레임캐시 겸용).
        store_lbls = sorted(p for d in sorted(autolabel.AUTOLABELS.glob("*/labels")) for p in d.glob("*.txt"))
        if not store_lbls:
            raise RuntimeError("오토라벨 라벨이 없습니다. 부품을 먼저 탭·라벨 생성하세요.")

        def _part_of_label(lp):     # results/autolabels/<부품>/labels/<file> → 부품폴더명
            return lp.parent.parent.name

        def _img_of_label(lp):      # 라벨 <video>_<frame> → 이미지 images/<video>/<frame>.jpg
            ls = lp.stem
            video = _video_stem(ls)
            frame = ls[len(video) + 1:]
            return lp.parent.parent / "images" / video / f"{frame}.jpg"

        # 1) 클래스 매핑 통합 (학습 입력 스테이징 = results/<시각>/train)
        j.update(stage="build", note="라벨 클래스 매핑 통합")
        logln("라벨 클래스 매핑 통합 중...", "info")
        oi, ol = sess / "train" / "images", sess / "train" / "labels"
        oi.mkdir(parents=True, exist_ok=True); ol.mkdir(parents=True, exist_ok=True)
        sel = set(only_classes) if only_classes else None    # 선택한 클래스만 학습(없으면 전체)
        # 고른 것만 학습한다. 화면에서 부품을 고르는 의미가 사라지면 안 된다(사용자 지시).
        # 학습은 매번 from-scratch 라 고르지 않은 부품은 이 모델에서 사라진다. 그래서 경고만 남기고,
        # 되넣기(리플레이)는 replay=True 로 요청할 때만 한다. 적용 전에 평가 화면이 비교로 잡는다.
        replay = set()
        if sel is not None:
            sv = served_model()
            gone = (set(sv.get("classes", [])) - sel) if sv else set()
            if gone and replay_served:
                replay = gone
                sel = sel | replay
                logln(f"망각 방지: 기존 서비스 모델 부품 {len(replay)}종을 학습에 자동 포함", "info")
            elif gone:
                # 화면에는 적지 않는다(적용 화면의 비교가 같은 사실을 수치로 보여준다)
                logger.info(f"[multiclass:{job_id}] 선택 {len(sel)}종만 학습 · 서비스 모델의 {len(gone)}종 제외")
        per, miss = {}, {}
        input_cnt = 0        # 선택한 클래스의 자동생성 라벨(입력 데이터) 총수
        drop_noimg = 0       # 대응 이미지 없음으로 산입 제외
        drop_empty = 0       # 빈 라벨(유효 박스 없음)로 산입 제외
        lbls = store_lbls
        if sel is not None:   # 선택한 부품만 산입 대상 → 진행바 분모도 선택분만
            lbls = [p for p in store_lbls if re.sub(r"\s+", "_", _part_of_label(p).lower()) in sel]
        n_total = len(lbls)
        j.update(ingest_total=n_total, ingest_done=0)         # 산입 진행바(선택 부품 기준)
        # 1차 스캔: 부품별 라벨 수집. 부품 판정 = 라벨이 놓인 부품 폴더명(파일명 아님 → 안 꼬임).
        # 전역 목록(part_codes.json)을 그대로 쓰면 미학습 부품 뉴런이 남아 오검출 → 실제 학습 부품만 0..N-1 재색인.
        staged = []          # [(이미지경로, stem, 부품클래스, [박스4값,...]), ...]
        for k, lp in enumerate(lbls, 1):
            if k % 20 == 0 or k == n_total:
                j.update(ingest_done=k)                       # 산입 진행 실시간 갱신
                if j.get("cancel"):                           # 학습 전(라벨 산입) 단계에서도 중단이 먹게
                    raise _Cancelled
            part = _part_of_label(lp)
            cls = re.sub(r"\s+", "_", part.lower())
            if sel is not None and cls not in sel:
                continue
            input_cnt += 1
            if cls not in name2idx:
                miss[cls] = miss.get(cls, 0) + 1; continue   # 코드표 미등록 부품 → 산입 실패
            ip = _img_of_label(lp)
            if not ip.exists():
                drop_noimg += 1; continue                    # 대응 프레임 이미지 없음 → 산입 실패
            boxes = [p[1:5] for p in (l.split() for l in lp.read_text(encoding="utf-8").splitlines()) if len(p) == 5]
            if not boxes:
                drop_empty += 1; continue                    # 빈 라벨 → 산입 실패
            staged.append((ip, lp.stem, cls, boxes))
            per[cls] = per.get(cls, 0) + 1
        # 압축 클래스공간: 실제 학습되는 부품만 정렬해 0..N-1 로 재색인(모델 nc = 학습 부품 수)
        train_names = sorted(per.keys())
        cidx = {c: i for i, c in enumerate(train_names)}
        for _k, (ip, stem, cls, boxes) in enumerate(staged):   # 2차: 재색인 인덱스로 이미지·라벨 기록(<stem>=<video>_<frame>)
            if _k % 200 == 0 and j.get("cancel"):
                raise _Cancelled
            shutil.copy(ip, oi / f"{stem}.jpg")
            (ol / f"{stem}.txt").write_text(
                "\n".join(f"{cidx[cls]} {b[0]} {b[1]} {b[2]} {b[3]}" for b in boxes) + "\n",
                encoding="utf-8")
        # 배경(네거티브) 산입: 부품이 없는 사진 = 라벨 파일 없이 이미지만 넣는다.
        # YOLO 는 라벨 없는 이미지를 '여기엔 아무 것도 없다'로 배운다(background). 클래스는 만들지 않는다.
        # 없으면 모델이 처음 보는 장면에서 36개 중 가장 비슷한 걸 골라 버린다(사무실 시리얼 사진을
        # gearbox 0.85 로 잡던 사고). 권장 비율은 학습셋의 10% 안쪽.
        n_bg = 0
        for _k, bp in enumerate(sorted((config.DATA_DIR / "bell412" / "_negatives" / "images").glob("*.jpg"))):
            if _k % 100 == 0 and j.get("cancel"):
                raise _Cancelled
            shutil.copy(bp, oi / f"bg_{bp.stem}.jpg")         # 라벨 파일을 쓰지 않는다 = 배경
            n_bg += 1
        if n_bg:
            logln(f"배경 사진 {n_bg}장 산입", "info")

        names = train_names        # 이하 클래스 표기·인덱스는 모두 압축공간 기준(model.names·검출평가와 일치)
        n_img = sum(per.values())
        if replay:
            missing = sorted(replay - set(per))
            if missing:
                logln(f"주의: 기존 부품 {len(missing)}종은 이 세션에 라벨이 없어 재학습에서 제외됨(별도 세션 라벨 필요) → {missing[:8]}", "err")
        if n_img < 5:
            raise RuntimeError(f"통합 라벨 부족({n_img}). 부품을 더 탭하세요. 미매핑: {miss}")
        # 학습률(산입률) = 정상 산입 ÷ 입력 × 100
        miss_total = sum(miss.values())
        learn_rate = round(n_img / input_cnt * 100, 1) if input_cnt else 0.0
        j.update(input_total=input_cnt, ingested=n_img, learn_rate=learn_rate)
        # 학습률 로그는 학습 완료 후에 출력(산입 + 1 Epoch 완료가 확정돼야 학습률로 성립)
        yml = sess / "train" / "data.yaml"
        nb = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
        d = oi.resolve().as_posix()
        yml.write_text(f"path: {(sess / 'train').resolve().as_posix()}\ntrain:\n  - {d}\nval:\n  - {d}\nnames:\n{nb}\n",
                       encoding="utf-8")

        # 1.5) (opt-in) 배경 합성 증강: 누끼 → 실배경 copy-paste 로 학습셋 보강(배경 과적합 완화)
        n_aug = 0
        if augment:
            j.update(note="배경 합성 증강 생성 중")
            n_aug = _synth_augment(oi, ol, logln, cancelled=lambda: bool(j.get("cancel"))) or 0

        # 2) 학습
        from ultralytics import YOLO
        j.update(stage="train", note=f"{n_img}장 / {len(per)}클래스", n_images=n_img,
                 n_augmented=n_aug, n_classes=len(per), epoch=0, total_epochs=epochs)
        logln(f"YOLO 학습 시작 (epochs={epochs}, imgsz=640, batch=8, device=0)", "info")
        model_dir = sess / "model"
        model = YOLO(config.PRETRAINED)
        bpe = max(1, (n_img + 7) // 8)      # 에폭당 배치 수(batch=8)
        total_batches = bpe * epochs
        _bcount = {"n": 0}

        def _on_batch(trainer):   # 배치마다 진행바 갱신 + 중단 요청 확인
            _bcount["n"] += 1
            j.update(train_frac=min(1.0, _bcount["n"] / total_batches))
            if j.get("cancel"):
                trainer.stop = True      # 정상 종료 경로(에폭 경계)
                raise _Cancelled         # 그 전에 지금 멈춘다

        # 에폭 진행은 콜백(on_fit_epoch_end) 대신 results.csv 를 읽어 만든다.
        # 이유: ultralytics 8.4.121 에서 이 콜백이 호출되지 않아 화면이 계속 Epoch 0 이었다
        # (배치 콜백은 정상). 같은 호출에서 name="model" 도 무시돼 run 폴더가 모델명으로 생겼다.
        # 버전이 또 바뀌어도 results.csv 는 항상 쓰이므로 파일을 진실로 삼는다.
        def _watch_csv(runs_dir, stop_evt):
            """학습 중 3초마다 results.csv 를 읽어 job(에폭·곡선·로그)을 갱신한다."""
            seen = 0
            while not stop_evt.wait(3):
                try:
                    rows = _csv_epochs(runs_dir)
                except Exception:   # noqa: BLE001 - 표시용이라 실패해도 학습은 계속
                    continue
                if len(rows) <= seen:
                    continue
                for r in rows[seen:]:
                    loss = None
                    if None not in (r["box"], r["cls"], r["dfl"]):
                        loss = round(r["box"] + r["cls"] + r["dfl"], 4)
                    j.update(epoch=r["epoch"], total_epochs=epochs, cur_loss=loss,
                             cur_map=r["map50"], cur_map5095=r["map5095"])
                    j.setdefault("curve", []).append(r)
                    seg = [f"Epoch {r['epoch']}/{epochs}"]
                    if loss is not None:
                        seg.append(f"loss {loss:.3f}")
                    if r["r"] is not None:
                        seg.append(f"Recall {r['r'] * 100:.1f}%")
                    if r["p"] is not None:
                        seg.append(f"Precision {r['p'] * 100:.1f}%")
                    if r.get("f1") is not None:
                        seg.append(f"F1 {r['f1'] * 100:.1f}%")
                    logln("  ".join(seg), "ok" if (r["r"] or 0) > 0 else "info")
                seen = len(rows)

        model.add_callback("on_train_batch_end", _on_batch)
        runs_dir = model_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        _stop = threading.Event()
        _watcher = threading.Thread(target=_watch_csv, args=(runs_dir, _stop), daemon=True)
        _watcher.start()
        try:
            model.train(data=str(yml), epochs=epochs, imgsz=640, batch=8, device=0, workers=0,
                        project=str(runs_dir), name="model", exist_ok=True, verbose=False,
                        plots=False, degrees=15.0)   # workers=0: 윈도우 spawn 크래시(pickle truncated) 회피
        except _Cancelled:
            pass                 # 아래 cancel 분기에서 정리한다
        finally:
            _stop.set()
            _watcher.join(timeout=5)
            try:   # 마지막 에폭이 감시 주기 사이에 끝났을 수 있어 한 번 더 읽는다
                _rows = _csv_epochs(runs_dir)
                if _rows and _rows[-1]["epoch"] > (j.get("epoch") or 0):
                    j.update(epoch=_rows[-1]["epoch"], total_epochs=epochs,
                             cur_map=_rows[-1]["map50"], cur_map5095=_rows[-1]["map5095"])
            except Exception:   # noqa: BLE001
                pass
        if j.get("cancel"):
            logln("학습을 중단했습니다.", "err")
            j.update(stage="cancelled", running=False)
            del model; gc.collect(); torch.cuda.empty_cache()
            return
        w1 = Path(model.trainer.best)
        logln("학습 완료", "ok")
        onnx_src = None
        try:   # XR 배포용 ONNX 자동 변환(onnx 미설치 등 실패 시 건너뜀)
            onnx_src = Path(model.export(format="onnx", imgsz=640))
        except Exception as e:
            logln(f"ONNX 변환 건너뜀 ({type(e).__name__})", "info")
        del model; gc.collect(); torch.cuda.empty_cache()

        # 3) test 영상 검출 평가 (정답 라벨 없어 mAP 아님 = 검출률+신뢰도+클래스분포+육안)
        eval_res = []
        # 학습한 부품 중 내부 평가 영상(data/_eval/<부품>)이 있는 것만 검출률을 낸다. 없으면 건너뛴다.
        tests = [(cls, vp) for cls in sorted(per.keys()) for vp in eval_videos(cls)]
        if tests:
            j.update(stage="eval", eval_total=len(tests), eval_done=0, eval_frac=0.0)
            det = YOLO(str(w1))
            frames_of = lambda vp: autolabel._cached_frames(autolabel._frames_dir(vp))
            total_frames = sum(len(frames_of(vp)) for _, vp in tests) or 1
            done_frames = 0
            for part, vp_ts in tests:
                fs = frames_of(vp_ts)
                ts = vp_ts.stem                              # 결과 표기용 이름
                outd = sess / "pred" / part
                outd.mkdir(parents=True, exist_ok=True)
                hit, confs, cls_cnt = 0, [], {}
                for p in fs:
                    im = _rd(p)
                    r = det.predict(source=im, conf=VAL_CONF, imgsz=640, verbose=False)[0]
                    if len(r.boxes):
                        hit += 1
                        for b, c, cl in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                            r.boxes.cls.cpu().numpy()):
                            confs.append(float(c))
                            nm = names[int(cl)] if int(cl) < len(names) else str(int(cl))
                            cls_cnt[nm] = cls_cnt.get(nm, 0) + 1
                            x1, y1, x2, y2 = map(int, b)
                            _r, _bt, _ew = _ov_sizes(im.shape[1], im.shape[0])
                            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 150, 255), _bt)
                            cv2.putText(im, f"{nm} {float(c):.2f}", (x1, max(y1 - 6, 16)),
                                        cv2.FONT_HERSHEY_SIMPLEX, max(0.6, im.shape[1] / 1600),
                                        (0, 150, 255), _ew + 1)
                    cv2.imwrite(str(outd / f"{p.stem}.jpg"), im)
                    done_frames += 1
                    if done_frames % 15 == 0:
                        j.update(eval_frac=done_frames / total_frames)
                top = sorted(cls_cnt.items(), key=lambda x: -x[1])[:3]
                eval_res.append({"src": ts, "frames": len(fs), "detected": hit,
                                 "rate": round(hit / len(fs), 3) if fs else 0.0,
                                 "mean_conf": round(float(np.mean(confs)), 3) if confs else 0.0,
                                 "top_classes": top,
                                 "samples": [_b64(_rd(p), w=300) for p in sorted(outd.glob("*.jpg"))[::max(1, len(fs) // 6)]][:6]})
                j.update(eval_done=len(eval_res), eval_frac=done_frames / total_frames)
            del det; gc.collect(); torch.cuda.empty_cache()

        # 모델 버전 = 이 run 폴더(results/<시각>). best.pt·onnx → model/, 기록·지표 → meta.json.
        now = datetime.now()
        model_id = runid                              # 버전 id = 시각 폴더명(results/<시각>)
        parts_list = sorted(per.keys())
        parts_short = ", ".join(parts_list[:3]) + (f" 외 {len(parts_list) - 3}개" if len(parts_list) > 3 else "")
        label = f"{now.strftime('%Y-%m-%d %H:%M')} · {len(per)}종 · {parts_short}"
        model_dir.mkdir(parents=True, exist_ok=True)
        reg_pt = model_dir / "best.pt"
        shutil.copy(w1, reg_pt)
        logln(f"모델 저장: {runid}/model/best.pt", "ok")
        reg_onnx = None
        if onnx_src and Path(onnx_src).exists():
            reg_onnx = model_dir / "model.onnx"
            shutil.copy(onnx_src, reg_onnx)
            logln("ONNX 변환 완료", "ok")
        try:   # YOLO 학습 runs 원본(대용량) 삭제 — best.pt 복사본만 유지
            shutil.rmtree(model_dir / "runs", ignore_errors=True)
        except Exception:
            pass
        eval_meta = [{k: r[k] for k in ("src", "frames", "detected", "rate", "mean_conf", "top_classes")} for r in eval_res]
        meta = {"run": True, "model_id": model_id, "label": label, "time": now.strftime("%Y-%m-%d %H:%M:%S"),
                "session": runid, "classes": parts_list, "n_classes": len(per), "per_class": per,
                "n_images": n_img, "learn_rate": learn_rate, "input_total": input_cnt, "miss": miss,
                "weights": str(reg_pt), "onnx": str(reg_onnx) if reg_onnx else None,
                "curve": j.get("curve", []), "log": j.get("log", []), "eval": eval_meta}
        (sess / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        _prune_runs(keep=10)                          # 오래된 run 정리(현재 서비스 버전은 보호)
        _db_sync_runs()                               # 학습 이력을 DB 에 반영
        logln("전체 완료", "ok")
        j.update(stage="done", running=False, session=runid, model_id=model_id, label=label,
                 weights=str(reg_pt), onnx=str(reg_onnx) if reg_onnx else None,
                 n_images=n_img, n_classes=len(per), per_class=per, eval=eval_res, miss=miss)
    except _Cancelled:                # 중단은 오류가 아니다(라벨 산입·복사·학습 어느 단계든)
        logln("학습을 중단했습니다.", "err")
        j.update(stage="cancelled", running=False)
    except Exception as e:
        logln(f"오류: {type(e).__name__}: {e}", "err")
        logger.exception(f"[multiclass:{job_id}] 학습 실패 (session={session})")
        j.update(stage="error", error=f"{type(e).__name__}: {e}", running=False)
    finally:
        _BUSY["on"] = False
        gc.collect(); torch.cuda.empty_cache()

def start_multiclass(session, epochs, only_classes=None, augment=False, replay_served=False):
    with _LOCK:
        if _BUSY["on"]:
            return {"error": "이미 실행 중입니다."}
        _BUSY["on"] = True
    session = PERSIST                              # 단일 영속 세션(입력=부품폴더, 출력=results/parts/autolabel)
    jid = uuid.uuid4().hex[:8]
    JOBS[jid] = {"stage": "start", "running": True, "error": None, "log": [], "curve": []}
    _ACTIVE.update(job=jid, kind="multiclass", session=session)
    threading.Thread(target=_run_multiclass,
                     args=(jid, session, int(epochs or EPOCHS), only_classes or None, bool(augment), bool(replay_served)),
                     daemon=True).start()
    return {"job": jid}

def cancel_multiclass(job=None):
    """학습 중단 요청. 배치 콜백이 이 표시를 보고 즉시 예외로 빠져나온다.

    화면을 새로 열거나 탭을 옮기면 프론트가 job id 를 잃는다. 그때도 멈춰야 하므로
    id 가 없거나 모르는 값이면 진행 중인 잡(_ACTIVE)을 대상으로 삼는다.
    """
    j = JOBS.get(job)
    if not j:
        job = _ACTIVE.get("job")
        j = JOBS.get(job)
    if not j:
        return {"error": "진행 중인 학습이 없습니다."}
    j["cancel"] = True
    return {"ok": True, "job": job, "stage": j.get("stage")}

# ==================== 3단계: 모델 평가·적용 (A/B 비교 → 서비스 적용/롤백) ====================
def weights_of(runid):
    """run 의 가중치 경로. meta.json 의 weights 문자열은 학습한 기계의 절대경로라
    다른 기계(Thor·컨테이너)에서 깨진다. 폴더 규약으로 직접 찾는 것이 유일하게 안전하다."""
    p = RESULTS / str(runid) / "model" / "best.pt"
    return p if p.exists() else None


SERVED_PTR = RESULTS / "_served.json"        # 현재 서비스 모델 포인터 {"run": "<시각>"}
_RUN_RE = re.compile(r"^\d{6}_\d{6}$")       # 학습 run 폴더명 = YYMMDD_HHMMSS
_ACTIVE = {"job": None, "kind": None, "session": None}   # 현재 진행 중인 학습/평가 잡(재진입 복구용)

def active_job():
    """지금 진행 중인 학습/평가 잡의 현재 상태. 페이지 재진입 시 이 잡에 다시 붙어 복구한다."""
    a = _ACTIVE
    if not a.get("job"):
        return {}
    j = JOBS.get(a["job"])
    if not j or not j.get("running"):
        return {}
    return {"job": a["job"], "kind": a["kind"], "session": a.get("session"), **j}

def _last_map50(curve):
    """학습 곡선에서 마지막 유효 mAP@0.5."""
    for e in reversed(curve or []):
        if e.get("map50") is not None:
            return round(float(e["map50"]), 3)
    return None

def _run_dirs():
    """results/ 아래 학습 run 폴더(최신순). meta.json 있는 시각(YYMMDD_HHMMSS) 폴더만."""
    if not RESULTS.exists():
        return []
    ds = [d for d in RESULTS.iterdir() if d.is_dir() and _RUN_RE.match(d.name) and (d / "meta.json").exists()]
    return sorted(ds, key=lambda d: d.name, reverse=True)

def _run_meta(runid):
    mp = RESULTS / str(runid) / "meta.json"
    if not runid or not mp.exists():
        return None
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        return None

def _latest_run():
    ds = _run_dirs()
    return ds[0].name if ds else None

def _served_runid():
    if SERVED_PTR.exists():
        try:
            return json.loads(SERVED_PTR.read_text(encoding="utf-8")).get("run")
        except Exception:
            return None
    return None

def _sync_served_files(runid):
    """현재 서비스 모델(run 의 best.pt·model.onnx)을 고정 경로로 복사한다.
    run 폴더(results/<시각>/model)는 버전 이력=소스로 유지하고, 외부 서빙·Thor TensorRT 는
    항상 같은 고정 경로(config.NEW_MODEL_PT=models/model.pt, NEW_MODEL_ONNX=models/model.onnx)를 쓴다.
    onnx 가 없으면(구 run) 고정 onnx 를 지워 pt 와 안 맞는 낡은 onnx 서빙을 막는다."""
    m = _run_meta(runid) or {}
    src_pt = weights_of(runid) or Path(m.get("weights", ""))
    if not src_pt.exists():
        return
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_pt, config.NEW_MODEL_PT)
    src_onnx = src_pt.parent / "model.onnx"
    if src_onnx.exists():
        shutil.copy(src_onnx, config.NEW_MODEL_ONNX)
    elif config.NEW_MODEL_ONNX.exists():
        try:
            config.NEW_MODEL_ONNX.unlink()
        except Exception:
            pass
    _deploy_to_yolo_server()   # 신규 모델 적용/롤백 시 YOLO 추론 서버로 자동 배포


def _deploy_to_yolo_server():
    """현재 고정 모델(models/model.pt·onnx)을 YOLO 추론 서버(backend/yolo_server)로 복사 + 핫 리로드.
    별도 배포 버튼 없이 '신규 모델 적용'만으로 운영 서버에 반영된다. 서버 미기동이면 파일만 배포(다음 기동 시 로드)."""
    dst = config.YOLO_SERVER_DIR
    if not dst.exists():
        return
    try:
        if config.NEW_MODEL_PT.exists():
            shutil.copy(config.NEW_MODEL_PT, dst / "model.pt")
        if config.NEW_MODEL_ONNX.exists():
            shutil.copy(config.NEW_MODEL_ONNX, dst / "model.onnx")
        elif (dst / "model.onnx").exists():
            (dst / "model.onnx").unlink()   # pt 와 안 맞는 낡은 onnx 제거
        logger.info(f"[deploy] YOLO 서버로 모델 배포: {dst}")
    except Exception as e:
        logger.warning(f"[deploy] YOLO 서버 파일 복사 실패: {e}")
        return
    try:   # 서버가 떠 있으면 핫 리로드
        import urllib.request
        urllib.request.urlopen(urllib.request.Request(config.YOLO_SERVER_URL + "/reload", method="POST"), timeout=5)
        logger.info("[deploy] YOLO 서버 모델 리로드 완료")
    except Exception as e:
        logger.info(f"[deploy] YOLO 서버 리로드 스킵(미기동?): {e}")


def _set_served(runid):
    SERVED_PTR.write_text(json.dumps(
        {"run": runid, "applied": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        ensure_ascii=False, indent=2), encoding="utf-8")
    _sync_served_files(runid)   # 현재 모델을 고정 경로(models/model.pt·onnx)로 동기화
    _db_sync_runs()             # is_active(현재 서비스 모델)까지 DB 에 반영

def _prune_runs(keep=10):
    """오래된 run 폴더 정리. 현재 서비스(served) run 은 10개 밖이어도 보호."""
    served = _served_runid()
    for d in _run_dirs()[keep:]:
        if d.name == served:
            continue
        shutil.rmtree(d, ignore_errors=True)
        logger.info(f"[prune] 오래된 run 파기: {d.name}")

def served_model():
    """현재 서비스 중인 모델 = _served.json 이 가리키는 run. 없으면 None."""
    runid = _served_runid()
    m = _run_meta(runid) if runid else None
    w = weights_of(runid) if runid else None
    if m and w:
        cls = m.get("classes", [])
        return {"weights": str(w), "classes": cls, "n_classes": len(cls),
                "label": m.get("label", ""), "session": runid, "applied": runid, "model_id": runid,
                "time": m.get("time", ""), "map50": _last_map50(m.get("curve")),
                "gen_rate": m.get("gen_rate"), "newp_rate": m.get("newp_rate")}
    return None

def list_models():
    """run 버전 목록(최신순) — 버전 히스토리·선택형 롤백 UI 재료."""
    active_mid = _served_runid()
    out = []
    for d in _run_dirs():
        info = _run_meta(d.name) or {}
        mid = info.get("model_id") or d.name
        out.append({"model_id": mid, "label": info.get("label", ""), "time": info.get("time", ""),
                    "classes": info.get("classes", []),
                    "n_classes": info.get("n_classes", len(info.get("classes", []))),
                    "n_images": info.get("n_images"), "map50": _last_map50(info.get("curve")),
                    "gen_rate": info.get("gen_rate"), "newp_rate": info.get("newp_rate"),
                    "has_pt": weights_of(mid) is not None,
                    "is_active": (active_mid is not None and active_mid == mid)})
    return {"models": out, "active": active_mid}

def apply_model(session=None):
    """학습 run 을 서비스로 지정(_served.json 포인터).

    session 은 run id(results/<시각>)여야 한다. 프론트가 평가 중인 run 을 명시해서 보낸다.
    유효한 run id 가 아니면 최신 run 으로 폴백하지만, 그 경우 평가 중에 끝난 다른 학습이
    올라갈 수 있어 경고를 남긴다."""
    runid = session if _run_meta(session) else _latest_run()
    if session and runid != session:
        logger.warning(f"[apply] run id 가 아닌 값({session}) -> 최신 run({runid}) 으로 폴백")
    if not runid:
        return {"error": "학습된 모델이 없습니다."}
    m = _run_meta(runid) or {}
    _set_served(runid)
    return {"ok": True, "session": runid, "label": m.get("label", ""), "n_classes": m.get("n_classes", 0)}

def rollback_to(model_id):
    """선택형 롤백: 과거 run(model_id=<시각>)을 서비스로 지정."""
    m = _run_meta(model_id)
    if not m:
        return {"error": "해당 버전이 없습니다."}
    _set_served(model_id)
    return {"ok": True, "model_id": model_id, "label": m.get("label", ""), "n_classes": m.get("n_classes", 0)}

def rollback():
    """신규 모델 폐기(서비스 모델 미변경). 프론트가 라벨 화면으로 복귀."""
    return {"ok": True}

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None

def _run_compare(job_id, session, base_model_id=None):
    """신규 모델 vs 기존(서비스) 모델을 같은 테스트 영상에 돌려 '정답 클래스 검출률'(인식률) 비교.
    기존 부품(망각 여부)/신규 부품 성능을 분리 집계하고 적용·롤백을 권장한다."""
    j = JOBS[job_id]
    session = _latest_run()                        # 신규 모델 = 방금 학습한 최신 run(results/<시각>)
    logger.info(f"[compare:{job_id}] 시작 (session={session})")
    try:
        sys.path.insert(0, str(config.BASE_DIR / "scripts" / "experiments"))
        import build_multiclass as bm
        from ultralytics import YOLO
        meta = _run_meta(session)
        if not meta:
            raise RuntimeError("학습된 모델이 없습니다. 먼저 학습을 실행하세요.")
        new_w = str(weights_of(session) or meta.get("weights"))
        new_classes = list(meta.get("per_class", {}).keys())
        new_label = meta.get("label", session)
        if base_model_id:   # 타임라인에서 특정 과거 버전(run)을 기준으로 선택한 경우
            info = _run_meta(base_model_id)
            bw = str(weights_of(base_model_id) or (info or {}).get("weights", ""))
            base = ({"weights": bw, "classes": info.get("classes", []),
                     "label": info.get("label", "") or base_model_id, "session": base_model_id}
                    if info and Path(bw).exists() else served_model())
        else:
            base = served_model()
        base_classes = base["classes"] if base else []
        base_set = set(base_classes)
        base_label = (base.get("label") or base.get("session")) if base else "기존 모델 없음"

        # 클래스 → 비교에 쓸 영상. 내부 평가 영상(data/_eval/<부품>)이 있으면 그것(미학습),
        # 없으면 그 부품 영상 아무거나(학습에 쓴 영상일 수 있음 = in-sample).
        vid_by_class = {}
        for vp in autolabel._videos():
            vid_by_class.setdefault(bm.stem_to_class(vp.stem + "_0"), vp)
        def _pick_video(c):
            ev = eval_videos(c)
            return ev[0] if ev else vid_by_class.get(c)

        # 테스트 대상: 신규 부품(전부) + 기존 부품 표본(최대 12)
        items = []   # (class, video_path, kind)
        for c in [x for x in new_classes if x not in base_set]:
            if _pick_video(c):
                items.append((c, _pick_video(c), "new"))
        for c in sorted(base_classes)[:12]:
            if _pick_video(c):
                items.append((c, _pick_video(c), "base"))
        if not items:
            raise RuntimeError("비교할 테스트 영상을 찾지 못했습니다.")

        total = len(items)
        j.update(stage="compare", compare_total=total, compare_done=0, compare_frac=0.0)
        new_model = YOLO(new_w)
        base_model = YOLO(base["weights"]) if base else None

        def _dets(model, img, cls):
            """검출 결과를 '사진에 굽지 않고' 좌표로 돌려준다(정답 여부 ok 포함).

            사진에 직접 그리지 않는 이유
              1) 화면에서 박스를 켜고 끌 수 있어야 한다(덜 학습된 모델은 한 프레임에
                 수백 개를 뱉어 사진이 박스로 덮인다).
              2) 기존/신규 두 모델이 같은 프레임을 쓰므로 사진은 한 장만 보내면 된다.
            신뢰도 높은 순으로 BOX_CAP 개만 보내고, 실제 검출 개수(n)는 따로 알려준다.
            """
            r = model.predict(source=img, conf=VAL_CONF, imgsz=640, verbose=False)[0]
            names = model.names
            out = []
            for b, cf, cl in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                 r.boxes.cls.cpu().numpy()):
                nm = names[int(cl)] if int(cl) in names else str(int(cl))
                x1, y1, x2, y2 = (int(v) for v in b)
                out.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
                            "cls": nm, "conf": round(float(cf), 2), "ok": nm == cls})
            out.sort(key=lambda d: -d["conf"])
            return {"n": len(out), "box": out[:BOX_CAP]}

        def _rate(model, sub, cls):     # 프레임들 중 '정답 클래스'가 검출된 비율(인식률)
            names = model.names
            hit = 0
            for p in sub:
                r = model.predict(source=_rd(p), conf=VAL_CONF, imgsz=640, verbose=False)[0]
                if len(r.boxes) and any(
                        (names[int(cl)] if int(cl) in names else str(int(cl))) == cls
                        for cl in r.boxes.cls.cpu().numpy()):
                    hit += 1
            return round(hit / len(sub), 3) if sub else 0.0

        def _pick(seq, k):     # seq에서 k장을 균등 간격(양끝 포함)으로 고른다(부품 테이크 서브샘플)
            n = len(seq)
            if n == 0 or k <= 0:
                return []
            if k >= n:
                return list(seq)
            if k == 1:
                return [seq[0]]
            out, seen = [], set()
            for i in range(k):
                idx = round(i * (n - 1) / (k - 1))
                if idx not in seen:
                    seen.add(idx); out.append(seq[idx])
            return out

        # Before/After 샘플: 부품당 여러 프레임(신규 최대 5 / 기존 최대 3), 전체 상한 ~30장.
        # 부품 수가 많아 상한을 넘길 것 같으면 부품당 프레임 수를 비례 축소(부품당 최소 1).
        TOTAL_CAP = 30
        PART_CAP = {"new": 5, "base": 3}
        want = sum(PART_CAP[kd] for (_, _, kd) in items)
        scale = min(1.0, TOTAL_CAP / want) if want else 1.0

        rows, samples = [], []
        for k, (c, vp, kind) in enumerate(items, 1):
            fs = autolabel._cached_frames(autolabel._frames_dir(vp))
            sub = fs[::max(1, len(fs) // 15)][:15]
            new_r = _rate(new_model, sub, c)
            base_r = _rate(base_model, sub, c) if base_model else None
            # 화면의 '모델 결과 비교'를 없애면서 샘플 이미지 생성도 멈췄다. 정답이 없어 판단 근거가
            # 못 되는데 두 모델로 프레임을 다시 추론해 시간만 들었다. 판단은 인식률 하락 경고로 한다.
            # ponytail: 샘플이 다시 필요하면 이 자리에서 _dets·_pick 로 되살리면 된다.
            rows.append({"part": c, "kind": kind, "new_rate": new_r, "base_rate": base_r})
            j.update(compare_done=k, compare_frac=round(k / total, 3))

        del new_model, base_model
        gc.collect(); torch.cuda.empty_cache()

        base_rows = [r for r in rows if r["kind"] == "base"]
        new_rows = [r for r in rows if r["kind"] == "new"]
        gen_after = _mean([r["new_rate"] for r in base_rows])     # 기존 부품: 신규 모델
        gen_before = _mean([r["base_rate"] for r in base_rows])   # 기존 부품: 기존 모델
        newp_after = _mean([r["new_rate"] for r in new_rows])     # 신규 부품: 신규 모델
        newp_before = _mean([r["base_rate"] for r in new_rows])   # 신규 부품: 기존 모델(대개 ~0)
        gen_drop = round((gen_before - gen_after) * 100, 1) if (gen_before is not None and gen_after is not None) else 0.0

        # 판정 문구는 두 줄로 보낸다(한 줄로 길게 흐르면 읽기 어렵다).
        # 프론트가 .verdict-msg { white-space: pre-line } 으로 줄바꿈을 살려 그린다.
        if base_rows and gen_drop >= 10:
            reco = {"level": "rollback",
                    "msg": f"신규 부품은 인식하지만 기존 부품들의 인식률이 {gen_drop:.0f}%p 하락했습니다.\n"
                           "라벨링 마스크 영역을 좁게 재조정하여 다시 학습하는 것을 권장합니다."}
        elif (newp_after or 0) >= 0.7 and (not base_rows or gen_drop < 5):
            reco = {"level": "apply",
                    "msg": "새로운 부품이 성공적으로 학습되었으며,\n"
                           "기존 부품의 인식률도 안정적입니다."}
        else:
            reco = {"level": "review",
                    "msg": "신규 부품 인식률과 기존 부품 유지 여부가 애매합니다.\n"
                           "아래 샘플 이미지를 확인한 뒤 결정하세요."}

        if session:   # 신규 모델 run(results/<시각>)의 meta.json 에 인식률 스냅샷 저장(드롭다운 표시용)
            mjp = RESULTS / session / "meta.json"
            if mjp.exists():
                try:
                    minfo = json.loads(mjp.read_text(encoding="utf-8"))
                    minfo["gen_rate"], minfo["newp_rate"] = gen_after, newp_after
                    mjp.write_text(json.dumps(minfo, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    logger.exception("모델 인식률 스냅샷 저장 실패")

        j.update(stage="done", running=False, session=session,
                 new_label=new_label, base_label=base_label,
                 baseline={"session": base["session"] if base else None, "n_classes": len(base_classes)},
                 # n = 실제로 검사한 표본 종수, n_total = 기존 모델이 가진 전체 종수(표본만 재는 이유는 시간)
                 gen={"before": gen_before, "after": gen_after, "n": len(base_rows),
                      "n_total": len(base_classes)},
                 newp={"before": newp_before, "after": newp_after, "n": len(new_rows)},
                 rows=rows, samples=samples, recommend=reco)
    except Exception as e:
        logger.exception("백그라운드 작업 오류")
        j.update(stage="error", error=f"{type(e).__name__}: {e}", running=False)
    finally:
        _BUSY["on"] = False
        gc.collect(); torch.cuda.empty_cache()

def start_compare(session, base_model_id=None):
    with _LOCK:
        if _BUSY["on"]:
            return {"error": "이미 실행 중입니다. 끝난 뒤 다시 시도하세요."}
        if not session:
            return {"error": "세션이 없습니다."}
        _BUSY["on"] = True
    jid = uuid.uuid4().hex[:8]
    JOBS[jid] = {"stage": "compare", "running": True, "error": None}
    _ACTIVE.update(job=jid, kind="compare", session=session)
    threading.Thread(target=_run_compare, args=(jid, session, base_model_id or None), daemon=True).start()
    return {"job": jid}

def delete_model(model_id):
    """버전(run=<시각>) 폴더 삭제. 현재 서비스 중인 run 은 삭제 금지."""
    if not model_id:
        return {"error": "model_id 없음"}
    if _served_runid() == model_id:
        return {"error": "현재 서비스 중인 모델은 삭제할 수 없습니다. 먼저 다른 버전으로 롤백하세요."}
    d = RESULTS / str(model_id)
    if _run_meta(model_id) is None and not d.exists():
        return {"error": "해당 버전이 없습니다."}
    shutil.rmtree(d, ignore_errors=True)
    logger.info(f"[delete_model] run {model_id} 삭제")
    return {"ok": True, "model_id": model_id}

# ==================== 학습 데이터(생성 라벨) 검수·삭제 ====================
# 라벨은 부품별 영속 저장소(data/bell412/<부품>/autolabel)에 있고, 프론트에는 단일 세션(PERSIST)으로 노출.
# 프레임 파일명(<영상>_<프레임>)만으로 어느 부품 저장소인지 역추적한다(session 인자는 무시).

def labeled_parts():
    """autolabel 라벨이 하나라도 있는 부품(폴더=클래스) 집합 — 기존/신규 판정·검수 활성 판단용."""
    parts = set()
    for lbl_dir in autolabel.AUTOLABELS.glob("*/labels"):
        if any(lbl_dir.glob("*.txt")):
            parts.add(lbl_dir.parent.name)     # results/autolabels/<부품>/labels → <부품>
    return {"parts": sorted(parts)}

def list_part_frames(part, limit=400):
    """특정 부품(폴더=클래스)의 생성된 학습(라벨) 프레임 수집(검수용). part = 폴더명."""
    if not part:
        return {"part": part, "count": 0, "frames": []}
    lbl_dir = autolabel.AUTOLABELS / part / "labels"
    out = []
    if lbl_dir.exists():
        for lp in sorted(lbl_dir.glob("*.txt")):
            out.append({"session": PERSIST, "name": lp.stem + ".jpg", "part": part})
            if len(out) >= limit:
                return {"part": part, "count": len(out), "frames": out, "truncated": True}
    return {"part": part, "count": len(out), "frames": out}

def train_frame_jpeg(session, name, w=360, part=None):
    """학습 프레임에 YOLO 라벨 bbox(초록)를 그려 JPEG 바이트로. 검수에서 어떤 라벨이 생성됐는지 확인용.
    ※ 썸네일로 축소한 뒤 박스를 그린다(고해상도에 그린 뒤 축소하면 선이 1px 미만으로 뭉개져 안 보임).
    part 지정 시 그 부품 저장소를 직접 사용(같은 이름 영상 모호성 제거), 없으면 파일명으로 역추적."""
    w = max(64, min(int(w or 360), 4096))   # 음수 w 로 cv2.resize 가 터지는 것 방지(frame_jpeg 와 동일)
    if not name or "/" in name or ".." in name:
        return None
    if part and "/" not in part and ".." not in part:
        store = _store_for_part(part)
    else:
        store = _store_for_frame(name)              # 파일명 → 그 부품 저장소(폴백)
    stem_name = Path(name).stem                     # <video>_<frame>
    video = _video_stem(stem_name)
    frame = stem_name[len(video) + 1:]
    ip = store["images"] / video / f"{frame}.jpg"   # 프레임 = images/<video>/<frame>.jpg (캐시 겸용)
    if not ip.exists():
        return None
    im = _rd(ip)
    boxes = []
    lp = store["labels"] / (stem_name + ".txt")
    if lp.exists():
        for line in lp.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) == 5:
                boxes.append(tuple(map(float, p[1:])))
    if w and im.shape[1] > w:                       # 먼저 축소
        im = cv2.resize(im, (w, int(im.shape[0] * w / im.shape[1])))
    h, wid = im.shape[:2]
    _r, _bt, _ew = _ov_sizes(wid, h)                # 축소된 표시 크기 기준(다른 화면과 같은 규칙)
    for cx, cy, bw, bh in boxes:
        cv2.rectangle(im, (int((cx - bw / 2) * wid), int((cy - bh / 2) * h)),
                      (int((cx + bw / 2) * wid), int((cy + bh / 2) * h)), (0, 180, 0), _bt)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None

def delete_train_frame(session, name, part=None):
    """검수에서 잘못된 프레임(이미지+라벨+마스크) 삭제. part 지정 시 그 부품 저장소 직접, 없으면 파일명 역추적."""
    if not name or "/" in name or ".." in name:
        return {"error": "잘못된 이름"}
    if part and "/" not in part and ".." not in part:
        store = _store_for_part(part)
    else:
        store = _store_for_frame(name)
    stem_name = Path(name).stem
    removed = False
    for p in (store["labels"] / (stem_name + ".txt"), store["boxs"] / name):   # 라벨·boxs 제거(프레임캐시는 보존)
        if p.exists():
            p.unlink(missing_ok=True); removed = True
    return {"ok": True, "name": name} if removed else {"error": "대상 없음"}

# ==================== 영상 삭제(모달) — 원본·프레임·오토라벨 산출을 함께 정리 ====================
def delete_video(src):
    """영상 삭제: 원본 파일과 그 영상의 프레임·오토라벨 산출(<stem>_*)을 바로 지운다.
    src = 부품경로 키(bell412/<부품>/videos/<stem>) 또는 stem."""
    vp = autolabel.resolve_video(src)
    if not vp:
        return {"error": f"영상을 찾지 못함: {src}"}
    vp = vp.resolve()
    stem = vp.stem
    part_root = autolabel.part_root_of(vp)
    part = part_root.name

    # 1) 원본 영상 삭제
    if vp.exists():
        vp.unlink()

    # 2) 프레임 이미지 삭제: autolabels/<부품>/images/<stem> (저장소는 한 곳뿐) — 재생성 가능
    removed_cache = 0
    d = autolabel.cache_dir_of(vp)
    if d.exists():
        removed_cache += len(list(d.glob("*.jpg")))
        shutil.rmtree(d, ignore_errors=True)

    # 3) 오토라벨 산출(이 영상분 <stem>_*)만 삭제 — 라벨·boxs. 다른 영상 라벨은 보존
    store = _store_for_part(part)
    removed_labels = 0
    for g in (store["labels"].glob(f"{stem}_*.txt"), store["boxs"].glob(f"{stem}_*.jpg")):
        for f in list(g):
            f.unlink(missing_ok=True); removed_labels += 1

    # 4) 참조샷(shots.json)에서 이 영상 항목 제거 — 안 지우면 같은 이름으로 재업로드할 때 옛 탭이 되살아난다
    sj = store["root"] / "shots.json"
    if sj.exists():
        try:
            data = json.loads(sj.read_text(encoding="utf-8")) or {}
            if isinstance(data, dict) and stem in data:
                data.pop(stem)
                sj.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            logger.exception("shots.json 정리 실패(무시)")

    autolabel._videos(force=True)   # 영상 목록 캐시 무효화(삭제 즉시 목록/폴더 반영)
    _db_sync_part(part)             # DB 색인 정리(sync 가 사라진 영상·프레임 행을 prune 한다)
    logger.info(f"[delete_video] {part}/{stem} 삭제 (프레임 {removed_cache}장, 라벨 {removed_labels}건 정리)")
    return {"ok": True, "part": part, "stem": stem,
            "removed_cache_frames": removed_cache, "removed_label_files": removed_labels}
