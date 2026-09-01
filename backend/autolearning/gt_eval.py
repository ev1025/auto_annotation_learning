# -*- coding: utf-8 -*-
"""GT 만들기 + mAP@0.5 자체 채점.

두 가지를 한 모듈에서 담당한다.
  1) 평가영상(data/bell412/_gt_videos/*.mp4) -> 프레임 표본 -> 배포 모델 스캔 -> 사람이 박스 확정(GT)
  2) 그 GT 로 클래스별 AP@0.5 를 직접 계산해 mAP 를 낸다

ultralytics 의 val 을 쓰지 않는 이유
  val 은 데이터셋 하나를 통째로 보고 '모든 클래스가 모든 이미지에서 정답'이라고 가정한다.
  우리 GT 는 부품 폴더마다 그 부품 박스만 들어 있어서(같은 프레임이 여러 폴더에 복사됨)
  합쳐서 val 에 넣으면 옆 부품 박스가 전부 FP 로 잡혀 수치가 망가진다.
  그래서 클래스별로 그 클래스 gt 폴더만 써서 AP 를 따로 계산한다. 계산식은 _ap() 주석 참고.

프레임 캐시를 results/gt_frames/ 에 따로 두는 이유
  autolabel.cache_dir_of() 는 results/autolabels/<부품>/images/<stem> 을 준다. 그건 학습 이미지
  저장소라서 평가영상 프레임을 거기 넣으면 학습 데이터에 섞인다(= 시험 문제를 미리 보여주는 셈).
  게다가 평가영상은 부품 폴더 밖(_gt_videos)에 있어 part_root_of/resolve_video 로는 찾지도 못한다.
  평가 프레임은 학습 트리와 완전히 분리한다.
"""
import json
import shutil
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))   # 공용 config
sys.path.insert(0, str(Path(__file__).resolve().parent))                    # sam2_autolabel

import cv2
import numpy as np
import torch

import config
import sam2_autolabel as sa

# ---- 확정된 규약(계약서 0장). 바꾸면 심사 문서와 화면이 어긋난다 ----
GT_PARTS = ["seal_support", "r-ring", "shim", "split_gearbox", "bracket"]
PASS_THRESHOLD = 51            # mAP@0.5 x 100 이 이 값 이상이면 합격
REPORT_CONF = 0.70             # TP/FP/FN 을 보고하는 운영 임계
CURVE_CONF = 0.001             # PR 곡선을 그릴 하한(이 이상 전 구간)
EVAL_IMGSZ = config.IMG_SIZE   # 학습과 같은 입력 크기로 재야 의미가 있다

PARTS_ROOT = config.DATA_DIR / "bell412"
GT_VIDEO_DIR = PARTS_ROOT / "_gt_videos"
FRAMES_ROOT = config.BASE_DIR / "results" / "gt_frames"     # 평가 프레임 캐시(학습 트리와 분리)
SCAN_ROOT = config.BASE_DIR / "results" / "gt_scan"
EVAL_ROOT = config.BASE_DIR / "results" / "gt_eval"
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

_CUT_LOCK = threading.Lock()   # 같은 영상을 두 요청이 동시에 자르면 반쪽 파일이 남는다


# ==================== 파일 입출력(한글 경로 안전) ====================
# 이 레포는 'OneDrive/바탕 화면/...' 아래에 있다. cv2.imread/imwrite 는 비ASCII 경로에서
# 조용히 실패(None / False)하므로 항상 numpy 로 바이트를 주고받는다(sa._rd 와 같은 이유).
def _rd(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def _wr(p, img, q=95):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    if not ok:
        raise RuntimeError(f"JPEG 인코딩 실패: {p}")
    buf.tofile(str(p))


def _safe_name(s):
    """경로 조작 방지. 부품명·영상 stem 이 쿼리·본문으로 들어오므로 구분자를 막는다."""
    s = str(s or "").strip()
    if not s or "/" in s or "\\" in s or ".." in s:
        return None
    return s


def _videos():
    if not GT_VIDEO_DIR.exists():
        return []
    return sorted(p for p in GT_VIDEO_DIR.iterdir()
                  if p.is_file() and p.suffix.lower() in VIDEO_EXT)


def _video_of(stem):
    return next((p for p in _videos() if p.stem == stem), None)


def _cached(stem):
    """캐시된 평가 프레임 목록(파일명 = 원본 프레임 번호 5자리)."""
    d = FRAMES_ROOT / stem
    return sorted(d.glob("*.jpg")) if d.exists() else []


def _total_frames(vp):
    cap = cv2.VideoCapture(str(vp))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return n


# ==================== 2.1 영상 목록 ====================
def list_videos():
    vids = _videos()
    if not vids:
        # 목록이 비는 것과 폴더가 없는 것은 사람이 대응할 일이 다르다. 이유를 같이 준다
        return {"videos": [], "error": f"평가영상이 없습니다: {GT_VIDEO_DIR}"}
    out = []
    for vp in vids:
        cached = _cached(vp.stem)
        out.append({"stem": vp.stem,
                    "frames": _total_frames(vp),
                    "cached": len(cached),
                    "ready": bool(cached),
                    "mb": round(vp.stat().st_size / 1e6, 1)})
    return {"videos": out}


# ==================== 2.2 프레임 준비 ====================
def prepare(stem, step=15):
    """영상을 step 간격으로 잘라 results/gt_frames/<stem>/<원본프레임번호>.jpg 로 저장.

    멱등하게 만드는 방법: 뽑을 프레임 번호를 미리 정해 두고 이미 있는 파일은 건너뛴다.
    step 을 바꿔 다시 불러도 없는 번호만 추가된다(이미 만든 GT 가 가리키는 프레임이 사라지지 않는다).
    """
    stem = _safe_name(stem)
    if not stem:
        return {"error": "stem 이 올바르지 않습니다."}
    vp = _video_of(stem)
    if not vp:
        return {"error": f"평가영상을 찾지 못했습니다: {stem}"}
    step = max(1, min(int(step or 15), 300))
    dest = FRAMES_ROOT / stem
    total = _total_frames(vp)
    want = list(range(0, total, step)) if total else []
    # 전부 있으면 영상을 열지 않는다(3000 프레임 디코딩은 수십 초라 화면이 멈춘 것처럼 보인다)
    if want and all((dest / f"{i:05d}.jpg").exists() for i in want):
        return {"stem": stem, "count": len(_cached(stem)), "step": step, "ready": True}

    with _CUT_LOCK:
        dest.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(vp))
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            # step 간격 표본만 저장한다. 전 프레임을 다 쓰면 인접 프레임은 사실상 같은 그림이라
            # GT 장수만 늘고 평가 다양성은 늘지 않는다
            if i % step == 0:
                fp = dest / f"{i:05d}.jpg"
                if not fp.exists():
                    _wr(fp, fr)
            i += 1
        cap.release()
    return {"stem": stem, "count": len(_cached(stem)), "step": step, "ready": True}


# ==================== 2.3 프레임 이미지 ====================
def frame_jpeg(stem, idx, w=None):
    """캐시 프레임 한 장. idx 는 캐시 순번이 아니라 원본 프레임 번호다."""
    stem = _safe_name(stem)
    if not stem:
        return None
    fp = FRAMES_ROOT / stem / f"{int(idx):05d}.jpg"
    if not fp.exists():
        return None
    if not w:
        return fp.read_bytes()          # 원본 그대로(재인코딩 없음 = 박스 좌표 기준이 흔들리지 않는다)
    im = _rd(fp)
    if im is None:
        return None
    w = max(64, min(int(w), 4096))      # 음수·거대값이 그대로 resize 로 가면 예외 -> 500
    if im.shape[1] > w:
        im = cv2.resize(im, (w, int(im.shape[0] * w / im.shape[1])))
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None


# ==================== 모델 로딩 공용 ====================
def _resolve_weights(weights=None):
    """(가중치경로, 화면표시용 라벨). weights 가 없으면 현재 서비스 모델."""
    if weights:
        wp = Path(str(weights))
        if not wp.exists():
            return None, f"가중치가 없습니다: {weights}"
        label = ""
        try:   # run 폴더 규약(results/<시각>/model/best.pt)이면 그 run 의 라벨을 쓴다
            label = (sa._run_meta(wp.parent.parent.name) or {}).get("label", "")
        except Exception:   # noqa: BLE001 - 라벨은 표시용이라 없어도 채점은 된다
            label = ""
        return str(wp), (label or wp.name)
    m = sa.served_model()
    if not m:
        return None, "서비스 중인 모델이 없습니다. 먼저 모델을 적용하세요."
    return m["weights"], (m.get("label") or m.get("session") or "")


# ==================== 2.4 스캔 ====================
def _scan_runs():
    if not SCAN_ROOT.exists():
        return []
    return sorted((d.name for d in SCAN_ROOT.iterdir()
                   if d.is_dir() and (d / "scan.json").exists()), reverse=True)


def _run_scan(job_id, stems, step, conf):
    j = sa.JOBS[job_id]
    try:
        w, label = _resolve_weights(None)
        if not w:
            raise RuntimeError(label)
        from ultralytics import YOLO
        model = YOLO(w)
        names = model.names

        # 프레임을 먼저 다 준비한다. 추론 중간에 디스크 컷이 끼면 진행률이 튀어 멈춘 것처럼 보인다
        todo = []                        # (stem, 프레임경로)
        for s in stems:
            j.update(stage="scan", note=f"{s} 프레임 준비")
            prepare(s, step)
            todo += [(s, p) for p in _cached(s)]
        total = len(todo)
        if not total:
            raise RuntimeError("자른 프레임이 없습니다. 영상과 step 을 확인하세요.")
        j.update(stage="scan", total=total, done=0, note="스캔 시작")

        cls_frames = {}                  # 클래스 -> 그 클래스가 검출된 프레임 수
        frames = []
        for k, (s, p) in enumerate(todo, 1):
            im = _rd(p)
            if im is None:
                continue
            r = model.predict(source=im, conf=conf, imgsz=EVAL_IMGSZ, verbose=False)[0]
            dets = []
            for b, cf, cl in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                 r.boxes.cls.cpu().numpy()):
                nm = names[int(cl)] if int(cl) in names else str(int(cl))
                dets.append({"cls": nm, "conf": round(float(cf), 4),
                             "bbox": [int(v) for v in b]})     # 원본 픽셀 XYXY
            if dets:
                dets.sort(key=lambda d: -d["conf"])
                frames.append({"stem": s, "idx": int(p.stem), "dets": dets})
                for nm in {d["cls"] for d in dets}:   # 한 프레임에 두 개 있어도 1 로 센다
                    cls_frames[nm] = cls_frames.get(nm, 0) + 1
            if k % 5 == 0 or k == total:
                j.update(done=k, note=f"{k}/{total} · {s}")

        run = datetime.now().strftime("%y%m%d_%H%M%S")
        out = SCAN_ROOT / run
        out.mkdir(parents=True, exist_ok=True)
        payload = {"run": run, "weights": str(w), "label": label, "conf": conf, "step": step,
                   "stems": list(stems),
                   "classes": dict(sorted(cls_frames.items(), key=lambda kv: -kv[1])),
                   "frames": frames}
        (out / "scan.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
        j.update(stage="done", running=False, run=run, classes=payload["classes"],
                 frames=total, hit_frames=len(frames), note=f"검출 {len(frames)}/{total} 프레임")
    except Exception as e:   # noqa: BLE001 - 서버가 죽지 않게 하고 원인은 화면에 띄운다
        sa.logger.exception("gt scan 오류")
        j.update(stage="error", running=False, error=f"{type(e).__name__}: {e}"[:300])
    finally:
        sa._BUSY["on"] = False
        _free()


def start_scan(stems, step=15, conf=0.10):
    stems = [x for x in (_safe_name(s) for s in (stems or [])) if x]
    if not stems:
        return {"error": "스캔할 영상을 고르세요."}
    unknown = [s for s in stems if not _video_of(s)]
    if unknown:
        return {"error": f"평가영상을 찾지 못했습니다: {', '.join(unknown)}"}
    with sa._LOCK:
        # 학습·전파와 GPU 를 동시에 물면 OOM 이다. 기존의 '한 번에 한 잡' 규칙을 그대로 따른다
        if sa._BUSY["on"]:
            return {"error": "이미 실행 중입니다."}
        sa._BUSY["on"] = True
    jid = uuid.uuid4().hex[:8]
    sa.JOBS[jid] = {"stage": "scan", "running": True, "error": None,
                    "done": 0, "total": 0, "note": "준비 중"}
    threading.Thread(target=_run_scan,
                     args=(jid, stems, max(1, int(step or 15)), float(conf or 0.10)),
                     daemon=True).start()
    return {"job": jid}


# ==================== 2.5 스캔 결과 조회 ====================
def scan_result(run=None, parts=None, min_conf=0.0):
    runs = _scan_runs()
    if not runs:
        return {"error": "스캔 기록이 없습니다. 먼저 인식 스캔을 실행하세요.", "runs": []}
    run = _safe_name(run) or runs[0]
    fp = SCAN_ROOT / run / "scan.json"
    if not fp.exists():
        return {"error": f"스캔 기록이 없습니다: {run}", "runs": runs}
    d = json.loads(fp.read_text(encoding="utf-8"))
    want = {p for p in (parts or []) if p}
    mc = float(min_conf or 0.0)
    frames = []
    for fr in d.get("frames", []):
        dets = [x for x in fr.get("dets", []) if float(x.get("conf", 0)) >= mc]
        if want and not any(x["cls"] in want for x in dets):
            continue                     # 고른 부품이 하나도 없는 프레임은 볼 필요가 없다
        if dets:
            frames.append({"stem": fr["stem"], "idx": fr["idx"], "dets": dets})
    return {"run": run, "runs": runs, "weights": d.get("weights", ""), "label": d.get("label", ""),
            "step": d.get("step"), "conf": d.get("conf"), "classes": d.get("classes", {}),
            "frames": frames}


# ==================== 2.6 GT 현황 ====================
def _gt_dirs(part):
    g = PARTS_ROOT / part / "gt"
    return g / "images", g / "labels"


def gt_status(parts=None):
    parts = [p for p in (_safe_name(x) for x in (parts or GT_PARTS)) if p]
    m = sa.served_model() or {}
    in_model = set(m.get("classes", []))
    out = []
    for part in parts:
        imgs, lbls = _gt_dirs(part)
        names, boxes = [], 0
        if lbls.exists():
            for lp in sorted(lbls.glob("*.txt")):
                lines = [ln for ln in lp.read_text(encoding="utf-8").splitlines() if ln.strip()]
                if not lines:
                    continue             # 빈 라벨은 GT 가 아니다(삭제 잔여물)
                names.append(lp.stem)
                boxes += len(lines)
        n_img = len(list(imgs.glob("*.jpg"))) if imgs.exists() else 0
        out.append({"part": part, "images": n_img, "labels": len(names), "boxes": boxes,
                    "in_model": part in in_model, "frames": names})
    return {"parts": out}


# ==================== 2.7 GT 저장 ====================
def commit(part, stem, idx, boxes):
    part, stem = _safe_name(part), _safe_name(stem)
    if not part or not stem:
        return {"error": "part / stem 이 올바르지 않습니다."}
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        return {"error": "idx 가 정수가 아닙니다."}
    src = FRAMES_ROOT / stem / f"{idx:05d}.jpg"
    if not src.exists():
        return {"error": f"캐시 프레임이 없습니다: {stem}/{idx:05d}.jpg"}

    imgs, lbls = _gt_dirs(part)
    imgs.mkdir(parents=True, exist_ok=True)
    lbls.mkdir(parents=True, exist_ok=True)
    name = f"{stem}_{idx:05d}"
    ip, lp = imgs / f"{name}.jpg", lbls / f"{name}.txt"

    if not boxes:
        # 빈 배열 = 그 부품의 이 GT 를 지운다(사람이 잘못 그린 것을 되돌리는 유일한 경로)
        for p in (ip, lp):
            if p.exists():
                p.unlink()
        st = gt_status([part])["parts"][0]
        return {"ok": True, "part": part, "name": name, "boxes": 0,
                "images": st["images"], "labels": st["labels"]}

    lines = []
    for b in boxes:
        if not isinstance(b, (list, tuple)) or len(b) < 4:
            return {"error": f"박스 형식이 잘못됐습니다: {b}"}
        cx, cy, bw, bh = (min(max(float(v), 0.0), 1.0) for v in b[:4])
        if bw <= 0 or bh <= 0:
            return {"error": "박스 폭·높이가 0 입니다."}
        # class 는 항상 0. 부품 구분은 폴더가 한다(폴더마다 그 부품 박스만 들어간다)
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    shutil.copy(src, ip)                 # 원본 해상도 그대로 복사(재인코딩·축소 금지)
    lp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    st = gt_status([part])["parts"][0]
    return {"ok": True, "part": part, "name": name, "boxes": len(lines),
            "images": st["images"], "labels": st["labels"]}


# ==================== 2.8 SAM2 전파(후보 생성) ====================
def _free():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _norm_box(bb, w, h):
    """픽셀 XYXY -> 정규화 cx cy w h."""
    x1, y1, x2, y2 = bb
    return [round((x1 + x2) / 2 / w, 6), round((y1 + y2) / 2 / h, 6),
            round((x2 - x1) / w, 6), round((y2 - y1) / h, 6)]


def _run_propagate(job_id, stem, idx, box_n, span):
    """한 프레임의 박스를 앞뒤로 SAM2 전파해 후보를 만든다. GT 로 저장하지 않는다.

    sam2_autolabel._propagate_into 를 안 쓰는 이유(계약서 2.8): 그 함수는 점 프롬프트 전용이고
    학습 저장소(labels/boxs)에 바로 파일을 쓴다. 여기서는 박스 프롬프트가 필요하고
    사람이 확인하기 전에는 아무 것도 저장하면 안 된다.
    """
    j = sa.JOBS[job_id]
    predictor = state = None
    try:
        from sam2.build_sam import build_sam2_video_predictor
        files = _cached(stem)
        if not files:
            raise RuntimeError(f"프레임 캐시가 없습니다: {stem}")
        nums = [int(p.stem) for p in files]
        if idx not in nums:
            raise RuntimeError(f"캐시에 없는 프레임 번호: {idx}")
        pos = nums.index(idx)            # SAM2 는 캐시 순번으로 프레임을 센다(파일 번호가 아니다)
        im = _rd(files[pos])
        if im is None:
            raise RuntimeError(f"프레임을 읽지 못했습니다: {files[pos].name}")
        h, w = im.shape[:2]
        cx, cy, bw, bh = (float(v) for v in box_n[:4])
        px = [max(0.0, (cx - bw / 2) * w), max(0.0, (cy - bh / 2) * h),
              min(float(w), (cx + bw / 2) * w), min(float(h), (cy + bh / 2) * h)]
        if px[2] - px[0] < 2 or px[3] - px[1] < 2:
            raise RuntimeError("박스가 너무 작습니다.")
        j.update(stage="propagate", total=span * 2, done=0, note="SAM2 로딩")

        # _MASK_LOCK: 마스크 미리보기(mask_preview)와 예측기·GPU 를 동시에 쓰지 않게 줄을 세운다
        with sa._MASK_LOCK:
            predictor = build_sam2_video_predictor(sa.CFG, str(sa.CKPT), device=sa.DEV)
            logit_by_pos = {}
            with torch.inference_mode(), torch.autocast(sa.DEV, dtype=torch.bfloat16):
                state = predictor.init_state(video_path=str(FRAMES_ROOT / stem),
                                            offload_video_to_cpu=True, offload_state_to_cpu=True)
                # add_new_points_or_box(inference_state, frame_idx, obj_id, points, labels,
                #                       clear_old_points, normalize_coords, box)
                # box 는 원본 픽셀 XYXY. normalize_coords 기본 True 라 내부에서 나눠 준다
                predictor.add_new_points_or_box(inference_state=state, frame_idx=pos, obj_id=1,
                                                box=np.array(px, dtype=np.float32))
                for rev in (False, True):        # 앞으로 span, 뒤로 span
                    for fidx, _ids, logits in predictor.propagate_in_video(
                            state, start_frame_idx=pos,
                            max_frame_num_to_track=span, reverse=rev):
                        lg = logits[0].float().cpu().numpy()
                        logit_by_pos[int(fidx)] = lg[0] if lg.ndim == 3 else lg
                        j.update(done=len(logit_by_pos),
                                 note=f"전파 {len(logit_by_pos)}/{span * 2}")
        state = predictor = None
        sa.free_sam2()

        results = []
        for fpos, lg in sorted(logit_by_pos.items()):
            if fpos == pos:
                continue                 # 프롬프트 프레임은 사람이 이미 그린 것
            mk = lg > 0.0
            if not mk.any():
                continue
            # score = 마스크 내부 픽셀의 평균 확신도(sigmoid). 비디오 전파는 IoU 예측값을
            # 돌려주지 않으므로 로짓 세기를 사람이 볼 숫자로 환산한다
            score = float(1.0 / (1.0 + np.exp(-lg[mk])).mean())
            if mk.shape != (h, w):       # 마스크가 모델 입력 해상도로 나오는 경우 원본으로 되돌린다
                mk = cv2.resize(mk.astype(np.uint8), (w, h),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
            bb = sa._bbox(mk)
            if not bb or (bb[2] - bb[0]) <= 10 or (bb[3] - bb[1]) <= 10:
                continue                 # 점만 남은 마스크는 박스로 못 쓴다
            results.append({"idx": nums[fpos], "box": _norm_box(bb, w, h),
                            "score": round(score, 3)})
        j.update(stage="done", running=False, results=results, note=f"후보 {len(results)}개")
    except Exception as e:   # noqa: BLE001 - SAM2 미설치·체크포인트 없음도 여기로 온다
        sa.logger.exception("gt propagate 오류")
        j.update(stage="error", running=False,
                 error=f"SAM2 를 쓸 수 없습니다: {type(e).__name__}: {e}"[:300])
    finally:
        state = predictor = None
        sa._BUSY["on"] = False
        _free()


def start_propagate(part, stem, idx, box, span=5):
    stem = _safe_name(stem)
    if not stem:
        return {"error": "stem 이 올바르지 않습니다."}
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return {"error": "box 는 [cx, cy, w, h] 정규화 값이어야 합니다."}
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        return {"error": "idx 가 정수가 아닙니다."}
    span = max(1, min(int(span or 5), 30))
    if not sa.CKPT.exists():
        # 스레드를 띄우고 나서 실패하면 사람이 폴링을 기다려야 한다. 아는 실패는 즉시 알린다
        return {"error": f"SAM2 를 쓸 수 없습니다: 체크포인트 없음({sa.CKPT})"}
    with sa._LOCK:
        if sa._BUSY["on"]:
            return {"error": "이미 실행 중입니다."}
        sa._BUSY["on"] = True
    jid = uuid.uuid4().hex[:8]
    sa.JOBS[jid] = {"stage": "propagate", "running": True, "error": None, "done": 0,
                    "total": span * 2, "note": "준비 중", "part": part}
    threading.Thread(target=_run_propagate, args=(jid, stem, idx, list(box), span),
                     daemon=True).start()
    return {"job": jid}


# ==================== AP 계산(자체 구현) ====================
def _iou(a, b):
    """XYXY 두 박스의 IoU = 교집합 면적 / 합집합 면적."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _match(dets, gts, iou_thr):
    """conf 내림차순 그리디 매칭.

    dets : [(conf, img, (x1,y1,x2,y2))]  한 클래스의 검출 전부
    gts  : {img: [(x1,y1,x2,y2), ...]}   그 클래스 GT 전부
    반환 : [(conf, img, is_tp)]  conf 내림차순 순서 그대로

    규칙
      - 신뢰도 높은 검출부터 처리한다(낮은 검출이 GT 를 먼저 가져가지 못하게)
      - 같은 이미지에서 아직 안 쓰인 GT 중 IoU 가 가장 큰 것을 고르고, IoU >= 임계면 TP
      - GT 하나는 한 번만 매칭된다. 같은 GT 를 가리키는 두 번째 검출은 중복이라 FP
    """
    used = {k: [False] * len(v) for k, v in gts.items()}
    flags = []
    for conf, img, box in sorted(dets, key=lambda d: -d[0]):
        best, bi = 0.0, -1
        for i, g in enumerate(gts.get(img, [])):
            if used[img][i]:
                continue
            v = _iou(box, g)
            if v > best:
                best, bi = v, i
        if bi >= 0 and best >= iou_thr:
            used[img][bi] = True
            flags.append((conf, img, True))
        else:
            flags.append((conf, img, False))
    return flags


def _ap(flags, n_gt):
    """all-point interpolation AP 와 PR 배열.

    누적 계산(conf 내림차순 i 번째 검출까지)
        TP_i = sum(tp_1 .. tp_i),   FP_i = i - TP_i
        P_i  = TP_i / (TP_i + FP_i) = TP_i / i
        R_i  = TP_i / N_gt
    포락선(정밀도를 단조 감소로 보정)
        P~_i = max_{j >= i} P_j
    면적(모든 점 보간)
        AP = SUM_i (R_i - R_{i-1}) * P~_i        , R_0 = 0
    VOC2010 이후 방식 = COCO 와 같은 정의다. 11점 보간이 아니다.
    """
    if n_gt <= 0 or not flags:
        return 0.0, [0.0], [0.0]
    tp = np.array([1.0 if f[2] else 0.0 for f in flags])
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1.0 - tp)
    rec = ctp / float(n_gt)
    pre = ctp / np.maximum(ctp + cfp, 1e-12)
    # 앞에 (R=0, P=1), 뒤에 (R=마지막, P=0) 을 붙여 면적이 열려 있지 않게 한다
    mrec = np.concatenate(([0.0], rec, [rec[-1]]))
    mpre = np.concatenate(([1.0], pre, [0.0]))
    for i in range(mpre.size - 1, 0, -1):    # 오른쪽에서 왼쪽으로 최대값 전파 = 포락선
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    k = np.where(mrec[1:] != mrec[:-1])[0]   # recall 이 실제로 늘어난 지점만 면적에 기여
    ap = float(np.sum((mrec[k + 1] - mrec[k]) * mpre[k + 1]))
    return ap, rec.tolist(), pre.tolist()


def _thin(rec, pre, cap=200):
    """PR 배열을 cap 점 이하로 줄인다(검출 수천 개면 JSON 이 커지고 SVG 가 느려진다)."""
    n = len(rec)
    if n <= cap:
        return [round(v, 5) for v in rec], [round(v, 5) for v in pre]
    k = [round(i * (n - 1) / (cap - 1)) for i in range(cap)]
    return [round(rec[i], 5) for i in k], [round(pre[i], 5) for i in k]


def _at_conf(flags, n_gt, thr):
    """보고 임계에서의 TP/FP/FN/P/R/F1.

    flags 는 이미 conf 내림차순 그리디 매칭 결과다. 임계 위쪽만 세면 되는 이유는
    낮은 conf 검출이 GT 를 먼저 가져가는 일이 구조적으로 없기 때문이다.
    """
    tp = sum(1 for c, _, ok in flags if c >= thr and ok)
    fp = sum(1 for c, _, ok in flags if c >= thr and not ok)
    fn = max(0, n_gt - tp)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / n_gt if n_gt else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return tp, fp, fn, p, r, f1


# ==================== 2.9 채점 ====================
def _eval_runs():
    if not EVAL_ROOT.exists():
        return []
    return sorted((d.name for d in EVAL_ROOT.iterdir()
                   if d.is_dir() and (d / "summary.json").exists()), reverse=True)


def _load_gt(part):
    """부품의 GT. {이름: {"path":..., "boxes":[XYXY...]}}.

    라벨이 없거나 박스가 0개인 이미지는 뺀다. 그런 이미지는 '부품이 없는 배경'이 아니라
    '아직 라벨하지 않은 프레임'이라서, 넣으면 FP 만 늘어 수치가 왜곡된다.
    """
    imgs, lbls = _gt_dirs(part)
    out = {}
    if not imgs.exists() or not lbls.exists():
        return out
    for ip in sorted(imgs.glob("*.jpg")):
        lp = lbls / f"{ip.stem}.txt"
        if not lp.exists():
            continue
        im = _rd(ip)
        if im is None:
            continue
        h, w = im.shape[:2]
        boxes = []
        for ln in lp.read_text(encoding="utf-8").splitlines():
            f = ln.split()
            if len(f) < 5:
                continue
            cx, cy, bw, bh = (float(v) for v in f[1:5])
            boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h))
        if not boxes:
            continue
        out[ip.stem] = {"path": ip, "boxes": boxes}
    return out


def _split_name(name):
    """GT 파일 stem '<영상stem>_<00060>' -> ('<영상stem>', 60). 페이지2 썸네일이 이 값을 쓴다."""
    s, _, tail = name.rpartition("_")
    try:
        return s, int(tail)
    except ValueError:
        return name, 0


def _dump_series(out, series, report_conf):
    """AP 를 만든 누적 계열을 부품별 CSV 로 남긴다. 채점할 때마다 자동 실행된다.

    왜 필요한가
      화면 표의 TP/FP/FN 은 conf 0.70 한 지점 값이라 그 세 숫자로 AP 를 역산할 수 없다.
      심사원이 적분 과정을 검산하겠다고 하면 이 파일을 열어 보여주면 된다.
      한 줄이 검출 1건이고, 누적열을 그대로 적분하면 AP 가 나온다.

    열 의미
      i         신뢰도 내림차순 순번
      conf      그 검출의 신뢰도
      match     IoU 0.5 로 GT 와 맞았나 (TP/FP)
      image     그 검출이 나온 GT 이미지 이름
      cum_tp    i 번째까지 누적 TP
      cum_fp    i 번째까지 누적 FP
      precision cum_tp / i
      recall    cum_tp / 전체 GT 수
      at_report 이 줄이 보고 임계(conf 0.70) 안에 들어가는가
    """
    for part, d in series.items():
        rows = ["i,conf,match,image,cum_tp,cum_fp,precision,recall,at_report"]
        ctp = cfp = 0
        n_gt = d["n_gt"]
        for i, (cf, img, ok) in enumerate(d["flags"], 1):
            ctp += 1 if ok else 0
            cfp += 0 if ok else 1
            rows.append("%d,%.6f,%s,%s,%d,%d,%.6f,%.6f,%s" % (
                i, cf, "TP" if ok else "FP", img, ctp, cfp,
                ctp / i, (ctp / n_gt) if n_gt else 0.0,
                "Y" if cf >= report_conf else "N"))
        rows.append("")
        rows.append("# 전체 GT 수,%d" % n_gt)
        rows.append("# AP@0.5 (all-point interpolation),%.6f" % d["ap"])
        rows.append("# 보고 임계 conf,%.2f" % report_conf)
        (out / ("curve_%s.csv" % _safe_name(part))).write_text(
            "\n".join(rows), encoding="utf-8-sig")   # utf-8-sig: 엑셀에서 한글이 안 깨지게


def _run_eval(job_id, parts, iou, report_conf, weights):
    j = sa.JOBS[job_id]
    try:
        w, label = _resolve_weights(weights)
        if not w:
            raise RuntimeError(label)
        from ultralytics import YOLO
        model = YOLO(w)
        name_to_idx = {v: k for k, v in model.names.items()}

        gt_by_part = {p: _load_gt(p) for p in parts}
        total = sum(len(v) for v in gt_by_part.values())
        if total == 0:
            raise RuntimeError("GT 가 한 장도 없습니다. 먼저 GT 를 만드세요.")
        j.update(stage="eval", total=total, done=0, note="추론 시작")

        # 같은 프레임이 여러 부품 폴더에 복사돼 있다. 파일 이름이 곧 프레임 식별자이므로
        # 이름으로 캐시해 한 번만 추론한다(5종이면 최대 5배 절약)
        pred_cache = {}
        done = 0
        classes, missing = [], []
        series = {}          # 부품 -> AP 를 만든 누적 계열 원자료(CSV 로 남긴다)
        for part in parts:
            gts = gt_by_part[part]
            if not gts:
                # GT 0장. 계약서대로 ap 0 으로 평균에 포함하고 missing 에 넣는다
                missing.append(part)
                classes.append({"part": part, "in_model": part in name_to_idx,
                                "n_images": 0, "n_gt": 0, "ap": 0.0,
                                "tp": 0, "fp": 0, "fn": 0,
                                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                                "pr": {"recall": [0.0], "precision": [0.0]},
                                "conf_curve": [], "worst": []})
                continue
            cid = name_to_idx.get(part)          # 모델이 모르는 부품이면 검출이 0 -> 전부 FN
            dets, gt_map, n_gt = [], {}, 0
            for nm, g in gts.items():
                gt_map[nm] = g["boxes"]
                n_gt += len(g["boxes"])
                if cid is not None:
                    if nm not in pred_cache:
                        r = model.predict(source=_rd(g["path"]), conf=CURVE_CONF,
                                          imgsz=EVAL_IMGSZ, verbose=False)[0]
                        pred_cache[nm] = list(zip(
                            r.boxes.conf.cpu().numpy().tolist(),
                            r.boxes.cls.cpu().numpy().astype(int).tolist(),
                            r.boxes.xyxy.cpu().numpy().tolist()))
                    # 그 클래스 검출만 쓴다. 다른 클래스 검출은 무시(계약서 0장)
                    for cf, cl, bb in pred_cache[nm]:
                        if cl == cid:
                            dets.append((float(cf), nm, tuple(float(v) for v in bb)))
                done += 1
                if done % 5 == 0 or done == total:
                    j.update(done=done, note=f"{done}/{total} · {part}")

            flags = _match(dets, gt_map, iou)
            ap, rec, pre = _ap(flags, n_gt)
            tp, fp, fn, p, r, f1 = _at_conf(flags, n_gt, report_conf)
            # 심사원이 AP 적분을 직접 검산하겠다고 하면 이 계열을 보여준다.
            # 화면의 TP/FP/FN 표는 conf 0.70 한 지점이라 그 세 숫자로는 AP 를 되짚을 수 없다.
            # 링크를 누르지 않아도 채점할 때마다 자동으로 파일이 남게 한다.
            series[part] = {"flags": flags, "n_gt": n_gt, "ap": ap}
            # 곡선 왼쪽 끝 (R=0, P=1) 을 붙여 준다. recall 은 누적이라 이미 오름차순이다
            rec_t, pre_t = _thin([0.0] + rec, [1.0] + pre)

            # conf 곡선: 임계를 올릴 때 P/R 이 어떻게 움직이는지(운영 임계 선택 근거)
            curve = []
            for t in [round(x * 0.05, 2) for x in range(1, 20)]:
                _tp, _fp, _fn, _p, _r, _f1 = _at_conf(flags, n_gt, t)
                curve.append({"conf": t, "p": round(_p, 4), "r": round(_r, 4)})

            # 실패 사례: 보고 임계에서 놓친(fn) 프레임 우선, 그 다음 헛검출(fp)
            per_img = {nm: {"tp": 0, "fp": 0} for nm in gt_map}
            for cf, nm, ok in flags:
                if cf < report_conf:
                    continue
                per_img.setdefault(nm, {"tp": 0, "fp": 0})["tp" if ok else "fp"] += 1
            rows = []
            for nm, v in per_img.items():
                miss = len(gt_map.get(nm, [])) - v["tp"]
                if miss > 0 or v["fp"] > 0:
                    st, ix = _split_name(nm)
                    rows.append((-miss, -v["fp"], st, ix, "fn" if miss > 0 else "fp"))
            rows.sort(key=lambda x: (x[0], x[1]))
            worst = [{"stem": st, "idx": ix, "reason": rs} for _m, _f, st, ix, rs in rows[:6]]

            classes.append({"part": part, "in_model": cid is not None,
                            "n_images": len(gts), "n_gt": n_gt, "ap": round(ap, 4),
                            "tp": tp, "fp": fp, "fn": fn,
                            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
                            "pr": {"recall": rec_t, "precision": pre_t},
                            "conf_curve": curve, "worst": worst})

        # mAP = 요청한 부품 전부의 AP 산술평균. GT 없는 부품도 ap 0 으로 들어간다
        # (5종 평균이어야 심사 규격과 맞는다)
        m50 = sum(c["ap"] for c in classes) / len(classes) if classes else 0.0
        run = datetime.now().strftime("%y%m%d_%H%M%S")
        result = {"run": run, "weights": str(w), "model_label": label,
                  "iou": iou, "report_conf": report_conf, "pass_threshold": PASS_THRESHOLD,
                  "map50": round(m50, 4), "map50_pct": round(m50 * 100, 1),
                  "verdict": "pass" if m50 * 100 >= PASS_THRESHOLD else "fail",
                  "classes": classes, "missing": missing}
        out = EVAL_ROOT / run
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
        _dump_series(out, series, report_conf)
        j.update(stage="done", running=False, run=run, result=result,
                 note=f"mAP@0.5 = {result['map50_pct']}")
    except Exception as e:   # noqa: BLE001
        sa.logger.exception("gt eval 오류")
        j.update(stage="error", running=False, error=f"{type(e).__name__}: {e}"[:300])
    finally:
        sa._BUSY["on"] = False
        _free()


def start_eval(parts=None, iou=0.5, report_conf=REPORT_CONF, weights=None):
    parts = [p for p in (_safe_name(x) for x in (parts or GT_PARTS)) if p]
    if not parts:
        return {"error": "채점할 부품이 없습니다."}
    try:
        iou = float(0.5 if iou is None else iou)
        report_conf = float(REPORT_CONF if report_conf is None else report_conf)
    except (TypeError, ValueError):
        return {"error": "iou / report_conf 가 숫자가 아닙니다."}
    if not 0 < iou < 1 or not 0 < report_conf < 1:
        return {"error": "iou · report_conf 는 0 과 1 사이여야 합니다."}
    with sa._LOCK:
        if sa._BUSY["on"]:
            return {"error": "이미 실행 중입니다."}
        sa._BUSY["on"] = True
    jid = uuid.uuid4().hex[:8]
    sa.JOBS[jid] = {"stage": "eval", "running": True, "error": None,
                    "done": 0, "total": 0, "note": "준비 중"}
    threading.Thread(target=_run_eval, args=(jid, parts, iou, report_conf, weights),
                     daemon=True).start()
    return {"job": jid}


# ==================== 2.10 채점 결과 재조회 ====================
def eval_result(run=None):
    runs = _eval_runs()
    if not runs:
        return {"error": "채점 기록이 없습니다. 먼저 채점을 실행하세요.", "runs": []}
    run = _safe_name(run) or runs[0]
    fp = EVAL_ROOT / run / "summary.json"
    if not fp.exists():
        return {"error": f"채점 기록이 없습니다: {run}", "runs": runs}
    d = json.loads(fp.read_text(encoding="utf-8"))
    d["runs"] = runs
    return d


# ==================== 자체 검사 ====================
# ==================== 시나리오1 전용: 5종 x 10컷 수집 + 탭 마스크 ====================
# 이 화면은 일회용이다. 영상을 고르거나 임계값을 만지는 기능을 두지 않는다.
# 심사 기준(시나리오1)이 conf 0.7 이므로 수집도 그 기준으로만 한다.
PICK_CONF = REPORT_CONF        # 0.70. 심사 기준과 같은 값으로 프레임을 고른다
PICK_PER_PART = 10             # 종별 10컷


def _run_collect(job_id, step):
    """평가영상 전체를 conf 0.70 으로 훑어 5종별 10컷을 고른다.
    사람이 영상을 고르지 않는다. 있는 영상을 전부 본다."""
    j = sa.JOBS[job_id]
    try:
        stems = [v["stem"] for v in list_videos().get("videos", [])]
        if not stems:
            raise RuntimeError("평가영상이 없습니다: " + str(GT_VIDEO_DIR))
        w, label = _resolve_weights(None)
        if not w:
            raise RuntimeError(label)
        from ultralytics import YOLO
        model = YOLO(w)
        names = model.names

        todo = []
        for s in stems:
            j.update(stage="collect", note=s + " 프레임 준비")
            prepare(s, step)
            todo += [(s, p) for p in _cached(s)]
        total = len(todo)
        if not total:
            raise RuntimeError("자른 프레임이 없습니다.")
        j.update(stage="collect", total=total, done=0, note="탐지 시작")

        cand = {p: [] for p in GT_PARTS}          # 종별 후보 프레임
        for k, (s, p) in enumerate(todo, 1):
            im = _rd(p)
            if im is None:
                continue
            r = model.predict(source=im, conf=PICK_CONF, imgsz=EVAL_IMGSZ, verbose=False)[0]
            best = {}
            for b, cf, cl in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                 r.boxes.cls.cpu().numpy()):
                nm = names[int(cl)] if int(cl) in names else str(int(cl))
                if nm not in cand:
                    continue
                cf = float(cf)
                if nm not in best or cf > best[nm]["conf"]:
                    ih, iw = im.shape[:2]
                    best[nm] = {"stem": s, "idx": int(p.stem), "conf": round(cf, 4),
                                "bbox": [int(v) for v in b], "wh": [int(iw), int(ih)]}
            for nm, d in best.items():
                cand[nm].append(d)
            if k % 5 == 0 or k == total:
                j.update(done=k, note=str(k) + "/" + str(total) + " · " + s)

        # 후보를 추려서 주지 않는다. 사람이 전부 보고 고른다.
        # 자동 선정(_spread) 을 쓰다 폐기했다: 모델이 shim 과 r-ring 을 같은 물체에 겹쳐
        # 잡아 놓은 것도 그대로 10장에 들어갔고, 뽑힌 장면이 서로 비슷해 다양성도 낮았다.
        for v in cand.values():
            v.sort(key=lambda x: (x["stem"], x["idx"]))
        run = datetime.now().strftime("%y%m%d_%H%M%S")
        out = SCAN_ROOT / run
        out.mkdir(parents=True, exist_ok=True)
        payload = {"run": run, "weights": str(w), "label": label, "conf": PICK_CONF,
                   "step": step, "per_part": PICK_PER_PART, "stems": stems,
                   "found": {nm: len(v) for nm, v in cand.items()},
                   "cands": cand}
        (out / "collect.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
        j.update(stage="done", running=False, run=run,
                 found=payload["found"], note="수집 완료")
    except Exception as e:   # noqa: BLE001 - 서버가 죽지 않게 하고 원인은 화면에 띄운다
        sa.logger.exception("gt collect 오류")
        j.update(stage="error", running=False, error=(type(e).__name__ + ": " + str(e))[:300])
    finally:
        sa._BUSY["on"] = False
        _free()


def start_collect(step=15):
    with sa._LOCK:
        if sa._BUSY["on"]:
            return {"error": "이미 실행 중입니다. 끝난 뒤 다시 시도하세요."}
        sa._BUSY["on"] = True
    jid = uuid.uuid4().hex[:8]
    sa.JOBS[jid] = {"stage": "start", "running": True, "error": None}
    sa._ACTIVE.update(job=jid, kind="gt_collect", session="gt")
    threading.Thread(target=_run_collect, args=(jid, int(step)), daemon=True).start()
    return {"job": jid}


def _collect_runs():
    if not SCAN_ROOT.exists():
        return []
    return sorted((d.name for d in SCAN_ROOT.iterdir()
                   if d.is_dir() and (d / "collect.json").exists()), reverse=True)


def collect_result(run=None):
    """수집 결과 + 종별 GT 저장 현황을 함께 준다(화면이 한 번에 다 그린다)."""
    runs = _collect_runs()
    if not runs:
        return {"error": "수집 기록이 없습니다. 먼저 수집을 실행하세요.", "runs": []}
    run = _safe_name(run) if run else runs[0]
    f = SCAN_ROOT / run / "collect.json"
    if not f.exists():
        return {"error": "없는 실행입니다: " + run, "runs": runs}
    d = json.loads(f.read_text(encoding="utf-8"))
    d["runs"] = runs
    saved = {}
    for part in GT_PARTS:
        gi, gl = _gt_dirs(part)
        saved[part] = sorted(p.stem for p in gl.glob("*.txt")) if gl.exists() else []
    d["saved"] = saved
    d["sel"] = _load_select(run)
    return d


def label_of(part, stem, idx):
    """저장한 GT 한 장의 박스(정규화 cx cy w h). 화면이 다시 열어 고칠 수 있게 한다."""
    part, stem = _safe_name(part), _safe_name(stem)
    if part not in GT_PARTS or not stem:
        return {"error": "part / stem 이 올바르지 않습니다."}
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        return {"error": "idx 가 정수가 아닙니다."}
    lp = _gt_dirs(part)[1] / f"{stem}_{idx:05d}.txt"
    if not lp.exists():
        return {"boxes": []}
    boxes = []
    for ln in lp.read_text(encoding="utf-8").splitlines():
        t = ln.split()
        if len(t) >= 5:
            try:
                boxes.append([float(t[1]), float(t[2]), float(t[3]), float(t[4])])
            except ValueError:
                continue
    return {"boxes": boxes}


def _select_path(run):
    return SCAN_ROOT / _safe_name(run) / "select.json"


def _load_select(run):
    """부품별로 사람이 고른 프레임 목록. 키는 "<stem>|<프레임번호>"."""
    f = _select_path(run)
    if not f.exists():
        return {}
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    return {k: list(v) for k, v in d.items() if isinstance(v, list)}


def set_select(run, part, keys):
    """한 부품의 선택 목록을 통째로 덮어쓴다(화면이 항상 전체를 보내온다)."""
    run = _safe_name(run)
    if not (SCAN_ROOT / run).is_dir():
        return {"error": "없는 실행입니다: " + str(run)}
    if part not in GT_PARTS:
        return {"error": "모르는 부품입니다: " + str(part)}
    if not isinstance(keys, list):
        return {"error": "선택 목록이 배열이 아닙니다."}
    ok = []
    for k in keys[:400]:            # 화면 실수로 수천 개가 와도 파일이 부풀지 않게 한다
        if isinstance(k, str) and "|" in k:
            stem, _, idx = k.partition("|")
            if _safe_name(stem) and idx.isdigit():
                ok.append(_safe_name(stem) + "|" + str(int(idx)))
    sel = _load_select(run)
    sel[part] = sorted(set(ok))
    _select_path(run).write_text(json.dumps(sel, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
    return {"ok": True, "part": part, "n": len(sel[part])}


def tap_box(stem, idx, points):
    """탭한 점으로 SAM2 마스크를 만들고 그 마스크의 외곽 박스를 돌려준다.

    학습 화면의 '입력 마스크' 와 같은 엔진을 쓴다. 다만 이쪽 프레임은 평가 트리
    (results/gt_frames)에 있어 sa.mask_preview 가 쓰는 영상키로는 찾지 못한다.
    그래서 프레임을 직접 읽어 같은 절차를 밟는다.

    points: [[rx, ry, label]] 0~1 비율. label 1=포함점, 0=제외점.
    """
    stem = _safe_name(stem)
    fp = FRAMES_ROOT / stem / ("%05d.jpg" % int(idx))
    if not fp.exists():
        return {"error": "프레임이 없습니다: " + stem + " #" + str(idx)}
    if not points:
        return {"error": "점이 없습니다."}
    im = _rd(fp)
    if im is None:
        return {"error": "프레임을 읽지 못했습니다."}
    h, w = im.shape[:2]
    pts = [[min(max(float(x), 0.0), 1.0), min(max(float(y), 0.0), 1.0), int(l)]
           for x, y, l in points]
    try:
        with sa._MASK_LOCK:            # 학습·전파와 SAM2 를 동시에 물지 않는다
            pred = sa._img_predictor()
            with torch.inference_mode(), torch.autocast(sa.DEV, dtype=torch.bfloat16):
                pred.set_image(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                arr = np.array([[rx * w, ry * h] for rx, ry, _ in pts], dtype=np.float32)
                lab = np.array([int(l) for l in (p[2] for p in pts)], dtype=np.int32)
                masks, scores, _ = pred.predict(point_coords=arr, point_labels=lab,
                                                multimask_output=True)
    except Exception as e:   # noqa: BLE001
        sa.logger.exception("tap_box 실패")
        return {"error": (type(e).__name__ + ": " + str(e))[:200]}
    m = masks[int(np.argmax(scores))].astype(bool)
    bb = sa._bbox(m)
    if not bb:
        return {"error": "마스크가 비었습니다. 부품 안쪽을 다시 눌러 보세요."}
    area = float(m.sum()) / (w * h)
    vis = sa._overlay(im, m)
    rr, bt, ew = sa._ov_sizes(w, h)
    # 박스는 서버가 그리지 않는다. 화면에서 사람이 끌어 조절하기 때문에, 여기서 그려 보내면
    # 조절 중인 박스와 구워진 박스가 겹쳐 두 개로 보인다. 좌표(bbox·box)만 넘기고 그리기는 화면이 한다.
    for rx, ry, l in pts:
        cv2.circle(vis, (int(rx * w), int(ry * h)), rr, (255, 60, 0) if l else (0, 0, 255), -1)
        cv2.circle(vis, (int(rx * w), int(ry * h)), rr, (255, 255, 255), ew)
    _free()
    return {"combo": sa._b64(vis), "bbox": bb, "box": _norm_box(bb, w, h),
            "area_frac": round(area, 4),
            "verdict": "over" if area > 0.5 else "ok"}


def _selfcheck():
    """AP 계산이 깨지면 여기서 터진다. GPU·모델 없이 도는 순수 계산 검사.
    실행: venv/Scripts/python.exe backend/autolearning/gt_eval.py"""
    assert abs(_iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-9
    assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(_iou((0, 0, 10, 10), (5, 0, 15, 10)) - (50 / 150)) < 1e-9

    # 완벽 검출 1개 -> AP 1
    gts = {"a": [(0, 0, 10, 10)]}
    ap, _, _ = _ap(_match([(0.9, "a", (0, 0, 10, 10))], gts, 0.5), 1)
    assert abs(ap - 1.0) < 1e-6, ap

    # GT 하나에 검출 둘 -> 두 번째는 중복이라 FP. 그래도 최대 recall 구간 P=1 이므로 AP 1
    flags = _match([(0.9, "a", (0, 0, 10, 10)), (0.8, "a", (0, 0, 10, 10))], gts, 0.5)
    assert [f[2] for f in flags] == [True, False], flags
    assert abs(_ap(flags, 1)[0] - 1.0) < 1e-6

    # GT 2개 중 1개만 맞춤 -> AP 0.5
    gts2 = {"a": [(0, 0, 10, 10)], "b": [(0, 0, 10, 10)]}
    ap, rec, pre = _ap(_match([(0.9, "a", (0, 0, 10, 10))], gts2, 0.5), 2)
    assert abs(ap - 0.5) < 1e-6, ap
    assert rec == [0.5] and pre == [1.0], (rec, pre)

    # 입력 순서가 섞여도 conf 내림차순으로 다시 세워야 한다(FP 가 먼저 오면 AP 가 떨어진다)
    flags = _match([(0.4, "a", (0, 0, 10, 10)), (0.9, "b", (99, 99, 109, 109))], gts2, 0.5)
    assert [f[2] for f in flags] == [False, True], flags
    assert abs(_ap(flags, 2)[0] - 0.25) < 1e-6   # R=0.5 지점 포락선 P=0.5 -> 0.5 * 0.5

    # IoU 임계 미달은 FP
    assert _match([(0.9, "a", (7, 0, 17, 10))], gts, 0.5)[0][2] is False

    # GT 0 장이면 AP 0(평균에는 포함된다)
    assert _ap([], 0)[0] == 0.0

    # 보고 임계 컷: 0.3 검출은 conf 0.7 보고에서 빠지고 그 GT 는 FN 이 된다
    flags = _match([(0.9, "a", (0, 0, 10, 10)), (0.3, "b", (0, 0, 10, 10))], gts2, 0.5)
    tp, fp, fn, p, r, f1 = _at_conf(flags, 2, 0.7)
    assert (tp, fp, fn) == (1, 0, 1), (tp, fp, fn)
    assert abs(p - 1.0) < 1e-9 and abs(r - 0.5) < 1e-9 and abs(f1 - 2 / 3) < 1e-9

    assert _norm_box((0, 0, 100, 50), 200, 100) == [0.25, 0.25, 0.5, 0.5]
    assert _split_name("20260715_144424_00060") == ("20260715_144424", 60)
    assert _safe_name("../etc") is None and _safe_name("a/b") is None
    assert _safe_name(" seal_support ") == "seal_support"
    print("gt_eval 자체 검사 통과")


if __name__ == "__main__":
    _selfcheck()
