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
import shutil
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
import torch

import config
import autolabel   # _frames, FRAME_CACHE (프레임 소스·캐시 공용)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
CKPT = config.BASE_DIR / "models" / "sam2" / "sam2.1_hiera_base_plus.pt"
RESULTS = config.BASE_DIR / "results"
VAL_CONF = 0.4
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
    r = max(6, round(w / 82))            # 점 반지름
    bt = max(3, round(w / 95))           # 박스 두께(더 굵게)
    ew = max(2, round(w / 400))          # 점 흰 테두리
    vis = _overlay(im, m)
    if bb:
        cv2.rectangle(vis, (bb[0], bb[1]), (bb[2], bb[3]), (0, 165, 255), bt)
    for rx, ry, lab in points:   # 마스크가 초록이라 부품점은 파랑(BGR), 제외점은 빨강
        cv2.circle(vis, (int(rx * w), int(ry * h)), r, (255, 60, 0) if lab else (0, 0, 255), -1)
        cv2.circle(vis, (int(rx * w), int(ry * h)), r, (255, 255, 255), ew)
    gc.collect(); torch.cuda.empty_cache()
    return {"combo": _b64(vis), "area_frac": round(float(m.sum()) / (w * h), 4), "bbox": bb}


def free_sam2():
    _IMG.clear()
    gc.collect(); torch.cuda.empty_cache()


# ==================== (2) 영상 전파 → 라벨 생성 ====================
def _run_propagate(job_id, src, shots):
    """shots=[[frame_idx,[[rx,ry,lab],...]],...] → SAM2 전파 → train 라벨 + train_box(박스표기) + tap(탭확인)."""
    j = JOBS[job_id]
    try:
        from sam2.build_sam import build_sam2_video_predictor
        fs = autolabel._frames(src)
        cache_dir = autolabel.FRAME_CACHE / src
        j.update(stage="load", total=len(fs))
        predictor = build_sam2_video_predictor(CFG, str(CKPT), device=DEV)
        h, w = _rd(fs[0]).shape[:2]
        runid = datetime.now().strftime("%y%m%d_%H%M%S")   # 실행마다 폴더 하나 (덮어쓰지 않음)
        work = RESULTS / f"{src}_{runid}"                  # 영상_실행시각
        ref_d = work / "tap"    # 탭 프레임: 점+마스크+박스를 한 장에
        seed_i, seed_l = work / "train" / "images", work / "train" / "labels"
        box_d = work / "train_box"   # 학습 이미지에 바운딩박스 그린 확인용
        for d in (ref_d, seed_i, seed_l, box_d, work / "unlabeled"):
            d.mkdir(parents=True, exist_ok=True)

        j.update(stage="propagate")
        with torch.inference_mode(), torch.autocast(DEV, dtype=torch.bfloat16):
            state = predictor.init_state(video_path=str(cache_dir), offload_video_to_cpu=True,
                                         offload_state_to_cpu=True)
            for fi, pts in shots:
                p = np.array([[rx * w, ry * h] for rx, ry, _ in pts], dtype=np.float32)
                l = np.array([int(lab) for *_, lab in pts], dtype=np.int32)
                predictor.add_new_points_or_box(inference_state=state, frame_idx=int(fi), obj_id=1,
                                                points=p, labels=l)
            masks = {}
            for fidx, _ids, logits in predictor.propagate_in_video(state):
                m = logits[0].cpu().numpy()
                masks[fidx] = (m[0] if m.ndim == 3 else m) > 0.0
        del predictor, state
        free_sam2()

        # tap (탭한 프레임: 마스크 초록 + 박스 주황 + 점(파랑=부품/빨강=제외) 한 장에)
        for fi, pts in shots:
            mk = masks.get(int(fi))
            base = _rd(fs[int(fi)])
            has_mk = mk is not None and mk.shape == (h, w)
            vis = _overlay(base, mk) if has_mk else base
            bb = _bbox(mk) if has_mk else None
            if bb:
                cv2.rectangle(vis, (bb[0], bb[1]), (bb[2], bb[3]), (0, 165, 255), 3)
            for rx, ry, lab in pts:
                cv2.circle(vis, (int(rx * w), int(ry * h)), 10, (255, 60, 0) if lab else (0, 0, 255), -1)
                cv2.circle(vis, (int(rx * w), int(ry * h)), 10, (255, 255, 255), 2)
            cv2.imwrite(str(ref_d / f"{Path(fs[int(fi)]).stem}.jpg"), vis)

        # 전 프레임: 마스크→박스 라벨(train) + 박스표기(train_box) / 빈 마스크=unlabeled
        labeled = unlabeled = 0
        for i, p in enumerate(fs):
            mk = masks.get(i)
            im = _rd(p)
            if mk is not None and mk.shape != (h, w):
                mk = cv2.resize(mk.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            bb = _bbox(mk) if mk is not None else None
            if bb and (bb[2] - bb[0]) > 10 and (bb[3] - bb[1]) > 10:
                cv2.imwrite(str(seed_i / f"{p.stem}.jpg"), im)   # 학습용 원본
                cx, cy = (bb[0] + bb[2]) / 2 / w, (bb[1] + bb[3]) / 2 / h
                bw, bh = (bb[2] - bb[0]) / w, (bb[3] - bb[1]) / h
                (seed_l / f"{p.stem}.txt").write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")
                boxed = im.copy()   # 박스 그린 확인용
                cv2.rectangle(boxed, (bb[0], bb[1]), (bb[2], bb[3]), (0, 165, 255), 3)
                cv2.putText(boxed, "part", (bb[0], max(bb[1] - 6, 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                cv2.imwrite(str(box_d / f"{p.stem}.jpg"), boxed)
                labeled += 1
            else:
                cv2.imwrite(str(work / "unlabeled" / f"{p.stem}.jpg"), im)
                unlabeled += 1
            j.update(done=i + 1)

        samples = [_b64(_overlay(_rd(fs[i]), masks[i]) if masks.get(i) is not None and masks[i].shape == (h, w)
                        else _rd(fs[i]), w=300)
                   for i in range(0, len(fs), max(1, len(fs) // 8))][:8]
        (work / "meta.json").write_text(json.dumps(
            {"run": runid, "src": src, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             "shots": [[int(fi), pts] for fi, pts in shots], "labeled": labeled,
             "unlabeled": unlabeled, "total": len(fs), "trained": False},
            ensure_ascii=False, indent=2), encoding="utf-8")
        j.update(stage="done", running=False, labeled=labeled, unlabeled=unlabeled,
                 total=len(fs), work=str(work), run=runid, samples=samples,
                 ref_masks=[_b64(_rd(p), w=300) for p in sorted(ref_d.glob("*.jpg"))])
    except Exception as e:
        j.update(stage="error", error=f"{type(e).__name__}: {e}", running=False)
    finally:
        _BUSY["on"] = False
        gc.collect(); torch.cuda.empty_cache()


def start_propagate(src, shots):
    with _LOCK:
        if _BUSY["on"]:
            return {"error": "이미 실행 중입니다. 끝난 뒤 다시 시도하세요."}
        if not shots:
            return {"error": "탭한 참조가 없습니다."}
        _BUSY["on"] = True
    jid = uuid.uuid4().hex[:8]
    JOBS[jid] = {"stage": "start", "done": 0, "total": 0, "running": True, "error": None}
    threading.Thread(target=_run_propagate, args=(jid, src, shots), daemon=True).start()
    return {"job": jid}


# ==================== (3) 학습 ====================
def _labels_dir(work):
    """실행 폴더의 학습 라벨 디렉토리 (train, 구 predict/seed 하위호환)."""
    for sub in ("train", "predict", "seed"):   # 신(train)·구(predict/seed) 하위호환
        if (work / sub / "images").exists():
            return work / sub
    return work / "train"


def _resolve_run(src, run):
    """실행 폴더 = results/<영상>_<실행시각>/. run 지정 없으면 그 영상의 가장 최근."""
    if run:
        return RESULTS / f"{src}_{run}"
    runs = sorted([d for d in RESULTS.glob(f"{src}_*")
                   if any((d / s / "images").exists() for s in ("train", "predict", "seed"))])
    return runs[-1] if runs else RESULTS / f"{src}_none"


def list_labeled():
    """라벨 생성된 실행 목록을 영상별로. 학습 세트 구성용 (train 영상 고르기)."""
    out = {}
    for stem in sorted({p.stem for p in autolabel._videos()}):
        runs = [r for r in list_runs(stem) if r.get("labeled", 0) > 0]
        if runs:
            out[stem] = runs
    return out


def _draw_pred(im, r):
    """예측 박스·신뢰도 그리기 → 검출 신뢰도 목록 반환."""
    confs = []
    for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 150, 255), 3)
        cv2.putText(im, f"{float(c):.2f}", (x1, max(y1 - 6, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 255), 2)
        confs.append(float(c))
    return confs


def _propagate_into(video, shots, pi, pl, box_d, ref_d):
    """영상 하나를 SAM2 전파 → train/train_box/tap 에 '영상이름_프레임' 이름으로 통합 저장. 라벨수 반환."""
    from sam2.build_sam import build_sam2_video_predictor
    fs = autolabel._frames(video)
    predictor = build_sam2_video_predictor(CFG, str(CKPT), device=DEV)
    h, w = _rd(fs[0]).shape[:2]
    with torch.inference_mode(), torch.autocast(DEV, dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(autolabel.FRAME_CACHE / video),
                                     offload_video_to_cpu=True, offload_state_to_cpu=True)
        for fi, pts in shots:
            p = np.array([[rx * w, ry * h] for rx, ry, _ in pts], dtype=np.float32)
            l = np.array([int(lab) for *_, lab in pts], dtype=np.int32)
            predictor.add_new_points_or_box(inference_state=state, frame_idx=int(fi), obj_id=1, points=p, labels=l)
        masks = {}
        for fidx, _ids, logits in predictor.propagate_in_video(state):
            m = logits[0].cpu().numpy(); masks[fidx] = (m[0] if m.ndim == 3 else m) > 0.0
    del predictor, state; free_sam2()

    for fi, pts in shots:   # tap: 탭 프레임에 점+마스크+박스 한 장에
        mk = masks.get(int(fi)); base = _rd(fs[int(fi)]); has = mk is not None and mk.shape == (h, w)
        vis = _overlay(base, mk) if has else base
        bb = _bbox(mk) if has else None
        if bb:
            cv2.rectangle(vis, (bb[0], bb[1]), (bb[2], bb[3]), (0, 165, 255), 3)
        for rx, ry, lab in pts:
            cv2.circle(vis, (int(rx * w), int(ry * h)), 10, (255, 60, 0) if lab else (0, 0, 255), -1)
            cv2.circle(vis, (int(rx * w), int(ry * h)), 10, (255, 255, 255), 2)
        cv2.imwrite(str(ref_d / f"{video}_{Path(fs[int(fi)]).stem}.jpg"), vis)

    n = 0
    for i, p in enumerate(fs):
        mk = masks.get(i); im = _rd(p)
        if mk is not None and mk.shape != (h, w):
            mk = cv2.resize(mk.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        bb = _bbox(mk) if mk is not None else None
        nm = f"{video}_{p.stem}"   # 영상이름_번호 → 한 폴더에 통합해도 안 겹침
        if bb and (bb[2] - bb[0]) > 10 and (bb[3] - bb[1]) > 10:
            cv2.imwrite(str(pi / f"{nm}.jpg"), im)
            cx, cy = (bb[0] + bb[2]) / 2 / w, (bb[1] + bb[3]) / 2 / h
            bw, bh = (bb[2] - bb[0]) / w, (bb[3] - bb[1]) / h
            (pl / f"{nm}.txt").write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")
            boxed = im.copy()
            cv2.rectangle(boxed, (bb[0], bb[1]), (bb[2], bb[3]), (0, 165, 255), 3)
            cv2.putText(boxed, "part", (bb[0], max(bb[1] - 6, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.imwrite(str(box_d / f"{nm}.jpg"), boxed)
            n += 1
    return n, len(fs)


def _run_session(job_id, part, train_shots, test_srcs):
    """부품 세션: train 영상들 각자 전파 → 라벨 통합 → YOLO 학습 → test 평가.

    part="gearbox", train_shots={영상:[[fi,[[rx,ry,lab],...]],...]}, test_srcs=["test1",...].
    한 실행 = 한 시각 폴더 results/<부품>/<시각>/ 에 tap·train·train_box·model/ 이 직접(안에 또 시각 없음).
    라벨은 영상이름_번호 파일명으로 합침.
    정답 라벨 없어 mAP 아님 = 검출률(프레임당 검출)+신뢰도+육안.
    """
    j = JOBS[job_id]
    try:
        runid = datetime.now().strftime("%y%m%d_%H%M%S")
        sess = RESULTS / part / runid                      # results/<부품>/<시각>/  (한 폴더에 tap·train·train_box·model 직접)
        pi, pl = sess / "train" / "images", sess / "train" / "labels"
        box_d, ref_d = sess / "train_box", sess / "tap"
        for d in (pi, pl, box_d, ref_d):
            d.mkdir(parents=True, exist_ok=True)

        # 1) 라벨 생성: train 영상마다 전파(영상별) → 한 폴더에 통합
        vids = [v for v in train_shots if train_shots[v]]
        total_lbl, used = 0, []
        for vi, video in enumerate(vids):
            j.update(stage="propagate", note=f"라벨 생성 {vi + 1}/{len(vids)} · {video}")
            n, tot = _propagate_into(video, train_shots[video], pi, pl, box_d, ref_d)
            total_lbl += n
            used.append({"video": video, "labels": n, "frames": tot})
        if total_lbl < 5:
            raise RuntimeError(f"학습 라벨 부족({total_lbl}) — 학습 영상에 점을 찍어 주세요")

        # 2) 학습: 통합 train 으로 YOLO 학습 (model 은 results/<부품>/<시각>/model/)
        from ultralytics import YOLO
        j.update(stage="train", note="", train_labels=total_lbl, train_srcs=[u["video"] for u in used])
        model_dir = RESULTS / part / runid / "model"       # 같은 runid 아래 labels 와 나란히
        yaml = sess / "train.yaml"
        d = pi.resolve().as_posix()
        yaml.write_text(f"path: {sess.resolve().as_posix()}\ntrain:\n  - {d}\nval:\n  - {d}\nnames:\n  0: part\n",
                        encoding="utf-8")
        model = YOLO(config.PRETRAINED)
        model.train(data=str(yaml), epochs=EPOCHS, imgsz=640, batch=8, device=0,
                    project=str(model_dir / "runs"), name="model", exist_ok=True, verbose=False,
                    plots=False, degrees=15.0)
        w1 = Path(model.trainer.best)   # 실제 저장 경로
        del model; gc.collect(); torch.cuda.empty_cache()

        # 3) test 평가
        j.update(stage="eval", eval_total=len(test_srcs), eval_done=0)
        det = YOLO(str(w1))
        eval_res = []
        for ts in test_srcs:
            fs = autolabel._frames(ts)
            outd = model_dir / "eval" / ts
            outd.mkdir(parents=True, exist_ok=True)
            hit, confs = 0, []
            for p in fs:
                im = _rd(p)
                r = det.predict(source=im, conf=VAL_CONF, imgsz=640, verbose=False)[0]
                if len(r.boxes):
                    confs += _draw_pred(im, r); hit += 1
                cv2.imwrite(str(outd / f"{p.stem}.jpg"), im)
            rate = hit / len(fs) if fs else 0.0
            eval_res.append({
                "src": ts, "frames": len(fs), "detected": hit, "rate": round(rate, 3),
                "mean_conf": round(float(np.mean(confs)), 3) if confs else 0.0,
                "samples": [_b64(_rd(p), w=300) for p in sorted(outd.glob("*.jpg"))[::max(1, len(fs) // 6)]][:6]})
            j.update(eval_done=len(eval_res))
        gc.collect(); torch.cuda.empty_cache()

        meta = {"part": part, "run": runid, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "labels_dir": str(sess), "model_dir": str(model_dir), "train": used,
                "train_labels": total_lbl, "weights": str(w1),
                "eval": [{k: r[k] for k in ("src", "frames", "detected", "rate", "mean_conf")} for r in eval_res]}
        (sess / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        (model_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        j.update(stage="done", running=False, run=runid, part=part, train_labels=total_lbl,
                 train_srcs=[u["video"] for u in used], weights=str(w1), eval=eval_res)
    except Exception as e:
        j.update(stage="error", error=f"{type(e).__name__}: {e}", running=False)
    finally:
        _BUSY["on"] = False
        gc.collect(); torch.cuda.empty_cache()


def start_session(part, train_shots, test_srcs):
    with _LOCK:
        if _BUSY["on"]:
            return {"error": "이미 실행 중입니다."}
        if not part or not train_shots:
            return {"error": "학습 영상(점 찍은 것)이 필요합니다."}
        if not test_srcs:
            return {"error": "평가할 test 영상을 하나 이상 고르세요."}
        _BUSY["on"] = True
    jid = uuid.uuid4().hex[:8]
    JOBS[jid] = {"stage": "start", "running": True, "error": None}
    threading.Thread(target=_run_session, args=(jid, part, train_shots, test_srcs), daemon=True).start()
    return {"job": jid}


def list_runs(src):
    """그 영상의 지난 실행 목록(최근순). results/<영상>_<실행시각>/meta.json 기반."""
    out = []
    for d in sorted(RESULTS.glob(f"{src}_*"), reverse=True):
        mp = d / "meta.json"
        if not mp.exists():
            continue
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            m = {}
        out.append({"run": d.name[len(src) + 1:], "time": m.get("time", ""), "labeled": m.get("labeled", 0),
                    "total": m.get("total", 0), "trained": bool(m.get("trained")),
                    "detected": m.get("detected")})
    return out


def job_status(jid):
    return JOBS.get(jid, {"error": "unknown job"})


# ==================== 멀티클래스 부품 라벨링 (장비별 위저드) ====================
# 여러 부품 영상을 한 세션 폴더(results/parts/<세션>/)에 누적. 파일명=<영상>_<프레임>.
# 부품 하나씩: 탭 → 라벨 생성(전파) → 다음 부품. 다 모으면 34클래스로 통합 학습.
PARTS_ROOT = RESULTS / "parts"


def parts_sessions():
    """results/parts/<세션>/ 목록(최근순) + 세션별 라벨된 영상·라벨수."""
    out = []
    if PARTS_ROOT.exists():
        for d in sorted(PARTS_ROOT.glob("*"), reverse=True):
            if not d.is_dir():
                continue
            mp = d / "labeling.json"
            m = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
            vids = m.get("videos", {})
            out.append({"session": d.name, "videos": vids,
                        "n_videos": len(vids),
                        "total_labels": sum(v.get("labels", 0) for v in vids.values()),
                        "updated": m.get("updated", "")})
    return out


def _run_parts_label(job_id, session, video, shots):
    """부품 영상 하나 → SAM2 전파 → results/parts/<세션>/train 등에 <영상>_<프레임>로 누적."""
    j = JOBS[job_id]
    try:
        sess = PARTS_ROOT / session
        pi, pl = sess / "train" / "images", sess / "train" / "labels"
        box_d, ref_d = sess / "train_box", sess / "tap"
        for d in (pi, pl, box_d, ref_d):
            d.mkdir(parents=True, exist_ok=True)
        j.update(stage="propagate", note=f"{video} 라벨 생성 중")
        # 이 영상의 기존 라벨 있으면 지우고 새로(재탭 반영)
        for g in (pi.glob(f"{video}_*.jpg"), pl.glob(f"{video}_*.txt"),
                  box_d.glob(f"{video}_*.jpg"), ref_d.glob(f"{video}_*.jpg")):
            for f in list(g):
                f.unlink()
        n, tot = _propagate_into(video, shots, pi, pl, box_d, ref_d)
        # 세션 메타 갱신
        mp = sess / "labeling.json"
        meta = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {"session": session, "videos": {}}
        meta["videos"][video] = {"labels": n, "frames": tot,
                                 "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        meta["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        taps = [_b64(_rd(p), w=520) for p in sorted(ref_d.glob(f"{video}_*.jpg"))][:6]
        j.update(stage="done", running=False, video=video, labels=n, frames=tot,
                 session=session, taps=taps)
    except Exception as e:
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
    session = session or datetime.now().strftime("%y%m%d_%H%M%S")
    jid = uuid.uuid4().hex[:8]
    JOBS[jid] = {"stage": "start", "running": True, "error": None, "session": session, "video": video}
    threading.Thread(target=_run_parts_label, args=(jid, session, video, shots), daemon=True).start()
    return {"job": jid, "session": session}


def _run_multiclass(job_id, session, epochs, test_srcs, only_classes=None):
    """세션에 누적된 per-part 라벨(class 0) → 영상명→부품→클래스 remap → YOLO 학습 → 검출 평가.
    only_classes 지정 시 그 클래스만 학습."""
    j = JOBS[job_id]
    try:
        sys.path.insert(0, str(config.BASE_DIR / "scripts" / "experiments"))
        import build_multiclass as bm
        names, name2idx = bm.load_classes()
        sess = PARTS_ROOT / session
        st = sess / "train"
        if not (st / "images").exists():
            raise RuntimeError("이 세션에 라벨이 없습니다. 부품을 먼저 탭·라벨 생성하세요.")

        # 1) 클래스 매핑 통합
        j.update(stage="build", note="라벨 클래스 매핑 통합")
        out = sess / "multiclass"
        oi, ol = out / "images", out / "labels"
        oi.mkdir(parents=True, exist_ok=True); ol.mkdir(parents=True, exist_ok=True)
        sel = set(only_classes) if only_classes else None    # 선택한 클래스만 학습(없으면 전체)
        per, miss = {}, {}
        for ip in sorted((st / "images").glob("*.jpg")):
            stem = ip.stem
            cls = bm.stem_to_class(stem)
            if sel is not None and cls not in sel:
                continue
            idx = name2idx.get(cls)
            if idx is None:
                miss[cls] = miss.get(cls, 0) + 1; continue
            lp = st / "labels" / f"{stem}.txt"
            if not lp.exists():
                continue
            lines = [f"{idx} {p[1]} {p[2]} {p[3]} {p[4]}"
                     for p in (l.split() for l in lp.read_text(encoding="utf-8").splitlines()) if len(p) == 5]
            if not lines:
                continue
            shutil.copy(ip, oi / f"{stem}.jpg")
            (ol / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            per[cls] = per.get(cls, 0) + 1
        n_img = sum(per.values())
        if n_img < 5:
            raise RuntimeError(f"통합 라벨 부족({n_img}). 부품을 더 탭하세요. 미매핑: {miss}")
        yml = out / "data.yaml"
        nb = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
        d = oi.resolve().as_posix()
        yml.write_text(f"path: {out.resolve().as_posix()}\ntrain:\n  - {d}\nval:\n  - {d}\nnames:\n{nb}\n",
                       encoding="utf-8")

        # 2) 학습
        from ultralytics import YOLO
        j.update(stage="train", note=f"{n_img}장 / {len(per)}클래스", n_images=n_img, n_classes=len(per))
        model_dir = out / "model"
        model = YOLO(config.PRETRAINED)
        model.train(data=str(yml), epochs=epochs, imgsz=640, batch=8, device=0,
                    project=str(model_dir / "runs"), name="model", exist_ok=True, verbose=False,
                    plots=False, degrees=15.0)
        w1 = Path(model.trainer.best)
        del model; gc.collect(); torch.cuda.empty_cache()

        # 3) test 영상 검출 평가 (정답 라벨 없어 mAP 아님 = 검출률+신뢰도+클래스분포+육안)
        eval_res = []
        if test_srcs:
            j.update(stage="eval", eval_total=len(test_srcs), eval_done=0)
            det = YOLO(str(w1))
            for ts in test_srcs:
                fs = autolabel._frames(ts)
                outd = model_dir / "eval" / ts
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
                            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 150, 255), 3)
                            cv2.putText(im, f"{nm} {float(c):.2f}", (x1, max(y1 - 6, 16)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 2)
                    cv2.imwrite(str(outd / f"{p.stem}.jpg"), im)
                top = sorted(cls_cnt.items(), key=lambda x: -x[1])[:3]
                eval_res.append({"src": ts, "frames": len(fs), "detected": hit,
                                 "rate": round(hit / len(fs), 3) if fs else 0.0,
                                 "mean_conf": round(float(np.mean(confs)), 3) if confs else 0.0,
                                 "top_classes": top,
                                 "samples": [_b64(_rd(p), w=300) for p in sorted(outd.glob("*.jpg"))[::max(1, len(fs) // 6)]][:6]})
                j.update(eval_done=len(eval_res))
            del det; gc.collect(); torch.cuda.empty_cache()

        meta = {"session": session, "n_images": n_img, "n_classes": len(per), "per_class": per,
                "classes": names, "weights": str(w1), "miss": miss,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "eval": [{k: r[k] for k in ("src", "frames", "detected", "rate", "mean_conf", "top_classes")}
                         for r in eval_res]}
        (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        j.update(stage="done", running=False, session=session, n_images=n_img, n_classes=len(per),
                 per_class=per, weights=str(w1), eval=eval_res, miss=miss)
    except Exception as e:
        j.update(stage="error", error=f"{type(e).__name__}: {e}", running=False)
    finally:
        _BUSY["on"] = False
        gc.collect(); torch.cuda.empty_cache()


def start_multiclass(session, epochs, test_srcs, only_classes=None):
    with _LOCK:
        if _BUSY["on"]:
            return {"error": "이미 실행 중입니다."}
        if not session:
            return {"error": "세션이 없습니다. 부품을 먼저 라벨 생성하세요."}
        _BUSY["on"] = True
    jid = uuid.uuid4().hex[:8]
    JOBS[jid] = {"stage": "start", "running": True, "error": None}
    threading.Thread(target=_run_multiclass, args=(jid, session, int(epochs or EPOCHS), test_srcs or [], only_classes or None),
                     daemon=True).start()
    return {"job": jid}
