# -*- coding: utf-8 -*-
"""대시보드 API 서버(FastAPI) + React 프론트 서빙.

프론트 소스: frontend/ (React + Vite). 빌드 산출물 frontend/dist 를 루트에 서빙.

실행:
  cd frontend && npm install && npm run build            # 프론트 변경 시 1회
  XR_DB_READS=1 python backend/autolearning/dashboard_api.py   # http://127.0.0.1:7862
"""
import json
import mimetypes
import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))  # 공용 config·experiments(build_multiclass)
sys.path.insert(0, str(Path(__file__).resolve().parent))                   # backend/autolearning (autolabel·sam2_autolabel)

import requests
import uvicorn
from fastapi import Body, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import DBAPIError

import config
import autolabel
import sam2_autolabel as sa

# 포트. 기본은 7862 이고, 한 PC 에서 두 개를 동시에 띄울 때만 바꾼다
# (예: 토르 것을 보면서 로컬 것도 켜 두는 경우. 둘 다 7862 면 뒤에 뜬 쪽이 죽는다)
#   DASH_PORT=1234 python backend/autolearning/dashboard_api.py
PORT = int(os.environ.get("DASH_PORT", "7862"))
# 바인딩 호스트. 기본은 로컬 전용(안전). 서버에서 같이 쓰는 사람이 브라우저로 보게 하려면
# DASH_HOST=0.0.0.0 으로 띄운다 (대시보드에 인증이 없으니 신뢰된 내부망에서만).
HOST = os.environ.get("DASH_HOST", "127.0.0.1")
DIST = config.BASE_DIR / "frontend" / "dist"

app = FastAPI(title="XR 오토러닝 대시보드 API")

# ---- 조회(read) 소스 선택 ----
# 조회는 전부 DB 로 한다. 파일 폴백은 없앴다.
# 이유: 폴백이 있으면 DB 가 끊길 때 화면이 반쪽이 된다. 폴백을 가진 엔드포인트(영상 목록·참조샷 등)는
# 파일에서 읽어 정상으로 보이는데, DB 전용인 부품 목록·카테고리는 비어 보였다(2026-08-21 확인).
# 그래서 실패는 실패로 드러내고, 모든 화면이 같은 소스(DB)를 본다.
# 사진·영상 바이트는 원래 파일시스템에서 서빙한다(DB 에 이미지를 넣지 않는다).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # backend (db 패키지)
from db import reads as _db_reads   # noqa: E402


# DB 가 끊기면 어느 엔드포인트든 같은 모양으로 503 을 준다.
# 엔드포인트마다 try/except 를 두는 대신 핸들러 하나로 받는다(등록·수정 같은 쓰기까지 포함).
@app.exception_handler(DBAPIError)
async def _db_unavailable(request, exc):    # noqa: ARG001
    return JSONResponse(status_code=503,
                        content={"error": "db_unavailable", "detail": str(exc)[:200],
                                 "at": request.url.path})


def _read(fn_name, *args):
    """DB 조회. 실패하면 그대로 오류를 돌려준다(조용한 폴백 금지)."""
    try:
        return getattr(_db_reads, fn_name)(*args)
    except Exception as e:   # noqa: BLE001 - 서버가 죽지 않게 하고 원인은 응답·로그로 보인다
        msg = f"{type(e).__name__}: {e}"
        print(f"[DB] {fn_name} 실패: {msg}", flush=True)
        return JSONResponse(status_code=503,
                            content={"error": "db_unavailable", "detail": msg[:200], "at": fn_name})


@app.get("/api/db/status")
def api_db_status():
    """DB 연결 상태와 행 수(운영 확인용). 화면 상단 배너가 이걸 본다."""
    try:
        return {"source": "db", "db": True, "counts": _db_reads.counts()}
    except Exception as e:   # noqa: BLE001
        return JSONResponse(status_code=503,
                            content={"source": "db", "db": False, "error": f"{type(e).__name__}: {e}"[:200]})


# ---- 부품 등록·목록(카테고리·부품·영상/3D 업로드) ----
# 이름·카테고리·설명은 파일로 표현이 안 되므로 DB 가 원본. DB 없으면 여기만 503.
try:
    import parts_registry as reg   # noqa: PLC0415
except Exception as e:   # noqa: BLE001
    reg = None
    print(f"[DB] 부품 등록 기능 비활성(DB 필요): {type(e).__name__}: {e}", flush=True)


def _need_reg():
    return JSONResponse({"error": "부품 등록 기능에는 DB 가 필요합니다. deploy/.env 와 DB 컨테이너를 확인하세요."},
                        status_code=503)


@app.get("/api/categories")
def api_categories():
    return reg.list_categories() if reg else _need_reg()


@app.post("/api/categories")
def api_category_add(payload: dict = Body(...)):
    return reg.add_category(payload.get("name")) if reg else _need_reg()


@app.patch("/api/categories/{cid}")
def api_category_rename(cid: int, payload: dict = Body(...)):
    return reg.rename_category(cid, payload.get("name")) if reg else _need_reg()


@app.delete("/api/categories/{cid}")
def api_category_delete(cid: int):
    return reg.delete_category(cid) if reg else _need_reg()


@app.get("/api/parts")
def api_parts():
    return reg.list_parts() if reg else _need_reg()


@app.post("/api/parts")
def api_part_create(payload: dict = Body(...)):
    if not reg:
        return _need_reg()
    # part_code 를 주면 그 번호로 등록(안 주면 다음 번호 자동). 클라이언트 매핑용 불변 번호다.
    return reg.create_part(payload.get("name"), payload.get("category_id"), payload.get("description", ""),
                           payload.get("part_code"))


@app.patch("/api/parts/{pid}")
def api_part_update(pid: int, payload: dict = Body(...)):
    if not reg:
        return _need_reg()
    return reg.update_part(pid, payload.get("name"),
                           payload.get("category_id", "keep"), payload.get("description"))


@app.delete("/api/parts/{pid}")
def api_part_delete(pid: int):
    return reg.delete_part(pid) if reg else _need_reg()


@app.post("/api/parts/{pid}/video")
async def api_part_video(pid: int, file: UploadFile = File(...)):
    """영상 업로드 → 저장 후 프레임 사전 추출을 백그라운드로 시작(job 반환)."""
    if not reg:
        return _need_reg()
    return reg.upload_video(pid, file.filename, await file.read())


@app.post("/api/parts/{pid}/model3d")
async def api_part_model3d(pid: int, file: UploadFile = File(...)):
    if not reg:
        return _need_reg()
    return reg.upload_model3d(pid, file.filename, await file.read())


@app.get("/api/parts/job")
def api_part_job(job: str):
    return reg.job_status(job) if reg else _need_reg()


@app.get("/api/parts/{pid}/videos")
def api_part_videos(pid: int):
    """영상 관리 모달용 목록(영상별 역할·프레임수·용량)."""
    return reg.list_part_videos(pid) if reg else _need_reg()


@app.delete("/api/parts/{pid}/video/{stem}")
def api_part_video_delete(pid: int, stem: str):
    return reg.delete_part_video(pid, stem) if reg else _need_reg()


@app.get("/api/parts/{pid}/video/{stem}/file")
def api_part_video_file(pid: int, stem: str, request: Request):
    """미리보기 재생용 영상 스트리밍. Range 를 처리해야 <video> 에서 탐색(시크)이 된다.
    (StarletteFileResponse 는 부분 응답을 안 하므로 직접 206 을 만든다)"""
    if not reg:
        return _need_reg()
    fp = reg.video_file_path(pid, stem)
    if not fp:
        return JSONResponse({"error": "영상을 찾지 못했습니다."}, status_code=404)
    size = fp.stat().st_size
    mime = mimetypes.guess_type(fp.name)[0] or "video/mp4"
    rng = request.headers.get("range") or request.headers.get("Range")
    if not rng:
        return FileResponse(fp, media_type=mime,
                            headers={"Accept-Ranges": "bytes", "Cache-Control": "no-store"})
    m = re.match(r"bytes=(\d*)-(\d*)", rng)
    if not m:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
    start = int(m.group(1)) if m.group(1) else 0
    end = int(m.group(2)) if m.group(2) else min(start + 4 * 1024 * 1024 - 1, size - 1)   # 4MB 청크
    end = min(end, size - 1)
    if start > end:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})

    def chunks():
        with open(fp, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                buf = f.read(min(256 * 1024, left))
                if not buf:
                    break
                left -= len(buf)
                yield buf

    return StreamingResponse(chunks(), status_code=206, media_type=mime, headers={
        "Content-Range": f"bytes {start}-{end}/{size}",
        "Content-Length": str(end - start + 1),
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-store",
    })


# ---- 외부 연동(XR) 전용 읽기 API ----
# 외부에 여는 것은 이 하나뿐이다. 부품 목록·번호만 주고, 등록·삭제·학습은 열지 않는다.
# 응답 필드를 고정해 두었으므로 내부 /api/parts 가 바뀌어도 상대는 깨지지 않는다.
@app.get("/api/xr/parts")
def api_xr_parts():
    return _read("xr_parts")


@app.get("/api/xr/parts/{name}/model3d")
def api_xr_model3d(name: str):
    """3D 모델 파일 다운로드. DB 에는 경로만 있고 바이트는 파일시스템에 있으므로 여기서 내려준다."""
    fp = _db_reads.xr_model3d_path(name)
    if not fp:
        return JSONResponse({"error": "3D 모델이 없습니다.", "part": name}, status_code=404)
    return FileResponse(fp, filename=fp.name, media_type="application/octet-stream",
                        headers={"Cache-Control": "no-store"})


@app.delete("/api/parts/{pid}/model3d")
def api_part_model3d_delete(pid: int):
    return reg.delete_model3d(pid) if reg else _need_reg()


@app.get("/api/part_codes")
def api_part_codes():
    """전역 부품 코드표 {부품명: 코드}. 클라이언트·문서용 참조.
    모델의 클래스 인덱스(detection_code)와 달리 재학습해도 바뀌지 않는다."""
    try:
        import part_codes   # noqa: PLC0415
        t = part_codes.table()
        return {"count": len(t), "codes": t}
    except Exception as e:   # noqa: BLE001
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


@app.get("/api/autolabel/folders")
def api_autolabel_folders():
    return _read("list_folders")


@app.get("/api/autolabel/frame")
def api_autolabel_frame(src: str, idx: int, w: int = 960):
    b = autolabel.frame_jpeg(src, idx, w)
    if b is None:
        return JSONResponse({"error": "frame not found"}, status_code=404)
    return Response(content=b, media_type="image/jpeg")


### ---- 폰 인식 테스트(recog.html) 용 추론 프록시 ----
# 추론서버(:9412)는 CORS 를 열지 않았고 포트도 다르다. 브라우저에서 직접 부르면 막히므로
# 대시보드(:7862)가 한 번 받아서 넘긴다(= 폰은 포트 하나만 알면 된다).
YOLO_URL = os.environ.get("YOLO_SERVER_URL", "http://127.0.0.1:9412").rstrip("/")


# ===== 임시 기능: 배경(오검) 자동 수집 — 배경 학습이 끝나면 이 블록과 recog.html 토글을 지운다 =====
# 부품이 없는 곳을 돌아다니며 켜 두면, 검출이 뜬 프레임 = 오검이므로 그대로 모은다.
# 사람이 따로 판정할 필요가 없다(그 자리에 부품이 없다는 사실이 정답이다).
NEG_DIR = config.DATA_DIR / "bell412" / "_negatives" / "images"
NEG_MIN_GAP = 1.5      # 초. 3fps 로 같은 자리를 계속 찍으면 같은 그림만 쌓인다
_neg_last = {"t": 0.0}


def _save_negative(raw: bytes, dets: list) -> int:
    """오검 프레임 저장. 반환값 = 지금까지 모인 장수(화면 카운터용)."""
    import time
    now = time.time()
    if now - _neg_last["t"] < NEG_MIN_GAP:
        return len(list(NEG_DIR.glob("*.jpg")))
    _neg_last["t"] = now
    NEG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%y%m%d_%H%M%S_") + f"{int(now * 1000) % 1000:03d}"
    (NEG_DIR / f"bg_{stamp}.jpg").write_bytes(raw)
    line = json.dumps({"file": f"bg_{stamp}.jpg", "time": stamp,
                       "false": [(d.get("detection_class"), d.get("confidence")) for d in dets]},
                      ensure_ascii=False)
    with open(NEG_DIR.parent / "collected.jsonl", "a", encoding="utf-8") as f:   # 무엇을 오검했는지 기록
        f.write(line + "\n")
    return len(list(NEG_DIR.glob("*.jpg")))


@app.post("/api/detect")
async def api_detect(image: UploadFile = File(...), camera_id: str = Form("PHONE"),
                     collect: str = Form("")):
    raw = await image.read()
    if not raw:
        return JSONResponse(status_code=400, content={"error": "invalid_image"})
    try:
        r = requests.post(f"{YOLO_URL}/detect",
                          files={"image": ("frame.jpg", raw, "image/jpeg")},
                          data={"camera_id": camera_id}, timeout=30)
        body = r.json()
        if collect and r.status_code == 200 and body.get("detections"):
            body["collected"] = _save_negative(raw, body["detections"])
        return JSONResponse(status_code=r.status_code, content=body)
    except Exception as e:   # noqa: BLE001 - 추론서버가 꺼져 있어도 페이지는 살아 있어야 한다
        return JSONResponse(status_code=502,
                            content={"error": "infer_unreachable", "detail": f"{type(e).__name__}: {e}"[:200]})


@app.get("/api/detect/health")
def api_detect_health():
    """페이지 상단에 '현재 모델·클래스'를 보여주기 위한 통과 조회."""
    try:
        r = requests.get(f"{YOLO_URL}/health", timeout=10)
        return JSONResponse(status_code=r.status_code, content=r.json())
    except Exception as e:   # noqa: BLE001
        return JSONResponse(status_code=502,
                            content={"ok": False, "error": f"{type(e).__name__}: {e}"[:200]})


@app.get("/api/autolabel/prepare")
def api_autolabel_prepare(src: str):
    r = autolabel.prepare(src)
    # 프레임을 자른 뒤 DB 의 part_videos.n_frames 를 갱신한다.
    # 목록 조회가 DB(reads.list_folders)이고 거기서 ready = bool(n_frames) 로 판단하므로,
    # 이걸 안 하면 이미 잘린 영상도 계속 '미준비'로 보여 화면을 열 때마다 전체를 다시 훑는다.
    if r.get("count"):
        try:
            sa._db_sync_part(src)
        except Exception as e:   # noqa: BLE001 - 색인 실패가 화면 동작을 막으면 안 된다
            print(f"[DB] prepare 색인 실패: {type(e).__name__}: {e}", flush=True)
    return r


# ---- SAM2 단계형 오토라벨 ----
@app.post("/api/sam2/mask")
def api_sam2_mask(payload: dict = Body(...)):
    return sa.mask_preview(payload.get("src"), int(payload.get("frame", 0)), payload.get("points", []))


@app.get("/api/sam2/parts_sessions")
def api_parts_sessions():
    return _read("parts_sessions")


@app.post("/api/sam2/delete_shot")
def api_sam2_delete_shot(payload: dict = Body(...)):
    """참조샷 1개 삭제(화면에서 x). shots.json 에 즉시 반영한다."""
    return sa.delete_shot(payload.get("video"), payload.get("frame"))


@app.get("/api/sam2/shots")
def api_sam2_shots():
    """부품별 참조샷 취합 → {"<영상>": {"<프레임>": [[rx,ry,lab],...]}}. 프론트 참조샷 복원."""
    return _read("load_shots")


@app.post("/api/sam2/parts_label")
def api_parts_label(payload: dict = Body(...)):
    return sa.start_parts_label(payload.get("session"), payload.get("video"), payload.get("shots", []))


@app.post("/api/sam2/parts_label_batch")
def api_parts_label_batch(payload: dict = Body(...)):
    return sa.start_parts_label_batch(payload.get("session"), payload.get("items", []))


@app.post("/api/sam2/multiclass")
def api_multiclass(payload: dict = Body(...)):
    # 빈 body 로도 전체 학습(36부품·기본 에폭)이 시작되던 문제. 학습은 GPU·디스크를 크게 쓰므로
    # epochs 를 필수로 받고, classes 는 빈 배열이면 거부한다(실수로 전체가 도는 것을 막는다).
    try:
        ep = int(payload.get("epochs"))
    except (TypeError, ValueError):
        return {"error": "epochs 를 지정하세요(1~1000)."}
    if not 1 <= ep <= 1000:
        return {"error": "epochs 는 1~1000 사이여야 합니다."}
    cls = payload.get("classes")
    if cls is not None and (not isinstance(cls, list) or not cls):
        return {"error": "학습할 부품을 선택하세요."}
    # replay=true 로 보내면 서비스 모델의 기존 부품도 함께 학습한다(기본은 고른 것만).
    return sa.start_multiclass(payload.get("session"), ep, cls, payload.get("augment", False),
                               payload.get("replay", False))


@app.post("/api/sam2/cancel")
def api_sam2_cancel(payload: dict = Body(...)):
    return sa.cancel_multiclass(payload.get("job"))


@app.post("/api/sam2/compare")
def api_sam2_compare(payload: dict = Body(...)):
    return sa.start_compare(payload.get("session"), payload.get("base_model_id"))


@app.post("/api/sam2/delete_model")
def api_sam2_delete_model(payload: dict = Body(...)):
    return sa.delete_model(payload.get("model_id"))


@app.get("/api/sam2/part_frames")
def api_sam2_part_frames(part: str):
    return _read("list_part_frames", part)


@app.get("/api/sam2/labeled_parts")
def api_sam2_labeled_parts():
    return _read("labeled_parts")


@app.get("/api/sam2/train_frame")
def api_sam2_train_frame(session: str, name: str, w: int = 360, part: str = None):
    b = sa.train_frame_jpeg(session, name, w, part)
    if b is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return Response(content=b, media_type="image/jpeg")


@app.post("/api/sam2/delete_train_frame")
def api_sam2_delete_train_frame(payload: dict = Body(...)):
    return sa.delete_train_frame(payload.get("session"), payload.get("name"), payload.get("part"))


@app.post("/api/sam2/delete_video")
def api_sam2_delete_video(payload: dict = Body(...)):
    return sa.delete_video(payload.get("src") or payload.get("video"))


@app.post("/api/sam2/apply_model")
def api_sam2_apply_model(payload: dict = Body(...)):
    return sa.apply_model(payload.get("session"))


@app.post("/api/sam2/rollback")
def api_sam2_rollback(payload: dict = Body(...)):
    return sa.rollback()


@app.get("/api/sam2/served")
def api_sam2_served():
    return _read("served_model") or {"none": True}


@app.get("/api/sam2/active")
def api_sam2_active():
    return sa.active_job()


@app.get("/api/sam2/models")
def api_sam2_models():
    return _read("list_models")


@app.post("/api/sam2/rollback_to")
def api_sam2_rollback_to(payload: dict = Body(...)):
    return sa.rollback_to(payload.get("model_id"))


@app.get("/api/sam2/status")
def api_sam2_status(job: str):
    return sa.job_status(job)


class NoCacheHTML(StaticFiles):
    """index.html 만 캐시 금지. 파일명에 해시가 붙는 asset 은 캐시해도 안전하지만,
    index.html 이 캐시되면 새로 빌드해도 브라우저가 옛 번들을 계속 불러온다."""

    async def get_response(self, path, scope):
        r = await super().get_response(path, scope)
        rel = str(path).replace("\\", "/")     # 윈도우에서는 'assets\index-xxx.js' 로 들어온다
        if path in ("", ".", "/") or rel.endswith(".html"):
            r.headers["Cache-Control"] = "no-store"
        elif rel.startswith("assets/"):
            # 빌드가 파일명에 해시를 붙이므로 내용이 바뀌면 이름이 바뀐다 -> 영구 캐시로 재요청을 없앤다
            r.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return r


if DIST.exists():
    app.mount("/", NoCacheHTML(directory=str(DIST), html=True), name="frontend")

if __name__ == "__main__":
    if not DIST.exists():
        print("경고: frontend/dist 없음. 먼저 빌드하세요: cd frontend && npm install && npm run build")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
