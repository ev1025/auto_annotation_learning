# -*- coding: utf-8 -*-
"""scripts/verify/dashboard_api.py - 대시보드 API 서버 (FastAPI) + React 프론트 서빙.

프론트 소스: dashboard/ (React + Vite). 빌드 산출물 dashboard/dist 를 루트에 서빙.
데이터 계층은 scripts/dashboard_core.py 공용 모듈.

실행:
  cd dashboard && npm install && npm run build   # 프론트 변경 시 1회
  python scripts/verify/dashboard_api.py             # http://127.0.0.1:7862
"""
import mimetypes
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))  # 공용 config·experiments(build_multiclass·point_ref_lib)
sys.path.insert(0, str(Path(__file__).resolve().parent))                   # backend/autolearning (dashboard_core·autolabel·sam2_autolabel)

import uvicorn
from fastapi import Body, FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

import config
import dashboard_core as core
import autolabel
import sam2_autolabel as sa

PORT = 7862
# 바인딩 호스트. 기본은 로컬 전용(안전). 서버에서 같이 쓰는 사람이 브라우저로 보게 하려면
# DASH_HOST=0.0.0.0 으로 띄운다 (대시보드에 인증이 없으니 신뢰된 내부망에서만).
HOST = os.environ.get("DASH_HOST", "127.0.0.1")
DIST = config.BASE_DIR / "frontend" / "dist"

app = FastAPI(title="XR 오토러닝 대시보드 API")

# ---- 조회(read) 소스 선택 ----
# XR_DB_READS=1 이면 조회를 DB 로 한다. 기본은 파일(현행 동작) — DB 없이도 대시보드가 그대로 돈다.
# 동등성은 backend/db/verify_reads.py 로 검증했다(파일판 vs DB판 결과 비교).
_db_reads = None
if os.environ.get("XR_DB_READS") == "1":
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # backend (db 패키지)
        from db import reads as _db_reads   # noqa: PLC0415
        print(f"[DB] 조회 경로 = DB / 행 수 {_db_reads.counts()}", flush=True)
    except Exception as e:   # noqa: BLE001  DB 문제로 서버가 못 뜨면 안 되므로 폴백
        print(f"[DB] 초기화 실패 -> 파일 조회로 폴백: {type(e).__name__}: {e}", flush=True)
        _db_reads = None


def _read(fn_name, file_fn):
    """DB 우선, 실패 시 파일 폴백. 조회 실패로 화면이 비지 않게 한다."""
    if _db_reads is not None:
        try:
            return getattr(_db_reads, fn_name)()
        except Exception as e:   # noqa: BLE001
            print(f"[DB] {fn_name} 실패 -> 파일 폴백: {type(e).__name__}: {e}", flush=True)   # 조용한 폴백 방지(운영 중 즉시 보이게)
    return file_fn()


@app.get("/api/db/status")
def api_db_status():
    """조회 소스와 DB 행 수(운영 확인용)."""
    if _db_reads is None:
        return {"source": "files", "db": False}
    try:
        return {"source": "db", "db": True, "counts": _db_reads.counts()}
    except Exception as e:   # noqa: BLE001
        return {"source": "files(폴백)", "db": False, "error": f"{type(e).__name__}: {e}"}


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
    return reg.create_part(payload.get("name"), payload.get("category_id"), payload.get("description", ""))


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


@app.get("/api/methods")
def api_methods():
    return [{"id": m["id"], "no": m["no"], "title": m["title"],
             "badge": m["badge"], "badge_label": m["badge_label"], "live": m["live"]}
            for m in core.METHODS]


@app.get("/api/method/{mid}")
def api_method(mid: str):
    m = core.method_by_id(mid)
    if not m:
        return JSONResponse({"error": "unknown method"}, status_code=404)
    return {"id": m["id"], "no": m["no"], "title": m["title"],
            "badge": m["badge"], "badge_label": m["badge_label"], "live": m["live"],
            "subtitle": m.get("subtitle", ""), "ordered": m.get("ordered", False),
            "tech": core.resolve_tech(m), "flow": m.get("flow", []),
            "bullets": m["bullets"], "code": m.get("code", []),
            "gallery": core.method_gallery(m),
            "metrics": core.method_metrics(m),
            "extras": core.method_extras(m)}


@app.get("/api/compare")
def api_compare(idx: int = 0, conf: float = 0.6):
    return core.compare(idx, conf)


@app.get("/api/glossary")
def api_glossary():
    return core.GLOSSARY


@app.get("/api/experiments")
def api_experiments():
    return {cat: list(topics.keys()) for cat, topics in core.EXPERIMENTS.items()}


@app.get("/api/experiment")
def api_experiment(cat: str, topic: str):
    return core.experiment_metrics(cat, topic)


@app.post("/api/export")
def api_export():
    return {"path": core.export_report()}


@app.get("/api/autolabel/folders")
def api_autolabel_folders():
    return _read("list_folders", autolabel.list_folders)


@app.get("/api/autolabel/frame")
def api_autolabel_frame(src: str, idx: int, w: int = 960):
    b = autolabel.frame_jpeg(src, idx, w)
    if b is None:
        return JSONResponse({"error": "frame not found"}, status_code=404)
    return Response(content=b, media_type="image/jpeg")


@app.get("/api/autolabel/prepare")
def api_autolabel_prepare(src: str):
    return autolabel.prepare(src)


# ---- SAM2 단계형 오토라벨 ----
@app.post("/api/sam2/mask")
def api_sam2_mask(payload: dict = Body(...)):
    return sa.mask_preview(payload.get("src"), int(payload.get("frame", 0)), payload.get("points", []))


@app.get("/api/sam2/parts_sessions")
def api_parts_sessions():
    return sa.parts_sessions()


@app.get("/api/sam2/shots")
def api_sam2_shots():
    """부품별 참조샷 취합 → {"<영상>": {"<프레임>": [[rx,ry,lab],...]}}. 프론트 참조샷 복원."""
    return _read("load_shots", sa.load_shots)


@app.post("/api/sam2/parts_label")
def api_parts_label(payload: dict = Body(...)):
    return sa.start_parts_label(payload.get("session"), payload.get("video"), payload.get("shots", []))


@app.post("/api/sam2/parts_label_batch")
def api_parts_label_batch(payload: dict = Body(...)):
    return sa.start_parts_label_batch(payload.get("session"), payload.get("items", []))


@app.post("/api/sam2/multiclass")
def api_multiclass(payload: dict = Body(...)):
    return sa.start_multiclass(payload.get("session"), payload.get("epochs"), payload.get("test_srcs", []),
                               payload.get("classes"), payload.get("augment", False))


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
    return sa.list_part_frames(part)


@app.get("/api/sam2/labeled_parts")
def api_sam2_labeled_parts():
    return _read("labeled_parts", sa.labeled_parts)


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
    return _read("served_model", sa.served_model) or {"none": True}


@app.get("/api/sam2/active")
def api_sam2_active():
    return sa.active_job()


@app.get("/api/sam2/models")
def api_sam2_models():
    return _read("list_models", sa.list_models)


@app.post("/api/sam2/rollback_to")
def api_sam2_rollback_to(payload: dict = Body(...)):
    return sa.rollback_to(payload.get("model_id"))


@app.get("/api/sam2/status")
def api_sam2_status(job: str):
    return sa.job_status(job)


@app.get("/previews/{sub}/{name}")
def api_preview(sub: str, name: str):
    p = (core.PREV_DIR / sub / name).resolve()
    if not str(p).startswith(str(core.PREV_DIR.resolve())) or not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p)


if DIST.exists():
    app.mount("/", StaticFiles(directory=str(DIST), html=True), name="frontend")

if __name__ == "__main__":
    if not DIST.exists():
        print("경고: frontend/dist 없음. 먼저 빌드하세요: cd frontend && npm install && npm run build")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
