# -*- coding: utf-8 -*-
"""scripts/verify/dashboard_api.py - 대시보드 API 서버 (FastAPI) + React 프론트 서빙.

프론트 소스: dashboard/ (React + Vite). 빌드 산출물 dashboard/dist 를 루트에 서빙.
데이터 계층은 scripts/dashboard_core.py 공용 모듈.

실행:
  cd dashboard && npm install && npm run build   # 프론트 변경 시 1회
  python scripts/verify/dashboard_api.py             # http://127.0.0.1:7862
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ 공용
sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn
from fastapi import Body, FastAPI
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

import config
import dashboard_core as core
import autolabel
import sam2_autolabel as sa

PORT = 7862
# 바인딩 호스트. 기본은 로컬 전용(안전). 서버에서 같이 쓰는 사람이 브라우저로 보게 하려면
# DASH_HOST=0.0.0.0 으로 띄운다 (대시보드에 인증이 없으니 신뢰된 내부망에서만).
HOST = os.environ.get("DASH_HOST", "127.0.0.1")
DIST = config.BASE_DIR / "dashboard" / "dist"

app = FastAPI(title="XR 오토러닝 대시보드 API")


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


@app.get("/api/autolabel/sources")
def api_autolabel_sources():
    return autolabel.list_sources()


@app.get("/api/autolabel/folders")
def api_autolabel_folders():
    return autolabel.list_folders()


@app.get("/api/autolabel/frame")
def api_autolabel_frame(src: str, idx: int, w: int = 960):
    b = autolabel.frame_jpeg(src, idx, w)
    if b is None:
        return JSONResponse({"error": "frame not found"}, status_code=404)
    return Response(content=b, media_type="image/jpeg")


@app.get("/api/autolabel/prepare")
def api_autolabel_prepare(src: str):
    return autolabel.prepare(src)


@app.post("/api/autolabel/run")
def api_autolabel_run(payload: dict = Body(...)):
    return autolabel.start_job(payload.get("src"), payload.get("shots", []),
                               payload.get("tau", 0.70))


@app.get("/api/autolabel/status")
def api_autolabel_status(job: str):
    return autolabel.job_status(job)


# ---- SAM2 단계형 오토라벨 ----
@app.post("/api/sam2/mask")
def api_sam2_mask(payload: dict = Body(...)):
    return sa.mask_preview(payload.get("src"), int(payload.get("frame", 0)), payload.get("points", []))


@app.post("/api/sam2/propagate")
def api_sam2_propagate(payload: dict = Body(...)):
    return sa.start_propagate(payload.get("src"), payload.get("shots", []))


@app.post("/api/sam2/train_eval")
def api_sam2_train_eval(payload: dict = Body(...)):
    return sa.start_train_eval(payload.get("train_runs", []), payload.get("test_srcs", []))


@app.post("/api/sam2/session")
def api_sam2_session(payload: dict = Body(...)):
    return sa.start_session(payload.get("part"), payload.get("train_shots", {}), payload.get("test_srcs", []))


@app.get("/api/sam2/parts_sessions")
def api_parts_sessions():
    return sa.parts_sessions()


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


@app.get("/api/sam2/train_frames")
def api_sam2_train_frames(session: str):
    return sa.list_train_frames(session)


@app.get("/api/sam2/part_frames")
def api_sam2_part_frames(part: str):
    return sa.list_part_frames(part)


@app.get("/api/sam2/labeled_parts")
def api_sam2_labeled_parts():
    return sa.labeled_parts()


@app.get("/api/sam2/train_frame")
def api_sam2_train_frame(session: str, name: str, w: int = 360):
    b = sa.train_frame_jpeg(session, name, w)
    if b is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return Response(content=b, media_type="image/jpeg")


@app.post("/api/sam2/delete_train_frame")
def api_sam2_delete_train_frame(payload: dict = Body(...)):
    return sa.delete_train_frame(payload.get("session"), payload.get("name"))


@app.post("/api/sam2/apply_model")
def api_sam2_apply_model(payload: dict = Body(...)):
    return sa.apply_model(payload.get("session"))


@app.post("/api/sam2/rollback")
def api_sam2_rollback(payload: dict = Body(...)):
    return sa.rollback()


@app.get("/api/sam2/served")
def api_sam2_served():
    return sa.served_model() or {"none": True}


@app.get("/api/sam2/active")
def api_sam2_active():
    return sa.active_job()


@app.get("/api/sam2/models")
def api_sam2_models():
    return sa.list_models()


@app.post("/api/sam2/rollback_to")
def api_sam2_rollback_to(payload: dict = Body(...)):
    return sa.rollback_to(payload.get("model_id"))


@app.get("/api/sam2/eval_frames")
def api_sam2_eval_frames(session: str, src: str):
    return sa.eval_frames(session, src)


@app.get("/api/sam2/eval_frame")
def api_sam2_eval_frame(session: str, src: str, idx: int = 0, w: int = 720):
    b = sa.eval_frame(session, src, idx, w)
    if b is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return Response(content=b, media_type="image/jpeg")


@app.get("/api/sam2/runs")
def api_sam2_runs(src: str):
    return sa.list_runs(src)


@app.get("/api/sam2/labeled")
def api_sam2_labeled():
    return sa.list_labeled()


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
        print("경고: dashboard/dist 없음. 먼저 빌드하세요: cd dashboard && npm install && npm run build")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
