# -*- coding: utf-8 -*-
"""scripts/verify/dashboard_api.py - 대시보드 API 서버 (FastAPI) + React 프론트 서빙.

프론트 소스: dashboard/ (React + Vite). 빌드 산출물 dashboard/dist 를 루트에 서빙.
데이터 계층은 scripts/dashboard_core.py 공용 모듈.

실행:
  cd dashboard && npm install && npm run build   # 프론트 변경 시 1회
  python scripts/verify/dashboard_api.py             # http://127.0.0.1:7862
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ 공용
sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
import dashboard_core as core

PORT = 7862
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
            "tech": m.get("tech", []), "flow": m.get("flow", []),
            "bullets": m["bullets"], "code": m.get("code", []),
            "gallery": core.method_gallery(m),
            "metrics": core.method_metrics(m)}


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
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
