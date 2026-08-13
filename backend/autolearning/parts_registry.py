# -*- coding: utf-8 -*-
"""parts_registry.py - 부품 등록/목록 백엔드(카테고리·부품·영상 업로드·프레임 사전 추출).

설계
- 이름·카테고리·설명은 파일로 표현할 수 없으므로 **DB 가 원본**이다(여기선 DB 필수, 폴백 없음).
- 영상·3D모델·프레임 같은 큰 파일은 내부 폴더에 두고 DB 엔 경로만 넣는다.
    영상   data/bell412/<부품>/videos/<파일>
    3D     data/bell412/<부품>/model3d/<파일>
    프레임 results/autolabels/<부품>/images/<stem>/00000.jpg   (저장소는 이 한 곳뿐)
- **프레임은 등록 시점에 백그라운드로 미리 추출**한다. 학습 화면에서 실시간으로 자르면
  화면 진입마다 OpenCV 연산을 기다려야 하고 부하가 반복된다.
- 삭제는 폴더를 지우지 않고 data/bell412/_trash 로 옮긴다(복구 가능, 기존 delete_video 와 동일 규약).
"""
from __future__ import annotations

import re
import shutil
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # backend (db 패키지)

import config
import autolabel
from db import migrate_from_files as dbsync
from db.models import Category, Part, PartVideo
from db.session import SessionLocal, rel

PARTS_ROOT = config.DATA_DIR / "bell412"
TRASH = PARTS_ROOT / "_trash"
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}
MODEL3D_EXT = {".glb", ".gltf", ".obj", ".stl", ".ply", ".fbx"}
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$")   # 폴더명 겸 YOLO 클래스명

JOBS: dict[str, dict] = {}   # 프레임 추출 진행상태 (job_id -> dict)


# ---------- 공용 ----------
def _err(msg: str) -> dict:
    return {"error": msg}


def valid_name(name: str) -> bool:
    """부품명은 폴더명·YOLO 클래스명으로 동시에 쓰이므로 영문·숫자·_·- 만 허용."""
    return bool(name and NAME_RE.match(name))


def _safe_filename(name: str) -> str:
    """업로드 파일명 정리(경로 주입·공백 방지). 확장자는 유지."""
    base = Path(str(name or "")).name                  # 디렉터리 성분 제거
    base = re.sub(r"[^A-Za-z0-9._\-]", "_", base)
    return base.lstrip(".") or f"upload_{int(time.time())}"


# ---------- 카테고리 ----------
def list_categories() -> dict:
    with SessionLocal() as s:
        rows = s.query(Category).order_by(Category.sort_order, Category.name).all()
        return {"categories": [{"id": c.id, "name": c.name} for c in rows]}


def add_category(name: str) -> dict:
    name = (name or "").strip()
    if not name:
        return _err("카테고리 이름을 입력하세요.")
    with SessionLocal() as s:
        if s.query(Category).filter_by(name=name).first():
            return _err(f"이미 있는 카테고리입니다: {name}")
        mx = s.query(Category).order_by(Category.sort_order.desc()).first()
        c = Category(name=name, sort_order=(mx.sort_order + 1) if mx else 0)
        s.add(c); s.commit()
        return {"ok": True, "id": c.id, "name": c.name}


def rename_category(cid: int, name: str) -> dict:
    name = (name or "").strip()
    if not name:
        return _err("카테고리 이름을 입력하세요.")
    with SessionLocal() as s:
        c = s.get(Category, cid)
        if not c:
            return _err("없는 카테고리입니다.")
        dup = s.query(Category).filter(Category.name == name, Category.id != cid).first()
        if dup:
            return _err(f"이미 있는 카테고리입니다: {name}")
        c.name = name; s.commit()
        return {"ok": True, "id": c.id, "name": c.name}


def delete_category(cid: int) -> dict:
    """카테고리만 삭제. 그 카테고리였던 부품은 남고 분류만 비워진다(FK SET NULL)."""
    with SessionLocal() as s:
        c = s.get(Category, cid)
        if not c:
            return _err("없는 카테고리입니다.")
        n = s.query(Part).filter_by(category_id=cid).count()
        s.delete(c); s.commit()
        return {"ok": True, "unassigned": n}


# ---------- 부품 ----------
def _part_payload(s, p: Part) -> dict:
    vids = s.query(PartVideo).filter_by(part_id=p.id).all()
    frames = sum(v.n_frames or 0 for v in vids)
    return {"id": p.id, "name": p.name,
            "category_id": p.category_id,
            "category": p.category.name if p.category else None,
            "description": p.description or "",
            "has_model3d": bool(p.model_3d_path),
            "videos": [{"stem": v.stem, "role": v.role, "frames": v.n_frames or 0} for v in vids],
            "n_videos": len(vids), "frames": frames,
            "created_at": p.created_at.isoformat() if p.created_at else None}


def list_parts() -> dict:
    with SessionLocal() as s:
        rows = s.query(Part).order_by(Part.name).all()
        return {"parts": [_part_payload(s, p) for p in rows]}


def create_part(name: str, category_id=None, description: str = "") -> dict:
    """부품 행 + 폴더(videos/) 생성. 영상은 이후 upload_video 로 붙인다."""
    name = (name or "").strip()
    if not valid_name(name):
        return _err("부품 이름은 영문·숫자·_·- 만 쓸 수 있습니다(첫 글자는 영문/숫자).")
    with SessionLocal() as s:
        if s.query(Part).filter_by(name=name).first():
            return _err(f"이미 등록된 부품입니다: {name}")
        pdir = PARTS_ROOT / name
        if pdir.exists() and any(pdir.iterdir()):
            return _err(f"같은 이름의 폴더가 이미 있습니다: {rel(pdir)}")
        cid = int(category_id) if category_id not in (None, "", "null") else None
        if cid is not None and not s.get(Category, cid):
            return _err("없는 카테고리입니다.")
        (pdir / "videos").mkdir(parents=True, exist_ok=True)
        p = Part(name=name, category_id=cid, description=(description or "").strip() or None,
                 folder=rel(pdir))
        s.add(p); s.commit()
        return {"ok": True, "id": p.id, "name": p.name, "folder": p.folder}


def update_part(pid: int, name=None, category_id="keep", description=None) -> dict:
    """이름·카테고리·설명 수정. 이름 변경은 폴더도 함께 옮긴다(경로가 DB 에 들어있으므로 재동기화)."""
    with SessionLocal() as s:
        p = s.get(Part, pid)
        if not p:
            return _err("없는 부품입니다.")
        if name and name.strip() != p.name:
            new = name.strip()
            if not valid_name(new):
                return _err("부품 이름은 영문·숫자·_·- 만 쓸 수 있습니다.")
            if s.query(Part).filter(Part.name == new, Part.id != pid).first():
                return _err(f"이미 등록된 부품입니다: {new}")
            old_dir, new_dir = PARTS_ROOT / p.name, PARTS_ROOT / new
            if new_dir.exists():
                return _err(f"같은 이름의 폴더가 이미 있습니다: {rel(new_dir)}")
            old_lbl = autolabel.AUTOLABELS / p.name
            if old_dir.exists():
                shutil.move(str(old_dir), str(new_dir))
            if old_lbl.exists():                     # 라벨·프레임 저장소도 같이 이동
                shutil.move(str(old_lbl), str(autolabel.AUTOLABELS / new))
            p.name, p.folder = new, rel(new_dir)
        if category_id != "keep":
            cid = int(category_id) if category_id not in (None, "", "null") else None
            if cid is not None and not s.get(Category, cid):
                return _err("없는 카테고리입니다.")
            p.category_id = cid
        if description is not None:
            p.description = description.strip() or None
        s.commit()
        nm = p.name
    autolabel._videos(force=True)          # 영상 인덱스 캐시 무효화(이름 바뀐 경우)
    dbsync.sync_part(nm)                   # 이동된 경로로 영상·프레임 행 재동기화
    return {"ok": True, "id": pid, "name": nm}


def delete_part(pid: int) -> dict:
    """부품 삭제. 폴더는 지우지 않고 _trash 로 옮긴다(복구 가능). DB 행은 CASCADE 로 함께 삭제."""
    with SessionLocal() as s:
        p = s.get(Part, pid)
        if not p:
            return _err("없는 부품입니다.")
        name = p.name
        s.delete(p); s.commit()            # part_videos·part_frames·sam2_annotations 는 CASCADE
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    TRASH.mkdir(parents=True, exist_ok=True)
    moved = []
    for src in (PARTS_ROOT / name, autolabel.AUTOLABELS / name):
        if src.exists():
            dst = TRASH / f"{name}_{stamp}_{src.parent.name}"
            shutil.move(str(src), str(dst))
            moved.append(rel(dst))
    autolabel._videos(force=True)
    return {"ok": True, "name": name, "moved_to": moved}


# ---------- 파일 업로드 ----------
def upload_model3d(pid: int, filename: str, data: bytes) -> dict:
    with SessionLocal() as s:
        p = s.get(Part, pid)
        if not p:
            return _err("없는 부품입니다.")
        fn = _safe_filename(filename)
        if Path(fn).suffix.lower() not in MODEL3D_EXT:
            return _err(f"3D 모델 확장자만 됩니다: {', '.join(sorted(MODEL3D_EXT))}")
        d = PARTS_ROOT / p.name / "model3d"
        d.mkdir(parents=True, exist_ok=True)
        fp = d / fn
        fp.write_bytes(data)
        p.model_3d_path = rel(fp); s.commit()
        return {"ok": True, "path": p.model_3d_path, "bytes": len(data)}


def upload_video(pid: int, filename: str, data: bytes) -> dict:
    """영상 저장 후 프레임 추출을 백그라운드로 시작. 반환된 job 으로 진행률을 폴링한다."""
    with SessionLocal() as s:
        p = s.get(Part, pid)
        if not p:
            return _err("없는 부품입니다.")
        part_name = p.name
    fn = _safe_filename(filename)
    if Path(fn).suffix.lower() not in VIDEO_EXT:
        return _err(f"영상 확장자만 됩니다: {', '.join(sorted(VIDEO_EXT))}")
    vdir = PARTS_ROOT / part_name / "videos"
    vdir.mkdir(parents=True, exist_ok=True)
    fp = vdir / fn
    if fp.exists():
        return _err(f"같은 이름의 영상이 이미 있습니다: {fn}")
    fp.write_bytes(data)
    autolabel._videos(force=True)          # 새 영상이 즉시 인덱스에 잡히게
    job = uuid.uuid4().hex[:8]
    JOBS[job] = {"job": job, "part": part_name, "video": fp.stem, "stage": "extract",
                 "running": True, "count": 0, "error": None}
    threading.Thread(target=_extract_worker, args=(job, part_name, fp), daemon=True).start()
    return {"ok": True, "job": job, "part": part_name, "video": fp.stem,
            "path": rel(fp), "bytes": len(data)}


def _extract_worker(job: str, part_name: str, vp: Path) -> None:
    """등록 시점 프레임 사전 추출 + DB 동기화. 학습 화면에서 기다리지 않게 하는 핵심."""
    j = JOBS[job]
    try:
        frames = autolabel._extract(vp, autolabel.cache_dir_of(vp))   # 저장소 한 곳(images/<stem>)
        j.update(count=len(frames), stage="sync")
        dbsync.sync_part(part_name)        # part_videos.n_frames·part_frames 행 생성
        j.update(stage="done", running=False)
    except Exception as e:   # noqa: BLE001
        j.update(stage="error", running=False, error=f"{type(e).__name__}: {e}")


def job_status(job: str) -> dict:
    return JOBS.get(job) or {"error": "없는 작업입니다."}


# ---------- 부품별 영상 관리(모달용) ----------
def list_part_videos(pid: int) -> dict:
    """영상 관리 모달 재료: 영상별 stem·역할·프레임수·용량·썸네일 소스 키."""
    with SessionLocal() as s:
        p = s.get(Part, pid)
        if not p:
            return _err("없는 부품입니다.")
        out = []
        for v in s.query(PartVideo).filter_by(part_id=p.id).order_by(PartVideo.stem).all():
            fp = config.BASE_DIR / v.path
            out.append({
                "stem": v.stem, "role": v.role, "frames": v.n_frames or 0,
                "path": v.path,
                "size_mb": round(fp.stat().st_size / 1048576, 1) if fp.exists() else None,
                "exists": fp.exists(),
                # 프레임 썸네일 조회용 키(기존 /api/autolabel/frame 의 src 규약)
                "src": f"bell412/{p.name}/videos/{v.stem}",
            })
        return {"part": p.name, "id": p.id, "videos": out}


def delete_part_video(pid: int, stem: str) -> dict:
    """부품에서 영상 하나 삭제. 원본은 부품 폴더의 _trash 로 이동(복구 가능),
    프레임·라벨·참조샷은 정리하고 DB 행도 prune 된다(sam2_autolabel.delete_video 재사용)."""
    with SessionLocal() as s:
        p = s.get(Part, pid)
        if not p:
            return _err("없는 부품입니다.")
        v = s.query(PartVideo).filter_by(part_id=p.id, stem=stem).first()
        if not v:
            return _err(f"없는 영상입니다: {stem}")
        part_name = p.name
    import sam2_autolabel as sa   # 지연 import (모듈 로드 시 SAM2 무게 회피)
    r = sa.delete_video(f"bell412/{part_name}/videos/{stem}")
    return r if r.get("ok") else _err(r.get("error", "삭제 실패"))
