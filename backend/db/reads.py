"""reads.py - 조회(read) 경로의 DB 구현.

목표: 파일 기반 함수와 **완전히 같은 모양**의 응답을 낸다. 그래야 프론트 수정 없이 갈아탈 수 있고,
verify_reads.py 로 두 구현의 결과를 기계적으로 비교해 동등성을 증명할 수 있다.

대응 관계
    autolabel.list_folders()      <-> list_folders()
    sam2_autolabel.load_shots()   <-> load_shots()
    sam2_autolabel.labeled_parts()<-> labeled_parts()
    sam2_autolabel.served_model() <-> served_model()
    sam2_autolabel.list_models()  <-> list_models()
    sam2_autolabel.parts_sessions()   <-> parts_sessions()
    sam2_autolabel.list_part_frames() <-> list_part_frames(part)

주의
- DB 에는 상대경로가 들어있지만, weights 처럼 호출측이 파일을 직접 여는 값은 절대경로 문자열로 되돌려 준다.
- 프레임 수(count)·준비여부(ready)는 part_videos.n_frames 를 쓴다(등록 시 미리 자른 결과).
"""
from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import config  # noqa: E402

from .models import Part, PartFrame, PartVideo, Sam2Annotation, TrainRun  # noqa: E402
from .session import SessionLocal, abspath  # noqa: E402


def _last_map50(curve) -> float | None:
    """학습 곡선에서 마지막 유효 mAP@0.5 (sam2_autolabel._last_map50 과 동일 규칙)."""
    for e in reversed(curve or []):
        if e.get("map50") is not None:
            return round(float(e["map50"]), 3)
    return None


def list_folders() -> list[dict]:
    """[{folder, label, videos:[{name,key,count,ready}]}] — autolabel.list_folders() 와 동일.

    folder = data/ 기준 상대경로(예: bell412/gearbox/videos)
    key    = 확장자 없는 data/ 기준 상대경로(예: bell412/gearbox/videos/train)
    """
    out: list[dict] = []
    with SessionLocal() as s:
        parts = s.scalars(select(Part).options(selectinload(Part.videos)).order_by(Part.name)).all()
        groups: dict[str, list[PartVideo]] = {}
        for p in parts:
            for v in p.videos:
                # 저장된 path 는 레포루트 기준. data/ 기준으로 다시 잘라 프론트 규약(folder/key)에 맞춘다.
                vp = abspath(v.path)
                try:
                    rel_dir = vp.parent.resolve().relative_to(config.DATA_DIR.resolve()).as_posix()
                except ValueError:
                    continue
                groups.setdefault(rel_dir, []).append(v)

        for rel_dir in sorted(groups):
            seg = rel_dir.split("/")
            label = "/".join(seg[:-1]) if seg[-1] == "videos" and len(seg) > 1 else rel_dir
            vids = []
            for v in sorted(groups[rel_dir], key=lambda x: abspath(x.path).name):
                key = Path(rel_dir + "/" + v.stem).as_posix()
                n = v.n_frames or 0
                vids.append({"name": v.stem, "key": key, "count": n, "ready": bool(n)})
            out.append({"folder": rel_dir, "label": label, "videos": vids})
    return out


def load_shots() -> dict:
    """{"<영상stem>": {"<프레임>": [[rx,ry,lab],...]}} — sam2_autolabel.load_shots() 와 동일.

    파일판이 부품을 가로질러 stem 키로 합치므로(같은 stem 이면 병합) 여기서도 같은 규약을 지킨다.
    """
    out: dict[str, dict] = {}
    with SessionLocal() as s:
        rows = s.execute(
            select(PartVideo.stem, PartFrame.frame_number, Sam2Annotation.tap_points)
            .join(PartFrame, PartFrame.video_id == PartVideo.id)
            .join(Sam2Annotation, Sam2Annotation.frame_id == PartFrame.id)
            .where(Sam2Annotation.is_reference.is_(True))
            .order_by(PartVideo.stem, PartFrame.frame_number)
        ).all()
    for stem, no, taps in rows:
        if not taps:
            continue
        pts = [[t["rx"], t["ry"], int(t["label"])] for t in taps]
        out.setdefault(stem, {})[str(no)] = pts
    return out


def labeled_parts() -> dict:
    """{"parts": [...]} — 라벨(박스)이 하나라도 있는 부품명 정렬 목록."""
    with SessionLocal() as s:
        names = s.scalars(
            select(Part.name)
            .join(PartVideo, PartVideo.part_id == Part.id)
            .join(PartFrame, PartFrame.video_id == PartVideo.id)
            .join(Sam2Annotation, Sam2Annotation.frame_id == PartFrame.id)
            .where(Sam2Annotation.label_path.is_not(None))
            .group_by(Part.name)
            .order_by(Part.name)
        ).all()
    return {"parts": list(names)}


def _weights(r: TrainRun):
    """저장된 경로가 다른 기계의 절대경로여서 존재하지 않으면 run 폴더 규약으로 되살린다."""
    w = abspath(r.weights_path) if r.weights_path else None
    if w and Path(w).exists():
        return w
    p = config.BASE_DIR / "results" / str(r.model_id) / "model" / "best.pt"
    return p if p.exists() else None


def _run_payload(r: TrainRun) -> dict:
    meta = r.meta or {}
    w = _weights(r)
    return {"model_id": r.model_id, "label": r.label or "", "time": (meta.get("time") or ""),
            "classes": r.classes or [],
            "n_classes": r.n_classes if r.n_classes is not None else len(r.classes or []),
            "n_images": r.n_images,
            "map50": r.map50 if r.map50 is not None else _last_map50(meta.get("curve")),
            "gen_rate": meta.get("gen_rate"), "newp_rate": meta.get("newp_rate"),
            "has_pt": bool(w and w.exists()),
            "is_active": bool(r.is_active)}


def list_models() -> dict:
    """{"models": [...최신순], "active": model_id|None} — sam2_autolabel.list_models() 와 동일."""
    with SessionLocal() as s:
        runs = s.scalars(select(TrainRun).order_by(TrainRun.model_id.desc())).all()
        active = next((r.model_id for r in runs if r.is_active), None)
        return {"models": [_run_payload(r) for r in runs], "active": active}


def served_model() -> dict | None:
    """현재 서비스 모델. 가중치 파일이 실제로 있어야 유효(파일판과 동일 판정)."""
    with SessionLocal() as s:
        r = s.scalars(select(TrainRun).where(TrainRun.is_active.is_(True))).first()
        if not r:
            return None
        w = _weights(r)
        if not (w and w.exists()):
            return None
        meta = r.meta or {}
        cls = r.classes or []
        return {"weights": str(w), "classes": cls, "n_classes": len(cls),
                "label": r.label or "", "session": r.model_id, "applied": r.model_id,
                "model_id": r.model_id, "time": (meta.get("time") or ""),
                "map50": r.map50 if r.map50 is not None else _last_map50(meta.get("curve")),
                "gen_rate": meta.get("gen_rate"), "newp_rate": meta.get("newp_rate")}


def counts() -> dict:
    """헬스체크·디버그용 행 수 요약."""
    with SessionLocal() as s:
        return {t.__tablename__: s.scalar(select(func.count()).select_from(t))
                for t in (Part, PartVideo, PartFrame, Sam2Annotation, TrainRun)}

def parts_sessions() -> list[dict]:
    """sam2_autolabel.parts_sessions() 의 DB 구현.

    영상별 (라벨 수, 프레임 수) 와 '이미 학습된 영상' 목록. 라벨 수는 label_path 가 있는
    sam2_annotations 개수, 프레임 수는 part_frames 개수로 센다.
    """
    with SessionLocal() as s:
        rows = s.execute(
            select(PartVideo.stem,
                   func.count(PartFrame.id),
                   func.count(Sam2Annotation.label_path))
            .join(PartFrame, PartFrame.video_id == PartVideo.id, isouter=True)
            .join(Sam2Annotation, Sam2Annotation.frame_id == PartFrame.id, isouter=True)
            .group_by(PartVideo.stem)
            .order_by(PartVideo.stem)
        ).all()
    # 파일판은 labels·frames 에 같은 값(라벨 수)을 넣었다. 여기서는 실제 프레임 수를 준다.
    # 프론트는 labels > 0 으로 학습 대상을 고르므로 동작은 같고, 표시되는 프레임 수만 정확해진다.
    videos = {stem: {"labels": int(nlbl), "frames": int(nfr)} for stem, nfr, nlbl in rows}

    trained = []                                  # 현재 서비스 모델이 가진 부품의 영상
    sv = served_model()
    if sv:
        classes = set(sv.get("classes", []))
        with SessionLocal() as s:
            pairs = s.execute(
                select(PartVideo.stem, Part.name).join(Part, Part.id == PartVideo.part_id)
            ).all()
        trained = [stem for stem, pname in pairs if pname in classes]
    return [{"session": "autolabel", "videos": videos, "n_videos": len(videos),
             "total_labels": sum(v["labels"] for v in videos.values()),
             "trained": trained, "updated": ""}]


def list_part_frames(part: str, limit: int = 400) -> dict:
    """sam2_autolabel.list_part_frames() 의 DB 구현. 라벨이 만들어진 프레임 목록(검수용)."""
    if not part:
        return {"part": part, "count": 0, "frames": []}
    with SessionLocal() as s:
        names = s.scalars(
            select(Sam2Annotation.label_path)
            .join(PartFrame, PartFrame.id == Sam2Annotation.frame_id)
            .join(PartVideo, PartVideo.id == PartFrame.video_id)
            .join(Part, Part.id == PartVideo.part_id)
            .where(Part.name == part, Sam2Annotation.label_path.is_not(None))
            .order_by(Sam2Annotation.label_path)
            .limit(limit + 1)
        ).all()
    out = [{"session": "autolabel", "name": Path(n).stem + ".jpg", "part": part} for n in names[:limit]]
    res = {"part": part, "count": len(out), "frames": out}
    if len(names) > limit:
        res["truncated"] = True
    return res

def xr_parts() -> dict:
    """외부 연동(XR)용 부품 목록. 이름·카테고리·3D 모델만 준다.

    규격
      name       부품 이름. 추론 응답의 detection_class 와 같은 값 -> 판별 키
      category   Gearbox / Tools (없으면 null)
      model3d    3D 모델 내려받기 URL (없으면 null)
    DB 에는 파일 경로만 있고 바이트는 파일시스템에 있으므로, 경로 대신 URL 을 준다.
    자동증가 id 는 기기마다 값이 달라 내보내지 않는다.
    """
    with SessionLocal() as s:
        rows = [(p.name, (p.category.name if p.category else None), p.model_3d_path)
                for p in s.query(Part).options(selectinload(Part.category)).order_by(Part.name).all()]
    parts = [{"name": n, "category": c,
              "model3d": (f"/api/xr/parts/{n}/model3d" if m3 else None)}
             for n, c, m3 in rows]
    return {"count": len(parts), "parts": parts}


def xr_model3d_path(name: str):
    """부품 이름 -> 3D 모델 파일의 절대경로(없으면 None). 파일 서빙용."""
    with SessionLocal() as s:
        p = s.query(Part).filter_by(name=name).first()
        if not p or not p.model_3d_path:
            return None
    fp = abspath(p.model_3d_path)
    return fp if fp.exists() else None
