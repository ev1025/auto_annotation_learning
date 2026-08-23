"""migrate_from_files.py - 기존 파일 구조를 DB 로 이관(여러 번 돌려도 안전한 멱등 스크립트).

읽는 원본(현행 구조, 그대로 보존한다):
    data/bell412/<부품>/videos/<stem>.mp4          등록 부품·촬영 테이크
    results/autolabels/<부품>/images/<stem>/*.jpg  미리 잘라둔 프레임
    results/autolabels/<부품>/labels/<stem>_N.txt  YOLO 정답(전파 결과)
    results/autolabels/<부품>/boxs/<stem>_N.jpg    박스 미리보기
    results/autolabels/<부품>/shots.json           참조샷 탭 좌표 {stem: {frame: [[rx,ry,label],...]}}
    results/<model_id>/meta.json                   학습 1회 결과
    results/_served.json                           현재 서비스 모델 포인터

사용:
    python backend/db/migrate_from_files.py            # 이관 실행
    python backend/db/migrate_from_files.py --dry-run  # 쓰지 않고 집계만
    python backend/db/migrate_from_files.py --reset    # 테이블 드롭 후 재생성(주의: DB 내용 삭제)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import config  # noqa: E402
from models import Base, Category, Part, PartFrame, PartVideo, Sam2Annotation, TrainRun  # noqa: E402
from session import SessionLocal, engine, rel  # noqa: E402

AUTOLABELS = config.BASE_DIR / "results" / "autolabels"
PARTS_ROOT = config.DATA_DIR / "bell412"
RESULTS = config.BASE_DIR / "results"
DEFAULT_CATEGORIES = ["공구", "계측기", "전장 부품", "기타"]   # UI 기본값과 동일


def frames_dir_for(part_dir: Path, stem: str) -> Path | None:
    """영상의 프레임 저장소(results/autolabels/<부품>/images/<stem>). 저장소는 한 곳뿐이다.

    레거시 _frame_cache(부품별·중앙) 폴백은 제거했다 — 잔여 프레임을 이 저장소로 이사 완료(2026-08-13).
    폴백을 남기면 폐지한 경로를 DB 에 다시 고착시키게 된다.
    """
    d = AUTOLABELS / part_dir.name / "images" / stem
    if d.exists() and any(f.suffix.lower() in config.IMG_EXTS for f in d.iterdir()):
        return d
    return None


def parse_label_txt(p: Path) -> list[dict] | None:
    """YOLO txt -> [{"cls":0,"cx":..,"cy":..,"w":..,"h":..}] (정규화 좌표 그대로)."""
    try:
        rows = []
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            rows.append({"cls": int(float(parts[0])), "cx": float(parts[1]), "cy": float(parts[2]),
                         "w": float(parts[3]), "h": float(parts[4])})
        return rows or None
    except Exception:
        return None


def parse_ts(model_id: str, meta: dict) -> datetime | None:
    """meta.json time('2026-08-13 10:54:30') 우선, 없으면 model_id(YYMMDD_HHMMSS)에서 추출."""
    t = meta.get("time")
    if t:
        try:
            return datetime.strptime(t, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    try:
        return datetime.strptime(model_id, "%y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def upsert_categories(s) -> None:
    for i, name in enumerate(DEFAULT_CATEGORIES):
        if not s.query(Category).filter_by(name=name).first():
            s.add(Category(name=name, sort_order=i))
    s.flush()


def migrate_parts(s, stats: dict) -> None:
    """부품 폴더(data/bell412/<부품>/videos) 전체를 이관. 부품 하나씩은 sync_part_dir 이 처리."""
    if not PARTS_ROOT.exists():
        print(f"  ! 부품 루트 없음: {PARTS_ROOT}")
        return
    for pdir in sorted(p for p in PARTS_ROOT.iterdir() if p.is_dir()):
        if pdir.name in ("backgrounds", "_synth"):   # 합성 증강 자원(배경·가림) — 부품이 아님
            continue
        sync_part_dir(s, pdir, stats)


def sync_part_dir(s, pdir: Path, stats: dict) -> None:
    """부품 폴더 하나를 DB 에 반영(멱등 upsert).

    라벨 생성 직후에도 이 함수를 그대로 호출해 DB 를 최신화한다(파일이 원본, DB 는 동기 색인).
    개별 INSERT 를 코드 곳곳에 흩뿌리는 대신 검증된 이 경로 하나만 쓴다.
    """
    vdir = pdir / "videos"
    if not vdir.exists():
        return
    vids = sorted(v for v in vdir.iterdir() if v.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"})
    if vids:
        part = s.query(Part).filter_by(name=pdir.name).first()
        if not part:
            part = Part(name=pdir.name, folder=rel(pdir))
            s.add(part)
            stats["parts_new"] += 1
        else:
            part.folder = rel(pdir)
        s.flush()
        stats["parts"] += 1

        shots = {}
        sj = AUTOLABELS / pdir.name / "shots.json"
        if sj.exists():
            try:
                shots = json.loads(sj.read_text(encoding="utf-8"))
            except Exception:
                shots = {}

        for v in vids:
            video = s.query(PartVideo).filter_by(part_id=part.id, stem=v.stem).first()
            if not video:
                video = PartVideo(part_id=part.id, stem=v.stem, path=rel(v), role="train")
                s.add(video)
                stats["videos_new"] += 1
            else:
                video.path, video.role = rel(v), "train"
            s.flush()
            stats["videos"] += 1

            img_dir = frames_dir_for(pdir, v.stem)
            if img_dir is None:
                continue
            frames = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in config.IMG_EXTS)
            video.n_frames = len(frames)

            taps_by_frame = shots.get(v.stem, {}) if isinstance(shots, dict) else {}
            lbl_dir = AUTOLABELS / pdir.name / "labels"
            box_dir = AUTOLABELS / pdir.name / "boxs"

            for f in frames:
                try:
                    no = int(f.stem)                     # 00000.jpg -> 0
                except ValueError:
                    continue
                fr = s.query(PartFrame).filter_by(video_id=video.id, frame_number=no).first()
                if not fr:
                    fr = PartFrame(video_id=video.id, frame_number=no, image_path=rel(f))
                    s.add(fr)
                    stats["frames_new"] += 1
                else:
                    fr.image_path = rel(f)
                s.flush()
                stats["frames"] += 1

                raw_taps = taps_by_frame.get(str(no))
                taps = ([{"rx": t[0], "ry": t[1], "label": int(t[2])} for t in raw_taps if len(t) >= 3]
                        if raw_taps else None)
                lbl = lbl_dir / f"{v.stem}_{no:05d}.txt"
                box_prev = box_dir / f"{v.stem}_{no:05d}.jpg"
                boxes = parse_label_txt(lbl) if lbl.exists() else None
                if not taps and not boxes:
                    # 라벨도 탭도 없다. 예전에 있었다가 지워진 경우 행이 남아 DB 가 디스크보다 많아진다
                    # (재탭으로 라벨을 다시 만들면 파일만 지워지므로). 그래서 여기서 지운다.
                    old_ann = s.query(Sam2Annotation).filter_by(frame_id=fr.id).first()
                    if old_ann:
                        s.delete(old_ann)
                        stats["ann_gone"] = stats.get("ann_gone", 0) + 1
                    continue

                ann = s.query(Sam2Annotation).filter_by(frame_id=fr.id).first()
                if not ann:
                    ann = Sam2Annotation(frame_id=fr.id)
                    s.add(ann)
                    stats["ann_new"] += 1
                ann.tap_points = taps
                ann.boxes = boxes
                ann.is_reference = bool(taps)
                ann.label_path = rel(lbl) if lbl.exists() else None
                ann.box_preview_path = rel(box_prev) if box_prev.exists() else None
                stats["ann"] += 1
                if taps:
                    stats["ann_ref"] += 1

            # 파일에서 사라진 프레임 행 정리(영상 재추출로 장수가 줄어든 경우)
            keep_nos = set()
            for f in frames:
                try:
                    keep_nos.add(int(f.stem))
                except ValueError:
                    pass
            gone = s.query(PartFrame).filter(PartFrame.video_id == video.id,
                                             PartFrame.frame_number.notin_(keep_nos or {-1})).all()
            for fr in gone:
                s.delete(fr)          # sam2_annotations 는 CASCADE
                stats["frames_del"] += 1

        # 파일에서 사라진 영상 행 정리(영상 삭제 시). 이게 없으면 목록에 유령 영상이 남는다.
        keep_stems = {v.stem for v in vids}
        for v in s.query(PartVideo).filter(PartVideo.part_id == part.id).all():
            if v.stem not in keep_stems:
                s.delete(v)           # part_frames·sam2_annotations 는 CASCADE
                stats["videos_del"] += 1
        s.flush()


def migrate_runs(s, stats: dict) -> None:
    """results/<model_id>/meta.json -> train_runs, _served.json -> is_active."""
    served_id = None
    sf = RESULTS / "_served.json"
    if sf.exists():
        try:
            served_id = (json.loads(sf.read_text(encoding="utf-8")) or {}).get("run")
        except Exception:
            served_id = None

    for d in sorted(p for p in RESULTS.iterdir() if p.is_dir() and p.name != "autolabels"):
        mj = d / "meta.json"
        if not mj.exists():
            continue
        try:
            meta = json.loads(mj.read_text(encoding="utf-8"))
        except Exception:
            continue
        mid = meta.get("model_id") or d.name

        run = s.query(TrainRun).filter_by(model_id=mid).first()
        if not run:
            run = TrainRun(model_id=mid)
            s.add(run)
            stats["runs_new"] += 1
        run.session = meta.get("session")
        run.label = meta.get("label")
        run.trained_at = parse_ts(mid, meta)
        run.classes = meta.get("classes")
        run.n_classes = meta.get("n_classes")
        run.n_images = meta.get("n_images")
        run.per_class = meta.get("per_class")
        run.learn_rate = meta.get("learn_rate")
        run.epochs = meta.get("epochs")
        run.map50 = meta.get("map50")
        # 절대경로(C:\Users\...)로 저장돼 있던 것을 상대경로로 정규화 — Thor 에서도 유효하게
        # meta["weights"] 는 학습 기계의 절대경로일 수 있다 -> run 폴더 규약을 우선한다
        wp = d / "model" / "best.pt"
        run.weights_path = rel(wp) if wp.exists() else (rel(meta["weights"]) if meta.get("weights") else None)
        run.onnx_path = rel(meta["onnx"]) if meta.get("onnx") else None
        run.meta = meta
        run.is_active = False
        stats["runs"] += 1
    s.flush()

    if served_id:
        active = s.query(TrainRun).filter_by(model_id=served_id).first()
        if active:
            active.is_active = True
            try:
                applied = (json.loads(sf.read_text(encoding="utf-8")) or {}).get("applied")
                if applied:
                    active.applied_at = datetime.strptime(applied, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            except Exception:
                pass
            stats["active"] = served_id
        else:
            print(f"  ! _served.json 이 가리키는 {served_id} 의 meta.json 이 없어 활성 표시 못 함")


# ─────────────────────────────────────────────────────────────────────
# 쓰기 경로에서 호출하는 공개 함수(파일에 쓴 직후 DB 를 최신화).
# 절대 예외를 밖으로 던지지 않는다 — DB 문제로 라벨 생성·학습이 실패하면 안 된다.
# ─────────────────────────────────────────────────────────────────────
_EMPTY_STATS = dict(parts=0, parts_new=0, videos=0, videos_new=0, videos_del=0, frames=0, frames_new=0,
                    frames_del=0, ann=0, ann_new=0, ann_gone=0, ann_ref=0, runs=0, runs_new=0, active=None)


def sync_part(part_name: str) -> dict | None:
    """부품 하나(폴더명=클래스명)를 DB 에 동기화. 라벨 생성 직후 호출.
    반환: 통계 dict (실패 시 None)."""
    try:
        pdir = PARTS_ROOT / part_name
        if not pdir.is_dir():
            return None
        stats = dict(_EMPTY_STATS)
        with SessionLocal() as s:
            sync_part_dir(s, pdir, stats)
            s.commit()
        return stats
    except Exception as e:   # noqa: BLE001
        print(f"[DB] sync_part({part_name}) 실패(무시하고 계속): {type(e).__name__}: {e}", flush=True)
        return None


def sync_runs() -> dict | None:
    """학습 이력·현재 서비스 모델을 DB 에 동기화. 학습 완료·적용·롤백 직후 호출."""
    try:
        stats = dict(_EMPTY_STATS)
        with SessionLocal() as s:
            migrate_runs(s, stats)
            s.commit()
        return stats
    except Exception as e:   # noqa: BLE001
        print(f"[DB] sync_runs 실패(무시하고 계속): {type(e).__name__}: {e}", flush=True)
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="쓰지 않고 집계만(롤백)")
    ap.add_argument("--reset", action="store_true", help="테이블 드롭 후 재생성(DB 내용 삭제)")
    args = ap.parse_args()

    if args.reset:
        print("! 테이블 드롭 후 재생성")
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    stats = dict(_EMPTY_STATS)

    with SessionLocal() as s:
        upsert_categories(s)
        print("부품·영상·프레임·SAM2 이관...")
        migrate_parts(s, stats)
        print("학습 이력 이관...")
        migrate_runs(s, stats)
        if args.dry_run:
            s.rollback()
            print("\n[dry-run] 롤백함(DB 변경 없음)")
        else:
            s.commit()

    print("\n=== 이관 결과 ===")
    print(f"  부품      {stats['parts']:5d} (신규 {stats['parts_new']})")
    print(f"  영상      {stats['videos']:5d} (신규 {stats['videos_new']}, 정리 {stats['videos_del']})")
    print(f"  프레임    {stats['frames']:5d} (신규 {stats['frames_new']}, 정리 {stats['frames_del']})")
    print(f"  SAM2주석  {stats['ann']:5d} (신규 {stats['ann_new']}, 참조샷 {stats['ann_ref']})")
    print(f"  학습이력  {stats['runs']:5d} (신규 {stats['runs_new']})")
    print(f"  서비스모델 {stats['active']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
