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


def trail_num(name: str) -> int:
    """이름 끝 숫자(테이크 번호). 없으면 0 — 프론트 trailNum 과 같은 규칙."""
    m = re.search(r"(\d+)\s*$", Path(name).stem)
    return int(m.group(1)) if m else 0


def take_roles(stems: list[str]) -> dict[str, str]:
    """프론트 takeRoles() 와 동일 규칙으로 stem -> 'train'|'test' 판정.
    - 이름에 test 가 있으면 그것들 중 마지막이 test, 나머지는 train
    - 없고 2개 이상이면 끝번호 가장 큰 것이 test
    - 1개면 그 영상이 train 겸 test(단일 테이크) -> 여기서는 train 으로 기록
    """
    if not stems:
        return {}
    explicit = [s for s in stems if re.search(r"test", s, re.I)]
    if explicit:
        test = explicit[-1]
    elif len(stems) >= 2:
        test = sorted(stems, key=trail_num)[-1]
    else:
        return {stems[0]: "train"}
    return {s: ("test" if s == test else "train") for s in stems}


def frames_dir_for(part_dir: Path, stem: str) -> Path | None:
    """영상의 프레임이 실제로 있는 폴더. 파일판(_source_info)과 같은 폴백 순서로 찾는다.
      1) results/autolabels/<부품>/images/<stem>   신 저장소
      2) data/bell412/<부품>/_frame_cache/<stem>   레거시(부품별)
      3) data/_frame_cache/<stem>                 레거시(중앙)
    이 순서를 안 맞추면 예전에 잘라둔 부품이 DB 에서 '프레임 0' 으로 보인다.
    """
    for d in (AUTOLABELS / part_dir.name / "images" / stem,
              part_dir / "_frame_cache" / stem,
              config.DATA_DIR / "_frame_cache" / stem):
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
    """부품 폴더(data/bell412/<부품>/videos) 기준으로 부품·영상·프레임·SAM2 이관."""
    if not PARTS_ROOT.exists():
        print(f"  ! 부품 루트 없음: {PARTS_ROOT}")
        return

    for pdir in sorted(p for p in PARTS_ROOT.iterdir() if p.is_dir()):
        if pdir.name == "backgrounds":          # 합성 증강용 배경 — 부품이 아님
            continue
        vdir = pdir / "videos"
        if not vdir.exists():
            continue
        vids = sorted(v for v in vdir.iterdir() if v.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"})
        if not vids:
            continue

        part = s.query(Part).filter_by(name=pdir.name).first()
        if not part:
            part = Part(name=pdir.name, folder=rel(pdir))
            s.add(part)
            stats["parts_new"] += 1
        else:
            part.folder = rel(pdir)
        s.flush()
        stats["parts"] += 1

        roles = take_roles([v.stem for v in vids])
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
                video = PartVideo(part_id=part.id, stem=v.stem, path=rel(v), role=roles.get(v.stem, "train"))
                s.add(video)
                stats["videos_new"] += 1
            else:
                video.path, video.role = rel(v), roles.get(v.stem, "train")
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
                    continue                              # 라벨도 탭도 없는 프레임은 주석 행 만들지 않음

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
        run.weights_path = rel(meta["weights"]) if meta.get("weights") else None
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="쓰지 않고 집계만(롤백)")
    ap.add_argument("--reset", action="store_true", help="테이블 드롭 후 재생성(DB 내용 삭제)")
    args = ap.parse_args()

    if args.reset:
        print("! 테이블 드롭 후 재생성")
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    stats = dict(parts=0, parts_new=0, videos=0, videos_new=0, frames=0, frames_new=0,
                 ann=0, ann_new=0, ann_ref=0, runs=0, runs_new=0, active=None)

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
    print(f"  영상      {stats['videos']:5d} (신규 {stats['videos_new']})")
    print(f"  프레임    {stats['frames']:5d} (신규 {stats['frames_new']})")
    print(f"  SAM2주석  {stats['ann']:5d} (신규 {stats['ann_new']}, 참조샷 {stats['ann_ref']})")
    print(f"  학습이력  {stats['runs']:5d} (신규 {stats['runs_new']})")
    print(f"  서비스모델 {stats['active']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
