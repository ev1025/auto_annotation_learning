"""0_import_render.py - 3D 렌더 합성 데이터 반입(클래스 등록 + 라벨 검증 + 배치).

3D 렌더 공급분은 라벨(.txt)의 클래스가 '번호'로만 적혀 있어서,
그 번호가 무슨 부품인지(0=bolt? 0=gear?)는 공급자와 약속으로만 존재한다.
이 스크립트가 그 약속을 파이프라인에 공식 등록하는 관문 역할을 한다.

하는 일:
  1) 클래스 정의(번호->부품명)를 받아 data.yaml 의 names 로 등록
     - --classes bolt nut gear      (나열 순서 = 0,1,2 번)
     - --classes-file classes.txt   (한 줄에 부품명 하나, 줄 순서 = 번호)
  2) 라벨 검증: 클래스 번호가 정의 범위 안인지 / 좌표가 0~1 인지 / 필드 5개인지
  3) 검증 통과분을 datasets/images + datasets/labels 로 복사(파이프라인 표준 위치)
  4) 클래스별 박스 분포를 출력해 "번호를 잘못 알고 반입"하는 사고를 조기 발견

입력 폴더 구조(둘 다 지원):
  <src>/images/*.jpg + <src>/labels/*.txt   또는   <src>/ 에 .jpg 와 .txt 나란히

실행:
  python 0_import_render.py --src ./render_delivery --classes bolt nut gear
  python 0_import_render.py --src ./render_delivery --classes-file ./render_delivery/classes.txt
"""
import argparse
import shutil
from collections import Counter
from pathlib import Path

import yaml

import config


def load_classes(args):
    """클래스 정의를 {번호: 부품명} 으로 만든다. 나열/파일 둘 중 하나는 필수."""
    if args.classes:
        names = [c.strip().lower() for c in args.classes]
    elif args.classes_file:
        lines = Path(args.classes_file).read_text(encoding="utf-8").splitlines()
        names = [ln.strip().lower() for ln in lines if ln.strip()]
    else:
        raise SystemExit("[오류] --classes 또는 --classes-file 로 클래스 정의를 지정하세요.\n"
                         "      예) --classes bolt nut gear   (순서 = 0,1,2번)")
    if len(names) != len(set(names)):
        raise SystemExit(f"[오류] 클래스명이 중복됩니다: {names}")
    return {i: n for i, n in enumerate(names)}


def find_pairs(src):
    """(이미지, 라벨) 짝 목록. labels/ 하위 또는 이미지 옆의 같은 stem .txt 를 찾는다."""
    pairs, missing = [], []
    imgs = [p for p in sorted(src.rglob("*"))
            if p.suffix.lower() in config.IMG_EXTS and "labels" not in p.parts]
    for img in imgs:
        cands = [img.with_suffix(".txt"),                       # 이미지 옆
                 src / "labels" / f"{img.stem}.txt"]            # labels/ 하위
        lbl = next((c for c in cands if c.exists()), None)
        (pairs if lbl else missing).append((img, lbl))
    return pairs, [m[0] for m in missing]


def validate_label(lbl_path, n_classes):
    """라벨 한 파일 검증. (정상 줄 목록, 오류 목록) 반환. 좌표는 0~1 로 클램프."""
    good, errors = [], []
    for i, ln in enumerate(lbl_path.read_text(encoding="utf-8").splitlines(), 1):
        parts = ln.split()
        if not parts:
            continue
        if len(parts) < 5:
            errors.append(f"{lbl_path.name}:{i} 필드 부족({len(parts)}개)")
            continue
        try:
            cid = int(float(parts[0]))
            vals = [float(x) for x in parts[1:5]]
        except ValueError:
            errors.append(f"{lbl_path.name}:{i} 숫자 아님")
            continue
        if not (0 <= cid < n_classes):
            # 클래스 번호가 약속 범위를 벗어남 = 공급자와 번호 약속이 어긋났다는 신호.
            errors.append(f"{lbl_path.name}:{i} 클래스 {cid} 가 정의 범위(0~{n_classes-1}) 밖")
            continue
        vals = [min(max(v, 0.0), 1.0) for v in vals]  # 경계 살짝 벗어난 좌표는 클램프
        good.append((cid, vals))
    return good, errors


def sync_data_yaml(class_names):
    """data.yaml 의 names 를 반입 클래스로 등록. 기존과 다르면 멈추고 확인 요구."""
    cfg = {}
    if config.DATA_YAML.exists():
        cfg = yaml.safe_load(config.DATA_YAML.read_text(encoding="utf-8")) or {}
        old = cfg.get("names")
        old_names = (list(old.values()) if isinstance(old, dict) else old) if old else None
        if old_names and [n.lower() for n in old_names] != list(class_names.values()):
            raise SystemExit(
                f"[중단] data.yaml 에 이미 다른 클래스가 등록돼 있습니다.\n"
                f"      기존: {old_names}\n"
                f"      반입: {list(class_names.values())}\n"
                f"      기존 데이터셋과 섞이면 라벨 번호가 꼬입니다. data.yaml/datasets 를 정리 후 재실행하세요.")
    cfg.setdefault("path", "./datasets")
    cfg.setdefault("train", "images/train")
    cfg.setdefault("val", "images/val")
    cfg["names"] = class_names
    config.DATA_YAML.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                                encoding="utf-8")
    print(f"[등록] data.yaml names <- {class_names}")


def main():
    ap = argparse.ArgumentParser(description="3D 렌더 데이터 반입(클래스 등록+검증+배치)")
    ap.add_argument("--src", required=True, help="공급받은 렌더 데이터 폴더")
    ap.add_argument("--classes", nargs="+", default=None, help="클래스명 나열(순서=번호). 예: bolt nut gear")
    ap.add_argument("--classes-file", default=None, help="클래스 정의 파일(한 줄=한 부품명, 줄 순서=번호)")
    ap.add_argument("--dry-run", action="store_true", help="복사 없이 검증·분포만 출력")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    if not src.is_dir():
        raise SystemExit(f"[오류] 폴더가 없습니다: {src}")

    class_names = load_classes(args)
    pairs, missing = find_pairs(src)
    if not pairs:
        raise SystemExit(f"[오류] {src} 에서 (이미지, 라벨) 짝을 찾지 못했습니다.")

    # 검증 + 분포 집계
    dist = Counter()
    all_errors = []
    validated = []  # (img, [(cid, vals)...])
    for img, lbl in pairs:
        good, errors = validate_label(lbl, len(class_names))
        all_errors.extend(errors)
        if good:
            validated.append((img, good))
            for cid, _ in good:
                dist[class_names[cid]] += 1

    print(f"[검증] 이미지 {len(pairs)}장 / 라벨 없는 이미지 {len(missing)}장 / 오류 줄 {len(all_errors)}개")
    for e in all_errors[:10]:
        print(f"  - {e}")
    if len(all_errors) > 10:
        print(f"  ... 외 {len(all_errors) - 10}개")
    print(f"[분포] 클래스별 박스 수: {dict(dist)}")
    zero = [n for n in class_names.values() if dist[n] == 0]
    if zero:
        print(f"[경고] 박스가 0개인 클래스: {zero} -> 번호 약속이 어긋났는지 확인 필요")

    if args.dry_run:
        print("\n(dry-run) 복사·등록 없이 종료")
        return

    # data.yaml 등록 + 표준 위치로 복사
    sync_data_yaml(class_names)
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    config.LABELS_DIR.mkdir(parents=True, exist_ok=True)
    for img, good in validated:
        shutil.copy2(img, config.IMAGES_DIR / img.name)
        lines = [f"{cid} " + " ".join(f"{v:.6f}" for v in vals) for cid, vals in good]
        (config.LABELS_DIR / f"{img.stem}.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n완료: {len(validated)}장 반입 -> {config.IMAGES_DIR.parent}")
    print("다음: python 2_train_pipeline.py  (첫 모델 학습)")


if __name__ == "__main__":
    main()
