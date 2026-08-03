"""scripts/data_import/coco_to_yolo.py - Roboflow COCO export 를 YOLOv8 export 와 같은 구조로 변환.

입력(Roboflow COCO):
    <src>/train/_annotations.coco.json + 이미지들   (valid/, test/ 동일)
출력(YOLOv8 export 동일 구조 -> scripts/data_import/import_roboflow.py / scripts/experiments/experiment_autolearn.py 가 그대로 소비):
    <dst>/train/images/*.jpg
    <dst>/train/labels/*.txt      (class cx cy w h, 0~1 정규화)
    <dst>/data.yaml               (names)

변환 규칙:
  - COCO bbox = [x_min, y_min, w, h] 픽셀  ->  YOLO = 중심좌표+크기 / 이미지 크기로 정규화
  - Roboflow COCO 는 id 0 에 더미 카테고리(supercategory)를 넣는다.
    어노테이션이 실제로 달린 카테고리만 골라 0..k-1 로 재매핑한다.

실행:
  python scripts/data_import/coco_to_yolo.py --src ./data/robo/coco --dst ./data/robo/yolo
"""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import yaml


def convert_split(split_dir, out_dir, cat_map):
    """한 split(train/valid/test)을 변환. (이미지수, 라벨수) 반환."""
    ann_file = split_dir / "_annotations.coco.json"
    coco = json.loads(ann_file.read_text(encoding="utf-8"))

    img_out = out_dir / "images"
    lbl_out = out_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    # image_id -> (파일명, W, H)
    imgs = {im["id"]: (im["file_name"], im["width"], im["height"]) for im in coco["images"]}

    # image_id 별로 어노테이션을 모아 한 번에 .txt 로 쓴다.
    lines_per_img = defaultdict(list)
    n_box = 0
    for a in coco["annotations"]:
        cid = a["category_id"]
        if cid not in cat_map:      # 더미/제외 카테고리는 버림
            continue
        fn, W, H = imgs[a["image_id"]]
        x, y, w, h = a["bbox"]      # 픽셀 [x_min, y_min, w, h]
        # YOLO: 중심좌표/크기를 0~1 로 정규화. 경계 밖 값은 클램프.
        cx = min(max((x + w / 2) / W, 0.0), 1.0)
        cy = min(max((y + h / 2) / H, 0.0), 1.0)
        nw = min(max(w / W, 0.0), 1.0)
        nh = min(max(h / H, 0.0), 1.0)
        lines_per_img[a["image_id"]].append(
            f"{cat_map[cid]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        n_box += 1

    n_img = 0
    for img_id, (fn, _, _) in imgs.items():
        src_img = split_dir / fn
        if not src_img.exists():
            continue
        shutil.copy2(src_img, img_out / fn)
        lines = lines_per_img.get(img_id, [])
        (lbl_out / f"{Path(fn).stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_img += 1
    return n_img, n_box


def main():
    ap = argparse.ArgumentParser(description="Roboflow COCO -> YOLO 구조 변환")
    ap.add_argument("--src", required=True, help="COCO export 폴더(train/valid/test 포함)")
    ap.add_argument("--dst", required=True, help="YOLO 구조 출력 폴더")
    args = ap.parse_args()

    src, dst = Path(args.src).resolve(), Path(args.dst).resolve()

    # 카테고리 결정: train JSON 기준, '어노테이션이 실제로 달린' 카테고리만 채택.
    # Roboflow COCO 는 id 0 에 박스가 하나도 없는 더미 카테고리(supercategory 자리)를
    # 자동으로 끼워 넣는다. YOLO 는 클래스 번호가 0부터 빈틈없이 이어져야 하므로,
    # 더미를 버리고 나머지를 0..k-1 로 다시 번호 매긴다(번호만 당겨지고 부품은 그대로).
    # 예) 이 데이터: 0(더미)=삭제, 1 bearing->0, 2 bolt->1, 3 gear->2, 4 nut->3
    train_json = json.loads((src / "train" / "_annotations.coco.json").read_text(encoding="utf-8"))
    used_ids = {a["category_id"] for a in train_json["annotations"]}
    cats = [c for c in train_json["categories"] if c["id"] in used_ids]
    cats.sort(key=lambda c: c["id"])
    cat_map = {c["id"]: i for i, c in enumerate(cats)}          # 원본id -> 새id(0..)
    names = {i: c["name"].lower() for i, c in enumerate(cats)}  # 소문자 통일
    print(f"[카테고리] {names} (더미 제외: "
          f"{[c['name'] for c in train_json['categories'] if c['id'] not in used_ids]})")

    total_i = total_b = 0
    for split in ("train", "valid", "test"):
        d = src / split
        if not d.is_dir():
            continue
        ni, nb = convert_split(d, dst / split, cat_map)
        total_i += ni
        total_b += nb
        print(f"[{split}] 이미지 {ni}장 / 박스 {nb}건")

    (dst / "data.yaml").write_text(
        yaml.safe_dump({"names": names, "nc": len(names)}, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    print(f"\n완료: 총 {total_i}장 / {total_b}건 -> {dst}")
    print(f"다음: python scripts/experiments/experiment_autolearn.py --src {dst} --classes bolt nut")


if __name__ == "__main__":
    main()
