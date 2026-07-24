"""dataset_utils.py - 데이터셋 등록/클래스 정의 공용 유틸.

여러 반입 스크립트(0_import_render, 0_import_roboflow)가 각자 data.yaml 을
고치면 클래스 번호가 어긋나는 사고가 생긴다. names 정규화와 data.yaml 등록을
한 구현으로 모아, 어느 반입 경로로 들어와도 같은 안전장치(기존 클래스 충돌 시
중단)를 거치게 한다.
"""
import yaml

import config


def write_yaml(path, cfg):
    """yaml 저장(한글 유지, 키 순서 유지)."""
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                    encoding="utf-8")


def normalize_names(raw):
    """data.yaml 의 names 를 {int id: 이름} 으로 정규화. 리스트/딕셔너리 둘 다 처리."""
    if isinstance(raw, list):
        return {i: n for i, n in enumerate(raw)}
    if isinstance(raw, dict):
        return {int(k): v for k, v in raw.items()}
    return None


def append_class(name):
    """클래스 1개를 data.yaml 에 추가하고 클래스 번호를 반환(이미 있으면 기존 번호).

    register_classes 와 달리 '추가'가 목적: 기존 번호는 절대 바꾸지 않고 끝에만
    붙이므로 기존 datasets/ 라벨과 호환된다. 부품 등록(labeling/register_part)처럼
    운영 중 클래스가 늘어나는 경로에서 쓴다.
    """
    cfg, names = {}, {}
    if config.DATA_YAML.exists():
        cfg = yaml.safe_load(config.DATA_YAML.read_text(encoding="utf-8")) or {}
        names = normalize_names(cfg.get("names")) or {}
    for k, v in names.items():
        if v.lower() == name.lower():
            return k
    new_id = max(names.keys(), default=-1) + 1
    names[new_id] = name
    cfg.setdefault("path", "./data/datasets")
    cfg.setdefault("train", "images/train")
    cfg.setdefault("val", "images/val")
    cfg["names"] = {int(k): names[k] for k in sorted(names)}
    write_yaml(config.DATA_YAML, cfg)
    print(f"[등록] 클래스 추가: {new_id} = {name}")
    return new_id


def register_classes(class_names):
    """클래스 정의({id: 이름})를 프로젝트 data.yaml 에 등록한다.

    기존에 다른 클래스가 등록돼 있으면 중단한다: 기존 datasets/ 라벨은 옛 번호로
    매겨져 있어서, names 만 갈아끼우면 라벨 번호가 통째로 꼬이기 때문이다.
    """
    cfg = {}
    if config.DATA_YAML.exists():
        cfg = yaml.safe_load(config.DATA_YAML.read_text(encoding="utf-8")) or {}
        old = normalize_names(cfg.get("names"))
        if old and [n.lower() for n in old.values()] != [n.lower() for n in class_names.values()]:
            raise SystemExit(
                f"[중단] data.yaml 에 이미 다른 클래스가 등록돼 있습니다.\n"
                f"      기존: {list(old.values())}\n"
                f"      반입: {list(class_names.values())}\n"
                f"      기존 데이터셋과 섞이면 라벨 번호가 꼬입니다. data.yaml/datasets 를 정리 후 재실행하세요.")
    cfg.setdefault("path", "./data/datasets")
    cfg.setdefault("train", "images/train")
    cfg.setdefault("val", "images/val")
    cfg["names"] = class_names
    write_yaml(config.DATA_YAML, cfg)
    print(f"[등록] data.yaml names <- {class_names}")
