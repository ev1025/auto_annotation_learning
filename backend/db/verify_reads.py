"""verify_reads.py - 파일판 vs DB판 조회 결과 동등성 검증.

'DB 로 갈아타도 화면이 똑같은가'를 눈으로 보지 않고 기계적으로 증명한다.
차이가 있으면 어느 필드가 어떻게 다른지 출력한다(무시해도 되는 차이는 화이트리스트로 명시).

사용:
    python backend/db/verify_reads.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "scripts"))       # config
sys.path.insert(0, str(HERE.parents[1] / "autolearning"))  # 파일판(autolabel·sam2_autolabel)
sys.path.insert(0, str(HERE.parents[1]))                   # db 패키지(backend/db)

import autolabel  # noqa: E402  (파일판)
import sam2_autolabel as sa  # noqa: E402
from db import reads  # noqa: E402  (DB판)


def norm(x):
    """비교용 정규화: dict 키 정렬, 경로 구분자 통일, 튜플->리스트."""
    if isinstance(x, dict):
        return {k: norm(v) for k, v in sorted(x.items())}
    if isinstance(x, (list, tuple)):
        return [norm(v) for v in x]
    if isinstance(x, float):
        return round(x, 6)
    if isinstance(x, str):
        return x.replace("\\", "/")
    return x


def diff(a, b, path="") -> list[str]:
    """중첩 구조의 차이를 경로와 함께 나열."""
    out = []
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return [f"{path}: 타입 {type(a).__name__} != {type(b).__name__} ({a!r} / {b!r})"]
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}.{k}: 파일판에만 없음 (DB={b[k]!r})")
            elif k not in b:
                out.append(f"{path}.{k}: DB판에만 없음 (파일={a[k]!r})")
            else:
                out += diff(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}: 길이 {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, f"{path}[{i}]")
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")
    return out


def check(name: str, file_fn, db_fn, ignore: set[str] | None = None) -> bool:
    try:
        fv, dv = norm(file_fn()), norm(db_fn())
    except Exception as e:  # noqa: BLE001
        print(f"[에러] {name}: {type(e).__name__}: {e}")
        return False
    ds = [d for d in diff(fv, dv, name) if not any(ig in d for ig in (ignore or set()))]
    if ds:
        print(f"[불일치] {name} ({len(ds)}건)")
        for d in ds[:12]:
            print("   ", d)
        if len(ds) > 12:
            print(f"    ... 외 {len(ds)-12}건")
        return False
    n = len(fv) if isinstance(fv, (list, dict)) else 1
    print(f"[일치] {name} (원소 {n})")
    return True


def main() -> int:
    print("파일판 vs DB판 조회 동등성 검증\n")
    ok = True
    # count/ready: 파일판은 캐시 없으면 영상 길이로 '추정'하지만 DB는 실제 추출 수를 쓴다.
    # 프레임이 잘려 있는 부품은 두 값이 같고, 아직 안 자른 부품은 추정치라 다를 수 있다 -> 별도 보고.
    ok &= check("list_folders", autolabel.list_folders, reads.list_folders,
                ignore={".count", ".ready"})
    ok &= check("load_shots", sa.load_shots, reads.load_shots)
    ok &= check("labeled_parts", sa.labeled_parts, reads.labeled_parts)
    ok &= check("served_model", lambda: sa.served_model() or {"none": True},
                lambda: reads.served_model() or {"none": True})
    ok &= check("list_models", sa.list_models, reads.list_models)

    # count/ready 는 따로 집계해 '추정치 때문에 다른 것'인지 확인
    f = {v["key"]: v for g in autolabel.list_folders() for v in g["videos"]}
    d = {v["key"]: v for g in reads.list_folders() for v in g["videos"]}
    both = sorted(set(f) & set(d))
    same = [k for k in both if f[k]["count"] == d[k]["count"]]
    ready_f = [k for k in both if f[k]["ready"]]
    print(f"\ncount 비교: 동일 {len(same)}/{len(both)} (파일판 ready=True 인 것 {len(ready_f)}개)")
    for k in both:
        if f[k]["count"] != d[k]["count"]:
            tag = "추출됨" if f[k]["ready"] else "미추출(파일판은 영상길이 추정)"
            print(f"   {k}: 파일 {f[k]['count']} / DB {d[k]['count']}  [{tag}]")

    print("\n행 수:", json.dumps(reads.counts(), ensure_ascii=False))
    print("\n결과:", "전부 일치" if ok else "불일치 있음(위 참조)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
