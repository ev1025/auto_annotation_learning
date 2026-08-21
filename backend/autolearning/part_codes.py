# -*- coding: utf-8 -*-
"""부품 전역 코드표. 클라이언트(태블릿·ML2)가 믿고 쓸 수 있는 불변 번호를 관리한다.

왜 필요한가
  학습은 그때 고른 부품만 모아 이름순으로 인덱스를 매긴다(0,1,2...). 그래서 모델의
  클래스 번호(detection_code)는 재학습마다 뜻이 바뀐다.
    어제  {0: a_test}                  -> 0 = a_test
    오늘  {0: a_test, 1: medicine}     -> 0 = a_test, 1 = medicine
    gearbox 만 학습 {0: gearbox}       -> 0 = gearbox  (같은 0 이 다른 부품!)
  클라이언트가 번호로 부품을 판별하면 조용히 틀린다. 그래서 모델과 무관한 전역 번호를 둔다.

규칙
  - 코드는 1부터, **추가만** 한다(append-only). 부품을 지워도 그 번호는 재사용하지 않는다.
  - 원천 파일: data/bell412/parts/part_codes.json  {"부품명": 코드}
  - 파일이 없으면 부품 폴더를 훑어 새로 만든다. 있으면 그 값이 원천이고 절대 바꾸지 않는다.
  - 파일은 기기 간 공유 대상이다(로컬·토르가 같아야 클라이언트 매핑이 일치한다).

사용
  from part_codes import code_of, ensure, table
  ensure(["a_test", "medicine"])      # 없으면 코드 부여(등록 시 호출)
  code_of("medicine")                 # -> 36
  table()                             # -> {"medicine": 36, ...}

CLI
  python backend/autolearning/part_codes.py            # 표 출력
  python backend/autolearning/part_codes.py sync       # data/bell412/<부품> 폴더를 훑어 코드 부여
"""
import json
import sys
import threading
from pathlib import Path

try:
    import config
except ImportError:   # 단독 실행(CLI)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import config

PARTS_DIR = config.DATA_DIR / "bell412"
CODES_FILE = PARTS_DIR / "parts" / "part_codes.json"
_LOCK = threading.Lock()


def _seed() -> dict[str, int]:
    """코드표가 없을 때: 등록된 부품 폴더를 이름순으로 훑어 1..N 을 부여한다.
    (초기 씨앗은 예전 classes.txt 순서였고, 그 순서는 이미 part_codes.json 에 반영돼 있다)"""
    names = sorted(p.parent.name for p in PARTS_DIR.glob("*/videos") if p.is_dir())
    return {n: i + 1 for i, n in enumerate(names)}


def _from_db() -> dict[str, int] | None:
    """DB(원천)에서 {부품명: code}. DB 가 없거나 열이 없으면 None(파일로 폴백)."""
    try:
        import sys as _sys
        from pathlib import Path as _P
        _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))     # backend
        from db.models import Part                                     # noqa: PLC0415
        from db.session import get_session                             # noqa: PLC0415
        s = get_session()
        try:
            rows = s.query(Part).filter(Part.code.isnot(None)).all()
            return {p.name: int(p.code) for p in rows} or None
        finally:
            s.close()
    except Exception:   # noqa: BLE001 - DB 없이도 동작해야 한다(추론 컨테이너 등)
        return None


def export_from_db() -> int:
    """DB -> part_codes.json 사본 갱신. 부품 등록·번호 변경 후 호출한다."""
    t = _from_db()
    if not t:
        return 0
    with _LOCK:
        _write(t)
    return len(t)


def table() -> dict[str, int]:
    """{부품명: 코드}. DB 가 원천이고, 못 읽으면 json 사본 -> 폴더 스캔 순으로 폴백."""
    db = _from_db()
    if db:
        return db
    with _LOCK:
        if CODES_FILE.exists():
            try:
                return {str(k): int(v) for k, v in json.loads(CODES_FILE.read_text(encoding="utf-8")).items()}
            except Exception:   # noqa: BLE001 - 깨진 파일이 서비스를 막지 않게
                pass
        t = _seed()
        _write(t)
        return t


def _write(t: dict[str, int]) -> None:
    CODES_FILE.parent.mkdir(parents=True, exist_ok=True)
    CODES_FILE.write_text(json.dumps(t, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def ensure(names) -> dict[str, int]:
    """목록 중 코드가 없는 부품에 다음 번호를 부여하고 전체 표를 돌려준다."""
    t = table()
    new = [n for n in dict.fromkeys(names) if n and n not in t]
    if new:
        nxt = max(t.values(), default=0) + 1
        for n in sorted(new):
            t[n] = nxt
            nxt += 1
        with _LOCK:
            _write(t)
    return t


def code_of(name) -> int | None:
    """부품 코드. 없으면 None(응답에서 null 로 나가고, 이름으로 판별하면 된다)."""
    return table().get(name)


def codes_for_names(model_names: dict) -> dict[int, int | None]:
    """모델 클래스 인덱스 -> 전역 코드(없는 부품은 코드 부여). 학습·적용 쪽에서 쓴다."""
    t = ensure(list(model_names.values()))
    return {int(i): t.get(n) for i, n in model_names.items()}


def map_names(model_names: dict) -> dict[int, int | None]:
    """읽기 전용 매핑. 추론서버처럼 코드표를 만들 권한이 없는 쪽에서 쓴다.

    추론 컨테이너에는 data 볼륨이 없어 코드표 파일이 안 보인다. 그때 ensure() 를 부르면
    빈 표에서 1,2... 로 새로 매겨 실제 코드와 어긋난다(실제로 그런 사고가 났다).
    그래서 여기서는 파일이 없으면 조용히 None 을 준다."""
    if not CODES_FILE.exists():
        return {int(i): None for i in model_names}
    t = table()
    return {int(i): t.get(n) for i, n in model_names.items()}


def _sync_from_folders() -> dict[str, int]:
    """data/bell412/<부품>/videos 가 있는 폴더명을 부품으로 보고 코드를 부여한다."""
    names = sorted(p.parent.name for p in PARTS_DIR.glob("*/videos") if p.is_dir())
    return ensure(names)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sync":
        t = _sync_from_folders()
        print(f"동기화 완료 · 부품 {len(t)}종 -> {CODES_FILE}")
    else:
        t = table()
        print(f"{CODES_FILE}  (부품 {len(t)}종)")
    for n, c in sorted(t.items(), key=lambda kv: kv[1]):
        print(f"  {c:>3}  {n}")
