# -*- coding: utf-8 -*-
"""parts.code 열 추가 + 기존 번호 이관(멱등, 여러 번 실행해도 안전).

왜 필요한가
  부품 번호를 자동증가 id 로 쓰면 기기마다 값이 갈린다. 행을 지워도 번호가 되돌아오지 않기 때문이다.
  실측: medicine 이 로컬 45, 토르 36 (로컬에서 임시부품 9개를 만들고 지운 탓).
  그래서 '우리가 정해서 넣는 번호' 열(code)을 따로 둔다. 클라이언트는 이 값으로 부품을 판별한다.

무엇을 하나
  1. parts.code 열과 유니크 인덱스를 만든다(없을 때만)
  2. 이미 클라이언트에 나간 번호를 유지하기 위해 data/bell412/parts/part_codes.json 값으로 채운다
  3. json 에 없는 부품은 이름순으로 다음 번호를 이어서 부여한다
  4. 결과를 다시 part_codes.json 으로 내보낸다(추론서버가 DB 없이도 읽을 수 있게)

사용
  python backend/db/add_part_code.py            # 마이그레이션 + 이관
  python backend/db/add_part_code.py --show     # 현재 상태만 출력
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # backend (db 패키지)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from sqlalchemy import text                                          # noqa: E402

import config                                                        # noqa: E402
from db.models import Part                                           # noqa: E402
from db.session import get_session                                   # noqa: E402

CODES_JSON = config.DATA_DIR / "bell412" / "parts" / "part_codes.json"


def add_column(s):
    """열·인덱스 추가(이미 있으면 통과)."""
    s.execute(text("ALTER TABLE parts ADD COLUMN IF NOT EXISTS code INTEGER"))
    s.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_parts_code ON parts (code)"))
    s.commit()


def load_json_codes() -> dict[str, int]:
    if not CODES_JSON.exists():
        return {}
    try:
        return {str(k): int(v) for k, v in json.loads(CODES_JSON.read_text(encoding="utf-8")).items()}
    except Exception as e:   # noqa: BLE001
        print(f"  경고: part_codes.json 을 못 읽었다({type(e).__name__}) -> 이름순으로 새로 부여한다")
        return {}


def export_json(s) -> int:
    """DB(원천) -> part_codes.json(사본). 추론 컨테이너가 DB 없이도 매핑을 읽게 한다."""
    rows = s.query(Part).filter(Part.code.isnot(None)).all()
    table = {p.name: p.code for p in rows}
    CODES_JSON.parent.mkdir(parents=True, exist_ok=True)
    CODES_JSON.write_text(json.dumps(table, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return len(table)


def main(show_only=False):
    s = get_session()
    try:
        if not show_only:
            add_column(s)
            j = load_json_codes()
            parts = s.query(Part).order_by(Part.name).all()
            used = {p.code for p in parts if p.code is not None}
            filled = kept = 0
            for p in parts:
                if p.code is not None:
                    kept += 1
                    continue
                want = j.get(p.name)
                if want is None or want in used:                      # json 에 없거나 충돌 -> 다음 번호
                    want = (max(used) + 1) if used else 1
                p.code = want
                used.add(want)
                filled += 1
            s.commit()
            n = export_json(s)
            print(f"  이관 완료: 유지 {kept}종 · 신규 부여 {filled}종 · json 내보냄 {n}종")

        rows = s.query(Part).order_by(Part.code).all()
        print(f"  {'code':>5}  {'id':>4}  이름")
        for p in rows:
            print(f"  {p.code if p.code is not None else '-':>5}  {p.id:>4}  {p.name}")
    finally:
        s.close()


if __name__ == "__main__":
    main(show_only="--show" in sys.argv)
