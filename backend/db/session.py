"""session.py - DB 접속(엔진·세션)과 경로 상대화 헬퍼.

접속 정보는 코드에 하드코딩하지 않는다(비밀번호 커밋 금지). 우선순위:
  1) 환경변수 XR_DB_URL 이 있으면 그대로 사용
  2) 없으면 deploy/.env + 환경변수의 XR_DB_{USER,PASSWORD,HOST,PORT,NAME} 로 조립
개발(Windows)은 deploy/.env, Thor(컨테이너)는 같은 키를 컨테이너 env 로 주면 코드 수정 없이 옮겨진다.
엔진을 SQLite 로 바꾸려면 XR_DB_URL=sqlite+pysqlite:///./xr.db 만 지정하면 된다.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# scripts/config.py 재사용(BASE_DIR 등 경로 단일 소스)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import config  # noqa: E402

ENV_FILE = config.BASE_DIR / "deploy" / ".env"


def _load_env_file(path: Path) -> dict[str, str]:
    """deploy/.env 를 읽어 dict 로. python-dotenv 의존성 없이 KEY=VALUE 만 처리한다."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.split("#")[0].strip()   # 값 뒤 주석 제거
    return out


def _build_url() -> str:
    if os.environ.get("XR_DB_URL"):
        return os.environ["XR_DB_URL"]
    env = {**_load_env_file(ENV_FILE), **os.environ}   # 환경변수가 .env 를 덮는다
    if env.get("XR_DB_URL"):
        return env["XR_DB_URL"]
    pw = env.get("XR_DB_PASSWORD")
    if not pw:
        raise RuntimeError(
            "DB 접속 정보가 없습니다. deploy/.env.example 을 deploy/.env 로 복사해 XR_DB_PASSWORD 를 채우거나, "
            "환경변수 XR_DB_URL 을 지정하세요."
        )
    user = env.get("XR_DB_USER", "xr")
    host = env.get("XR_DB_HOST", "localhost")
    port = env.get("XR_DB_PORT", "5432")
    name = env.get("XR_DB_NAME", "xr_autolearning")
    return f"postgresql+psycopg://{user}:{pw}@{host}:{port}/{name}"


DB_URL = _build_url()

# 접속 타임아웃이 없으면 DB 가 죽었을 때 요청이 수십 초~수 분 매달린다(파일 폴백이 무의미해짐).
# Postgres 는 connect_timeout 으로 빠르게 실패시키고, 풀 대기도 짧게 잡는다.
_CONNECT_TIMEOUT = int(os.environ.get("XR_DB_CONNECT_TIMEOUT", "3"))
_kw: dict = {"pool_pre_ping": True, "future": True}   # pre_ping: 컨테이너 재시작 후 죽은 커넥션 자동 정리
if DB_URL.startswith("postgresql"):
    _kw["connect_args"] = {"connect_timeout": _CONNECT_TIMEOUT}
    _kw["pool_timeout"] = _CONNECT_TIMEOUT
engine = create_engine(DB_URL, **_kw)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def get_session() -> Session:
    """스크립트용 세션. FastAPI 라우트에서는 with 문으로 감싸 쓴다."""
    return SessionLocal()


# ---------- 경로 규약 ----------
# DB 에는 레포 루트 기준 상대경로 + POSIX 구분자만 저장한다.
# 그래야 Windows(개발)에서 만든 행이 Thor(Linux 컨테이너)에서도 그대로 유효하다.

def rel(p: str | Path) -> str:
    """절대·상대 무엇이 와도 BASE_DIR 기준 POSIX 상대경로로 정규화."""
    path = Path(p)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(config.BASE_DIR).as_posix()
    except ValueError:
        # 레포 밖 경로(외부 스토리지 등)는 그대로 둔다 — 이 경우만 절대경로 허용
        return path.as_posix()


def abspath(stored: str | None) -> Path | None:
    """DB 에 저장된 상대경로를 현재 실행 환경의 절대경로로 되돌린다."""
    if not stored:
        return None
    p = Path(stored)
    return p if p.is_absolute() else (config.BASE_DIR / p)
