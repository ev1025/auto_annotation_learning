"""오토러닝 메타데이터 DB 레이어.

이미지·가중치 같은 큰 파일은 파일시스템에 그대로 두고, DB 는 메타데이터와 관계만 관리한다.
  models.py   스키마(부품·영상·프레임·SAM2·학습이력)
  session.py  접속(XR_DB_URL)과 경로 상대화 헬퍼
  migrate_from_files.py  기존 파일 구조(shots.json·meta.json) -> DB 이관(멱등)
"""
from .models import (  # noqa: F401
    Base, Category, Part, PartVideo, PartFrame, Sam2Annotation, TrainRun,
)
from .session import DB_URL, SessionLocal, abspath, engine, get_session, rel  # noqa: F401
