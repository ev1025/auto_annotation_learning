"""models.py - 오토러닝 메타데이터 스키마(SQLAlchemy 2.x).

설계 원칙
- 큰 파일(이미지·가중치·ONNX)은 DB 에 넣지 않는다. 파일시스템에 두고 DB 는 '경로'만 가진다.
- 경로는 전부 **레포 루트(config.BASE_DIR) 기준 상대경로 + POSIX 구분자**로 저장한다.
  meta.json 처럼 'C:\\Users\\...' 절대경로를 넣으면 Thor(Linux/컨테이너)에서 전부 깨진다.
- 좌표는 전부 **정규화(0~1)**. 원본 해상도가 바뀌어도 유효하다(현행 shots.json 과 동일 규약).

관계
    categories 1---N parts 1---N part_videos 1---N part_frames 1---1 sam2_annotations
    train_runs (독립: 학습 이력·현재 서비스 모델)
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func, text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


def _now() -> datetime:
    return datetime.now(timezone.utc)


# JSONB(Postgres)를 쓰되, SQLite 로 바꿔도 뜨도록 일반 JSON 으로 폴백
JSONType = JSONB().with_variant(JSON(), "sqlite")


class Base(DeclarativeBase):
    pass


class Category(Base):
    """부품 카테고리. UI 에서 추가·편집·삭제하므로 문자열이 아니라 테이블로 둔다.
    (varchar 로 두면 이름 편집 시 모든 부품 행을 일괄 수정해야 함)"""
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, server_default=func.now())

    parts: Mapped[list["Part"]] = relationship(back_populates="category")

    def __repr__(self) -> str:
        return f"<Category {self.name}>"


class Part(Base):
    """등록 부품. 폴더(data/bell412/<부품>)와 1:1."""
    __tablename__ = "parts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)   # YOLO 클래스명과 동일
    # 클라이언트(태블릿·ML2)가 부품을 판별하는 번호. id 와 다르다.
    #   id   = 자동증가. 행을 지워도 번호가 되돌아오지 않아 기기마다 값이 갈린다
    #          (실측: medicine 이 로컬 45, 토르 36. 로컬에서 임시부품 9개를 만들고 지운 탓)
    #   code = 우리가 정해서 넣는 번호. 등록 시 부여하고 절대 바꾸지 않으며 기기 간 동일하게 맞춘다.
    #          모델의 클래스 인덱스(detection_code)와도 무관하다(그건 학습마다 바뀜).
    code: Mapped[int | None] = mapped_column(Integer, unique=True, index=True)
    # 카테고리 삭제 시 부품은 남기고 분류만 비운다(SET NULL)
    category_id: Mapped[int | None] = mapped_column(ForeignKey("categories.id", ondelete="SET NULL"))
    description: Mapped[str | None] = mapped_column(Text)
    folder: Mapped[str] = mapped_column(String(512), nullable=False)              # 상대경로 data/bell412/<부품>
    model_3d_path: Mapped[str | None] = mapped_column(String(512))                # .glb/.obj/.stl/.ply
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now,
                                                 server_default=func.now())

    category: Mapped[Category | None] = relationship(back_populates="parts")
    videos: Mapped[list["PartVideo"]] = relationship(
        back_populates="part", cascade="all, delete-orphan", passive_deletes=True)

    def __repr__(self) -> str:
        return f"<Part {self.name}>"


class PartVideo(Base):
    """부품 촬영 테이크. 부품 1개에 영상 N개라서 별도 테이블이 필요하다.

    role 은 항상 'train' 이다. 파일명으로 학습/평가를 가르던 규칙은 제거했고(2026-08-14),
    평가용 영상은 제품 밖(data/_eval/<부품>)에 따로 둔다. 컬럼은 과거 행 호환으로 남겨둔다."""
    __tablename__ = "part_videos"
    __table_args__ = (UniqueConstraint("part_id", "stem", name="uq_part_video_stem"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    part_id: Mapped[int] = mapped_column(ForeignKey("parts.id", ondelete="CASCADE"), nullable=False, index=True)
    stem: Mapped[str] = mapped_column(String(255), nullable=False)        # 확장자 없는 파일명(train, test, take1 ...)
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="train")   # 'train' | 'test'
    path: Mapped[str] = mapped_column(String(512), nullable=False)        # 상대경로 .../videos/<stem>.mp4
    width: Mapped[int | None] = mapped_column(Integer)                   # 원본 해상도(정규화 좌표 역변환용)
    height: Mapped[int | None] = mapped_column(Integer)
    fps: Mapped[float | None] = mapped_column(Float)
    n_frames: Mapped[int | None] = mapped_column(Integer)                # 추출(서브샘플)된 프레임 수
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, server_default=func.now())

    part: Mapped[Part] = relationship(back_populates="videos")
    frames: Mapped[list["PartFrame"]] = relationship(
        back_populates="video", cascade="all, delete-orphan", passive_deletes=True)

    def __repr__(self) -> str:
        return f"<PartVideo {self.stem} ({self.role})>"


class PartFrame(Base):
    """등록 시 미리 잘라둔 프레임 1장. 학습 화면에서 실시간 추출하지 않고 이 경로만 읽는다."""
    __tablename__ = "part_frames"
    __table_args__ = (UniqueConstraint("video_id", "frame_number", name="uq_frame_no"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("part_videos.id", ondelete="CASCADE"), nullable=False, index=True)
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False)   # 서브샘플 후 순번(0-based)
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)  # results/autolabels/<부품>/images/<stem>/00000.jpg
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, server_default=func.now())

    video: Mapped[PartVideo] = relationship(back_populates="frames")
    annotation: Mapped["Sam2Annotation | None"] = relationship(
        back_populates="frame", cascade="all, delete-orphan", passive_deletes=True, uselist=False)

    def __repr__(self) -> str:
        return f"<PartFrame {self.frame_number}>"


class Sam2Annotation(Base):
    """프레임 1장의 SAM2 산출물.

    두 종류가 한 테이블에 들어간다(둘 다 프레임 단위라 1:1).
      - 참조샷(is_reference=True): 사용자가 직접 찍은 탭 포인트가 있는 프레임(영상당 ~10장)
      - 전파 결과: 탭 없이 SAM2 가 만든 박스만 있는 프레임(영상당 200장+)
    """
    __tablename__ = "sam2_annotations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    frame_id: Mapped[int] = mapped_column(ForeignKey("part_frames.id", ondelete="CASCADE"),
                                          nullable=False, unique=True)
    # 사용자가 찍은 점: [{"rx":0.239,"ry":0.318,"label":1}] (label 1=포함, 0=제외). 정규화 좌표.
    tap_points: Mapped[list | None] = mapped_column(JSONType)
    # YOLO 포맷 박스(정규화): [{"cls":0,"cx":0.367,"cy":0.406,"w":0.295,"h":0.233}]
    boxes: Mapped[list | None] = mapped_column(JSONType)
    is_reference: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    label_path: Mapped[str | None] = mapped_column(String(512))        # labels/<stem>_00000.txt (학습이 실제로 먹는 파일)
    box_preview_path: Mapped[str | None] = mapped_column(String(512))  # boxs/<stem>_00000.jpg (육안 확인용)
    mask_path: Mapped[str | None] = mapped_column(String(512))         # 마스크 이미지(있으면)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now,
                                                 server_default=func.now())

    frame: Mapped[PartFrame] = relationship(back_populates="annotation")

    def __repr__(self) -> str:
        return f"<Sam2Annotation frame={self.frame_id} ref={self.is_reference}>"


class TrainRun(Base):
    """학습 1회 = 모델 1개. 기존 results/<model_id>/meta.json + results/_served.json 을 대체한다.
    모델관리·롤백·현재 서비스 모델 판정이 전부 이 테이블에서 나온다."""
    __tablename__ = "train_runs"
    __table_args__ = (
        # 현재 서비스 모델은 항상 최대 1개. is_active=true 인 행에만 걸리는 부분 유니크 인덱스
        # (false 는 여러 개 허용해야 하므로 일반 unique 로는 안 된다)
        Index("uq_train_run_active", "is_active", unique=True,
              postgresql_where=text("is_active"), sqlite_where=text("is_active")),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)   # 260813_105307
    session: Mapped[str | None] = mapped_column(String(64))
    label: Mapped[str | None] = mapped_column(String(255))            # "2026-08-13 10:54 · 1종 · a_test"
    trained_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    classes: Mapped[list | None] = mapped_column(JSONType)            # ["a_test", ...]
    n_classes: Mapped[int | None] = mapped_column(Integer)
    n_images: Mapped[int | None] = mapped_column(Integer)
    per_class: Mapped[dict | None] = mapped_column(JSONType)          # {"a_test": 266}
    learn_rate: Mapped[float | None] = mapped_column(Float)           # 산입률(%)
    epochs: Mapped[int | None] = mapped_column(Integer)
    map50: Mapped[float | None] = mapped_column(Float)                # 공인 평가지표(있으면)
    weights_path: Mapped[str | None] = mapped_column(String(512))     # 상대경로 results/<id>/model/best.pt
    onnx_path: Mapped[str | None] = mapped_column(String(512))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)   # 현재 서비스 모델
    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    meta: Mapped[dict | None] = mapped_column(JSONType)               # 원본 meta.json 전체(손실 없이 보관)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, server_default=func.now())

    def __repr__(self) -> str:
        return f"<TrainRun {self.model_id}{' *active' if self.is_active else ''}>"
