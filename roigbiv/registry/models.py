"""SQLAlchemy ORM models for the registry.

Types are chosen to render identically on SQLite and Postgres:
 - UUIDs are stored as CHAR(36) strings (portable across both).
 - JSON columns use sqlalchemy.JSON → TEXT on SQLite, JSONB on Postgres.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    CHAR,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class FOV(Base):
    __tablename__ = "fov"

    fov_id: Mapped[str] = mapped_column(CHAR(36), primary_key=True)
    fingerprint_hash: Mapped[str] = mapped_column(CHAR(64), nullable=False)
    animal_id: Mapped[str] = mapped_column(String(128), nullable=False)
    region: Mapped[str] = mapped_column(String(64), nullable=False)
    mean_m_uri: Mapped[str] = mapped_column(String(512), nullable=False)
    centroid_table_uri: Mapped[str] = mapped_column(String(512), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    latest_session_date: Mapped[datetime] = mapped_column(Date, nullable=True)
    fingerprint_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    fov_embedding_uri: Mapped[str] = mapped_column(String(512), nullable=True)
    roi_embeddings_uri: Mapped[str] = mapped_column(String(512), nullable=True)

    cells = relationship("Cell", back_populates="fov", cascade="all, delete-orphan")
    sessions = relationship("Session", back_populates="fov", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("fingerprint_hash", name="uq_fov_fingerprint_hash"),
        Index("ix_fov_animal_region", "animal_id", "region"),
    )


class Cell(Base):
    __tablename__ = "cell"

    global_cell_id: Mapped[str] = mapped_column(CHAR(36), primary_key=True)
    fov_id: Mapped[str] = mapped_column(
        CHAR(36), ForeignKey("fov.fov_id", ondelete="CASCADE"), nullable=False,
    )
    first_seen_session_id: Mapped[str] = mapped_column(CHAR(36), nullable=True)
    morphology_summary: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    fov = relationship("FOV", back_populates="cells")

    __table_args__ = (
        Index("ix_cell_fov_id", "fov_id"),
    )


class Session(Base):
    __tablename__ = "session"

    session_id: Mapped[str] = mapped_column(CHAR(36), primary_key=True)
    fov_id: Mapped[str] = mapped_column(
        CHAR(36), ForeignKey("fov.fov_id", ondelete="CASCADE"), nullable=False,
    )
    session_date: Mapped[datetime] = mapped_column(Date, nullable=False)
    output_dir: Mapped[str] = mapped_column(String(1024), nullable=False)
    fov_sim: Mapped[float] = mapped_column(Float, nullable=True)
    fov_posterior: Mapped[float] = mapped_column(Float, nullable=True)
    n_matched: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_new: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_missing: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    # v3: blob URI for the per-session int32 ROICaT cluster label array.
    cluster_labels_uri: Mapped[str] = mapped_column(String(512), nullable=True)

    fov = relationship("FOV", back_populates="sessions")

    __table_args__ = (
        Index("ix_session_fov_date", "fov_id", "session_date"),
    )


class CellObservation(Base):
    __tablename__ = "cell_observation"

    observation_id: Mapped[str] = mapped_column(CHAR(36), primary_key=True)
    global_cell_id: Mapped[str] = mapped_column(
        CHAR(36), ForeignKey("cell.global_cell_id", ondelete="CASCADE"), nullable=False,
    )
    session_id: Mapped[str] = mapped_column(
        CHAR(36), ForeignKey("session.session_id", ondelete="CASCADE"), nullable=False,
    )
    local_label_id: Mapped[int] = mapped_column(Integer, nullable=False)
    match_score: Mapped[float] = mapped_column(Float, nullable=True)
    # v3: ROICaT cluster label for this observation. Mutable (may change when
    # the FOV is re-clustered); canonical per-ROI identity remains global_cell_id.
    cluster_label: Mapped[int] = mapped_column(Integer, nullable=True)

    __table_args__ = (
        UniqueConstraint("session_id", "local_label_id", name="uq_obs_session_label"),
        Index("ix_obs_cell", "global_cell_id"),
    )
