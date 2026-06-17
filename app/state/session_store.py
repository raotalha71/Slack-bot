"""
SQLite session store for persisting pipeline state across sessions.

Ensures the system remembers transcripts and drafts even if the user
returns the next day (assessment requirement).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    Column,
    DateTime,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import Session as DBSession
from sqlalchemy.orm import declarative_base, sessionmaker

from config import get_settings

Base = declarative_base()


# ---------------------------------------------------------------------------
# ORM Model
# ---------------------------------------------------------------------------

class SessionModel(Base):
    """Persistent session storing pipeline state for a Slack thread."""

    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    thread_ts = Column(String(64), unique=True, index=True, nullable=False)
    channel_id = Column(String(64), nullable=False)
    user_id = Column(String(64), nullable=False)
    state_json = Column(Text, nullable=False, default="{}")
    current_draft = Column(Text, nullable=True)
    transcript_text = Column(Text, nullable=True)
    status = Column(
        String(32),
        nullable=False,
        default="intake",
    )  # intake | researching | writing | complete | revising | error
    created_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    def get_state(self) -> dict[str, Any]:
        """Deserialize the stored JSON state."""
        return json.loads(self.state_json) if self.state_json else {}

    def set_state(self, state: dict[str, Any]) -> None:
        """Serialize state to JSON for storage."""
        # Filter out non-serializable fields (e.g. bytes)
        clean = {
            k: v for k, v in state.items()
            if not isinstance(v, bytes)
        }
        self.state_json = json.dumps(clean, default=str)


# ---------------------------------------------------------------------------
# Database Engine & Session Factory
# ---------------------------------------------------------------------------

_engine = None
_SessionLocal = None


def _get_engine():
    """Lazy-initialize the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(
            settings.DATABASE_URL,
            connect_args={"check_same_thread": False},  # SQLite-specific
            echo=False,
        )
    return _engine


def _get_session_factory():
    """Lazy-initialize the session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=_get_engine(),
        )
    return _SessionLocal


def get_db() -> DBSession:
    """Get a database session. Caller must close it."""
    factory = _get_session_factory()
    return factory()


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create all tables on startup. No Alembic needed at assessment scale."""
    engine = _get_engine()
    Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------------
# CRUD Operations
# ---------------------------------------------------------------------------

def create_session(
    thread_ts: str,
    channel_id: str,
    user_id: str,
    state: dict[str, Any],
    transcript_text: str = "",
) -> SessionModel:
    """Create a new pipeline session for a Slack thread."""
    db = get_db()
    try:
        session = SessionModel(
            thread_ts=thread_ts,
            channel_id=channel_id,
            user_id=user_id,
            transcript_text=transcript_text,
        )
        session.set_state(state)
        db.add(session)
        db.commit()
        db.refresh(session)
        return session
    finally:
        db.close()


def get_session(thread_ts: str) -> Optional[SessionModel]:
    """Retrieve a session by Slack thread timestamp."""
    db = get_db()
    try:
        return (
            db.query(SessionModel)
            .filter(SessionModel.thread_ts == thread_ts)
            .first()
        )
    finally:
        db.close()


def get_sessions_by_user(user_id: str) -> list[SessionModel]:
    """Get all sessions for a user (for 'next day' recall)."""
    db = get_db()
    try:
        return (
            db.query(SessionModel)
            .filter(SessionModel.user_id == user_id)
            .order_by(SessionModel.updated_at.desc())
            .all()
        )
    finally:
        db.close()


def update_session(
    thread_ts: str,
    state: Optional[dict[str, Any]] = None,
    status: Optional[str] = None,
    draft: Optional[str] = None,
) -> Optional[SessionModel]:
    """Update an existing session's state, status, and/or draft."""
    db = get_db()
    try:
        session = (
            db.query(SessionModel)
            .filter(SessionModel.thread_ts == thread_ts)
            .first()
        )
        if session is None:
            return None

        if state is not None:
            session.set_state(state)
        if status is not None:
            session.status = status
        if draft is not None:
            session.current_draft = draft

        session.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(session)
        return session
    finally:
        db.close()


def session_exists(thread_ts: str) -> bool:
    """Check if a session exists for a given thread."""
    db = get_db()
    try:
        return (
            db.query(SessionModel)
            .filter(SessionModel.thread_ts == thread_ts)
            .count()
            > 0
        )
    finally:
        db.close()
