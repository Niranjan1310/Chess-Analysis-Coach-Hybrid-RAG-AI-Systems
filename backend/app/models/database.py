"""Database session and engine setup."""

from __future__ import annotations

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from app.config import settings

Base = declarative_base()

_default_engine = None
_default_session_factory = None


def get_engine(db_url: str | None = None):
    """Create or return existing SQLAlchemy engine."""
    url = db_url or getattr(settings, "database_url", "sqlite:///./chessmind.db")
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args)


def get_session_factory(engine=None):
    """Return a sessionmaker bound to engine."""
    eng = engine or get_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=eng)


def init_db(engine=None) -> None:
    """Create all tables in the database."""
    eng = engine or get_engine()
    Base.metadata.create_all(bind=eng)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a db session."""
    session_factory = get_session_factory()
    db = session_factory()
    try:
        yield db
    finally:
        db.close()
