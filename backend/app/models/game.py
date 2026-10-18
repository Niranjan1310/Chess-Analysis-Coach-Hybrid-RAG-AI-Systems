"""Game SQLAlchemy model."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, DateTime, Text
from sqlalchemy.orm import relationship

from app.models.database import Base


def _utc_now():
    return datetime.now(timezone.utc)


class Game(Base):
    """Analyzed chess game record."""

    __tablename__ = "games"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    source = Column(String, default="upload")  # "upload", "lichess"
    player_analyzed = Column(String, nullable=True, index=True)
    white_player = Column(String, nullable=True)
    black_player = Column(String, nullable=True)
    result = Column(String, nullable=True)
    eco = Column(String, nullable=True)
    opening_name = Column(String, nullable=True)
    total_moves = Column(Integer, default=0)
    flagged_moves = Column(Integer, default=0)
    pgn = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utc_now)

    moves = relationship("MoveRecord", back_populates="game", cascade="all, delete-orphan")
