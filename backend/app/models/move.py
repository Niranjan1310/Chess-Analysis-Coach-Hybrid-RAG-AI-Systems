"""MoveRecord SQLAlchemy model."""

from __future__ import annotations

from sqlalchemy import Column, String, Integer, Text, ForeignKey
from sqlalchemy.orm import relationship

from app.models.database import Base


class MoveRecord(Base):
    """Detailed record of a single move played in a game."""

    __tablename__ = "move_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(String, ForeignKey("games.id", ondelete="CASCADE"), nullable=False, index=True)
    player_name = Column(String, nullable=True, index=True)
    move_number = Column(Integer, nullable=False)
    side = Column(String, nullable=False)  # "white" or "black"
    move_san = Column(String, nullable=False)
    move_uci = Column(String, nullable=False)
    fen_before = Column(String, nullable=False)
    fen_after = Column(String, nullable=False)
    phase = Column(String, nullable=False, index=True)  # "opening", "middlegame", "endgame"
    eco = Column(String, nullable=True, index=True)
    opening_name = Column(String, nullable=True)
    piece_moved = Column(String, nullable=False, index=True)  # "P", "N", "B", "R", "Q", "K"
    centipawn_loss = Column(Integer, default=0)
    move_class = Column(String, nullable=False, index=True)  # "good", "inaccuracy", "mistake", "blunder"
    best_move = Column(String, nullable=True)
    eval_before = Column(String, nullable=True)
    eval_after = Column(String, nullable=True)
    explanation = Column(Text, nullable=True)
    game_result_context = Column(String, nullable=True, index=True)  # "winning", "losing", "equal"

    game = relationship("Game", back_populates="moves")
