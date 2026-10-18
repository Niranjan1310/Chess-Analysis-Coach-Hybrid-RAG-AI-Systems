"""Pydantic schemas for game listing and Lichess import."""

from __future__ import annotations
from pydantic import BaseModel


class LichessImportRequest(BaseModel):
    """POST /api/games/lichess request."""
    username: str
    max_games: int = 5
    depth: int = 14


class LichessImportResponse(BaseModel):
    """POST /api/games/lichess response."""
    status: str
    player: str
    imported_games: int
    message: str


class GameListItem(BaseModel):
    """GET /api/games item."""
    id: str
    source: str
    player_analyzed: str | None
    white_player: str | None
    black_player: str | None
    result: str | None
    eco: str | None
    opening_name: str | None
    total_moves: int
    flagged_moves: int
    created_at: str
