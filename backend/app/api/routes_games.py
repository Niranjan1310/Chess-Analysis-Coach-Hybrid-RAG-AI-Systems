"""API routes for games listing and Lichess import."""

from __future__ import annotations

import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.models.database import get_db
from app.models.game import Game
from app.schemas.games import LichessImportRequest, LichessImportResponse, GameListItem
from app.services.lichess_client import LichessClient
from app.services.analysis_service import analyze_game, save_analyzed_game

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


@router.get("/games", response_model=List[GameListItem])
def list_games(
    player: str | None = Query(None, description="Filter games by player"),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> List[GameListItem]:
    """List analyzed games stored in the database."""
    query = db.query(Game)
    if player:
        query = query.filter(
            (Game.player_analyzed == player) |
            (Game.white_player == player) |
            (Game.black_player == player)
        )
    games = query.order_by(Game.created_at.desc()).limit(limit).all()

    return [
        GameListItem(
            id=g.id,
            source=g.source,
            player_analyzed=g.player_analyzed,
            white_player=g.white_player,
            black_player=g.black_player,
            result=g.result,
            eco=g.eco,
            opening_name=g.opening_name,
            total_moves=g.total_moves,
            flagged_moves=g.flagged_moves,
            created_at=g.created_at.isoformat() if g.created_at else "",
        )
        for g in games
    ]


@router.post("/games/lichess", response_model=LichessImportResponse)
def import_lichess_games(
    request: LichessImportRequest,
    db: Session = Depends(get_db),
) -> LichessImportResponse:
    """Import games from Lichess, analyze them with Stockfish, and save to DB."""
    client = LichessClient()
    try:
        pgns = client.fetch_user_games(request.username, max_games=request.max_games)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    imported_count = 0
    for pgn in pgns:
        try:
            result = analyze_game(pgn, depth=request.depth)
            save_analyzed_game(
                db=db,
                pgn=pgn,
                result=result,
                player_analyzed=request.username,
                source="lichess",
            )
            imported_count += 1
        except Exception as e:
            logger.warning("Failed to analyze imported game: %s", e)

    return LichessImportResponse(
        status="success",
        player=request.username,
        imported_games=imported_count,
        message=f"Successfully imported and analyzed {imported_count} games for {request.username}.",
    )
