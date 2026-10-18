"""API routes for weakness profile."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.models.database import get_db
from app.services.weakness_service import compute_weakness_summary, WeaknessReport

router = APIRouter(prefix="/api")


@router.get("/weaknesses", response_model=WeaknessReport)
def get_weaknesses(
    player: str = Query(..., description="Player username"),
    min_games: int | None = Query(None, description="Override minimum games required"),
    db: Session = Depends(get_db),
) -> WeaknessReport:
    """GET /api/weaknesses?player=<name> — compute weakness summary."""
    return compute_weakness_summary(player=player, db=db, min_games=min_games)
