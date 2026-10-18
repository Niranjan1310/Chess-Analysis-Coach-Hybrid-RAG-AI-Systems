"""
Weakness summary and blunder pattern profiling service.

Computes blunder + mistake rates across:
  - Game phase (opening, middlegame, endgame)
  - Opening / ECO family
  - Piece moved (P, N, B, R, Q, K)
  - Game context (winning, losing, equal)

Uses RAG to retrieve study material for top weaknesses and an LLM to
generate personalized coaching suggestions quoting the exact stats.
"""

from __future__ import annotations

import logging
from typing import Any
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.game import Game
from app.models.move import MoveRecord
from app.rag.retriever import HybridRetriever
from app.rag.llm_client import LLMClientProtocol, FakeLLMClient
from app.config import settings

logger = logging.getLogger(__name__)


class WeaknessCategory(BaseModel):
    """Statistical summary of error rate in one category."""
    dimension: str              # "phase", "opening", "piece", "game_context"
    category: str               # e.g. "endgame", "Sicilian Defense", "N"
    total_moves: int
    mistakes_and_blunders: int
    error_rate: float           # e.g. 0.35 = 35%
    delta_above_average: float  # difference from overall rate


class WeaknessReport(BaseModel):
    """Full weakness report for a player."""
    status: str                 # "ready" or "not_enough_games"
    message: str | None = None
    player: str
    total_games: int
    total_moves: int
    overall_error_rate: float
    top_weaknesses: list[WeaknessCategory]
    coaching_advice: str
    sources: list[str]


def compute_weakness_summary(
    player: str,
    db: Session,
    *,
    retriever: HybridRetriever | None = None,
    llm: LLMClientProtocol | None = None,
    min_games: int | None = None,
    min_moves: int | None = None,
) -> WeaknessReport:
    """Compute weakness summary for a player per Section 11b-2.

    Args:
        player: player username.
        db: SQLAlchemy database session.
        retriever: hybrid retriever for study material.
        llm: LLM client for generating advice.
        min_games: minimum analyzed games required (default: config).
        min_moves: minimum moves in category to qualify as weakness (default: 10).

    Returns:
        A ``WeaknessReport``.
    """
    if min_games is None:
        min_games = getattr(settings, "min_games_for_weakness", 5)
    if min_moves is None:
        min_moves = getattr(settings, "min_moves_for_category", 10)
    if llm is None:
        llm = FakeLLMClient()

    # Step 1: Count games for player
    games_count = db.query(Game).filter(
        (Game.player_analyzed == player) |
        (Game.white_player == player) |
        (Game.black_player == player)
    ).count()

    if games_count < min_games:
        return WeaknessReport(
            status="not_enough_games",
            message=f"Need at least {min_games} analyzed games to compute weaknesses (found {games_count}).",
            player=player,
            total_games=games_count,
            total_moves=0,
            overall_error_rate=0.0,
            top_weaknesses=[],
            coaching_advice="Analyze more games to unlock your weakness profile.",
            sources=[],
        )

    # Step 2: Query all moves by player
    moves = db.query(MoveRecord).filter(
        (MoveRecord.player_name == player)
    ).all()

    total_moves = len(moves)
    if total_moves == 0:
        return WeaknessReport(
            status="not_enough_games",
            message="No moves found for player.",
            player=player,
            total_games=games_count,
            total_moves=0,
            overall_error_rate=0.0,
            top_weaknesses=[],
            coaching_advice="",
            sources=[],
        )

    # Mistakes + Blunders
    total_errors = sum(1 for m in moves if m.move_class in ("mistake", "blunder"))
    overall_error_rate = total_errors / total_moves

    # Step 3: Compute stats per dimension
    dimensions = {
        "phase": lambda m: m.phase,
        "opening": lambda m: m.opening_name or m.eco or "Unknown",
        "piece": lambda m: m.piece_moved,
        "game_context": lambda m: m.game_result_context or "equal",
    }

    candidates: list[WeaknessCategory] = []

    for dim_name, extractor in dimensions.items():
        groups: dict[str, list[MoveRecord]] = {}
        for m in moves:
            val = extractor(m)
            if val:
                groups.setdefault(val, []).append(m)

        for cat_name, cat_moves in groups.items():
            count = len(cat_moves)
            if count < min_moves:
                continue

            errors = sum(1 for m in cat_moves if m.move_class in ("mistake", "blunder"))
            rate = errors / count
            delta = rate - overall_error_rate

            if delta > 0:
                candidates.append(WeaknessCategory(
                    dimension=dim_name,
                    category=cat_name,
                    total_moves=count,
                    mistakes_and_blunders=errors,
                    error_rate=round(rate, 3),
                    delta_above_average=round(delta, 3),
                ))

    # Sort descending by delta above average
    candidates.sort(key=lambda c: c.delta_above_average, reverse=True)
    top_weaknesses = candidates[:3]

    # Step 4: RAG step — retrieve study material for top weaknesses
    retrieved_chunks = []
    sources: list[str] = []

    if retriever is not None and top_weaknesses:
        for w in top_weaknesses:
            q = f"{w.dimension} {w.category} chess strategy mistakes improvement"
            chunks = retriever.retrieve(q, k=2)
            for c in chunks:
                if c.id not in sources:
                    sources.append(c.id)
                    retrieved_chunks.append(f"[{c.id}]: {c.text}")

    # Step 5: LLM coaching summary
    stats_lines = [
        f"- Overall error rate: {overall_error_rate:.1%} across {total_moves} moves in {games_count} games."
    ]
    for w in top_weaknesses:
        stats_lines.append(
            f"- {w.dimension.capitalize()} '{w.category}': {w.error_rate:.1%} error rate "
            f"({w.mistakes_and_blunders}/{w.total_moves} moves, +{w.delta_above_average:.1%} above average)."
        )
    stats_text = "\n".join(stats_lines)

    context_text = "\n\n".join(retrieved_chunks) if retrieved_chunks else "(No extra study notes found.)"

    prompt = f"""You are a chess coach analyzing student weaknesses.
Based ONLY on the statistical numbers and context below, write a short summary (2-3 paragraphs) of the student's top weaknesses with concrete study suggestions.
Quote only numbers given in the stats. Do not invent variations.

## Student: {player}
## Stats
{stats_text}

## Study Material
{context_text}
"""
    system_prompt = "You are a chess coach. Use only the provided statistics and study notes."

    try:
        coaching_advice = llm.generate(system_prompt, prompt)
    except Exception as e:
        logger.warning("LLM call failed for weakness summary: %s", e)
        coaching_advice = (
            f"Based on your {games_count} games, your main weakness is in {top_weaknesses[0].category if top_weaknesses else 'tactics'} "
            f"with an error rate of {top_weaknesses[0].error_rate:.1% if top_weaknesses else 'high'}."
        )

    return WeaknessReport(
        status="ready",
        player=player,
        total_games=games_count,
        total_moves=total_moves,
        overall_error_rate=round(overall_error_rate, 3),
        top_weaknesses=top_weaknesses,
        coaching_advice=coaching_advice,
        sources=sources,
    )
