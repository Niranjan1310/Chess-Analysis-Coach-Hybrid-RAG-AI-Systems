"""
Similar Master Position Search.

Goal: "Positions like this were usually won by doing X."

1. Describes the position using position_describer (phase, pawn structure, king safety, etc.).
2. Retrieves top-k nearest positions from the Chroma "positions" collection,
   filtering by side_to_move.
3. Computes move frequency and score % entirely in CODE (never invented by the LLM).
4. Guardrails: If fewer than 5 matches are found, returns "not_enough_matches".
5. LLM writes a 2-3 sentence factual summary citing the computed numbers.
"""

from __future__ import annotations

import logging
from typing import Any
import chess
from pydantic import BaseModel

from app.core.position_describer import describe_position
from app.rag.vector_store import VectorStore, Chunk
from app.rag.llm_client import LLMClientProtocol, FakeLLMClient

logger = logging.getLogger(__name__)


def get_material_signature(board: chess.Board) -> str:
    """Return compact non-pawn material signature (e.g. 'BNQR v BNQR')."""
    w_pieces = sorted([
        p.symbol().upper()
        for p in board.piece_map().values()
        if p.color == chess.WHITE and p.piece_type not in (chess.PAWN, chess.KING)
    ])
    b_pieces = sorted([
        p.symbol().upper()
        for p in board.piece_map().values()
        if p.color == chess.BLACK and p.piece_type not in (chess.PAWN, chess.KING)
    ])
    w_str = "".join(w_pieces) if w_pieces else "-"
    b_str = "".join(b_pieces) if b_pieces else "-"
    return f"{w_str} v {b_str}"


class SimilarMoveStat(BaseModel):
    """Statistical breakdown of a move played by masters."""
    move: str
    count: int
    score_pct: float
    wins: int
    draws: int
    losses: int


class ExampleGame(BaseModel):
    """Sample master game having this or a similar position."""
    white: str
    black: str
    year: str
    result: str
    next_move: str


class SimilarPositionsResult(BaseModel):
    """Result of similar master position search."""
    status: str                         # "ready" or "not_enough_matches"
    message: str | None = None
    matches_count: int
    side_to_move: str
    phase: str
    material_signature: str
    moves: list[SimilarMoveStat]
    example_games: list[ExampleGame]
    summary: str


def find_similar_positions(
    fen: str,
    k: int = 20,
    *,
    vector_store: VectorStore | None = None,
    llm: LLMClientProtocol | None = None,
    min_matches: int = 5,
) -> SimilarPositionsResult:
    """Find similar master positions and calculate move statistics.

    Args:
        fen: position FEN string.
        k: number of neighbors to retrieve.
        vector_store: Chroma vector store targeting the 'positions' collection.
        llm: LLM client for generating summary.
        min_matches: minimum matches required (default 5).

    Returns:
        A ``SimilarPositionsResult``.
    """
    try:
        board = chess.Board(fen)
    except Exception as e:
        raise ValueError(f"Invalid FEN string: {e}") from e

    desc = describe_position(fen)
    side_to_move = "white" if board.turn == chess.WHITE else "black"
    mat_sig = get_material_signature(board)

    if vector_store is None:
        vector_store = VectorStore(collection_name="positions")
    if llm is None:
        llm = FakeLLMClient()

    # Query Chroma
    chunks: list[Chunk] = []
    try:
        chunks = vector_store.query(
            query=desc.summary,
            k=k,
            where={"side_to_move": side_to_move},
        )
    except Exception as e:
        logger.warning("Vector store query for similar positions failed: %s", e)

    matches_count = len(chunks)

    # Guardrail: minimum 5 matches
    if matches_count < min_matches:
        return SimilarPositionsResult(
            status="not_enough_matches",
            message=f"Not enough similar positions found (minimum {min_matches} required, found {matches_count}).",
            matches_count=matches_count,
            side_to_move=side_to_move,
            phase=desc.phase,
            material_signature=mat_sig,
            moves=[],
            example_games=[],
            summary="Not enough similar master positions were found in the database.",
        )

    # Compute move statistics purely by code
    move_groups: dict[str, dict[str, int]] = {}
    for c in chunks:
        meta = c.metadata
        m = meta.get("next_move_san")
        res = meta.get("result", "*")
        if not m:
            continue

        if m not in move_groups:
            move_groups[m] = {"count": 0, "wins": 0, "draws": 0, "losses": 0}

        move_groups[m]["count"] += 1
        # Outcome for side to move
        if side_to_move == "white":
            if res == "1-0":
                move_groups[m]["wins"] += 1
            elif res == "1/2-1/2":
                move_groups[m]["draws"] += 1
            elif res == "0-1":
                move_groups[m]["losses"] += 1
        else:  # black
            if res == "0-1":
                move_groups[m]["wins"] += 1
            elif res == "1/2-1/2":
                move_groups[m]["draws"] += 1
            elif res == "1-0":
                move_groups[m]["losses"] += 1

    move_stats: list[SimilarMoveStat] = []
    for m, stat in move_groups.items():
        cnt = stat["count"]
        score_pct = (stat["wins"] + 0.5 * stat["draws"]) / cnt if cnt > 0 else 0.0
        move_stats.append(SimilarMoveStat(
            move=m,
            count=cnt,
            score_pct=round(score_pct, 3),
            wins=stat["wins"],
            draws=stat["draws"],
            losses=stat["losses"],
        ))

    # Sort moves by frequency descending
    move_stats.sort(key=lambda s: s.count, reverse=True)

    # Extract 2-3 sample games
    example_games: list[ExampleGame] = []
    seen_games = set()
    for c in chunks:
        meta = c.metadata
        key = (meta.get("white"), meta.get("black"), meta.get("year"))
        if key not in seen_games and meta.get("white") and meta.get("black"):
            seen_games.add(key)
            example_games.append(ExampleGame(
                white=meta.get("white", "Unknown"),
                black=meta.get("black", "Unknown"),
                year=meta.get("year", "????"),
                result=meta.get("result", "*"),
                next_move=meta.get("next_move_san", ""),
            ))
            if len(example_games) >= 3:
                break

    # Build prompt for LLM summary
    stats_lines = []
    for s in move_stats[:4]:
        stats_lines.append(
            f"- {s.move}: played {s.count} times ({s.score_pct:.1%} score, {s.wins}W-{s.draws}D-{s.losses}L)"
        )
    stats_summary = "\n".join(stats_lines)

    user_prompt = f"""You are a chess coach summarizing master game statistics for a position.
Based ONLY on the numbers below, write a 2-3 sentence summary of what masters tended to do.
Do not claim certainty. Remind the student that engine evaluation stays the ground truth.

## Position
Phase: {desc.phase}
Side to move: {side_to_move}
Material: {mat_sig}
Matches found: {matches_count}

## Master moves
{stats_summary}
"""
    system_prompt = "You are a factual chess coach. Use only the provided statistics. Do not invent moves."

    try:
        summary = llm.generate(system_prompt, user_prompt)
    except Exception as e:
        logger.warning("LLM call for similar positions summary failed: %s", e)
        top = move_stats[0] if move_stats else None
        if top:
            summary = (
                f"Masters most frequently played {top.move} ({top.count} games, scoring {top.score_pct:.1%}). "
                "Engine evaluation remains the ground truth for tactical correctness."
            )
        else:
            summary = "Similar master positions analyzed."

    return SimilarPositionsResult(
        status="ready",
        matches_count=matches_count,
        side_to_move=side_to_move,
        phase=desc.phase,
        material_signature=mat_sig,
        moves=move_stats,
        example_games=example_games,
        summary=summary,
    )
