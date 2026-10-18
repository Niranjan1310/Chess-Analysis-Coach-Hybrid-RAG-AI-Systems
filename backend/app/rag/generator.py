"""
Explanation generator — calls the LLM with context and validates output.

Key feature: **move-faithfulness check**. Every chess move mentioned
in the LLM's response is parsed with ``python-chess``. If any move
does NOT appear in the engine PV, the played move, or the game moves,
it is flagged. The generator retries once with a warning, then falls
back to an engine-only explanation.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import chess
from pydantic import BaseModel

from app.rag.llm_client import LLMClientProtocol, FakeLLMClient
from app.rag.prompts import SYSTEM_PROMPT, build_user_prompt
from app.rag.vector_store import Chunk
from app.core.engine import EvalResult
from app.core.mistake_detector import eval_to_cp

logger = logging.getLogger(__name__)

# Regex to find SAN-like moves in text (e.g. Nf3, O-O, exd5, Qxf7#)
_MOVE_PATTERN = re.compile(
    r'\b([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?'
    r'|O-O-O|O-O|0-0-0|0-0)\b'
)


class MoveContext(BaseModel):
    """All data needed to explain one mistake."""

    fen_before: str
    played_move_san: str
    played_move_uci: str
    best_move_uci: str
    best_line_uci: list[str]
    eval_before: EvalResult
    eval_after: EvalResult
    side: str   # "white" or "black"
    move_number: int
    position_description: str
    game_moves_san: list[str]  # all SAN moves in the game (for faithfulness)


class Explanation(BaseModel):
    """LLM-generated explanation for a mistake."""

    text: str
    sources: list[str]        # chunk IDs used
    faithful: bool            # True if no hallucinated moves detected
    engine_only: bool         # True if we fell back to engine-only


def _eval_to_str(ev: EvalResult) -> str:
    """Human-readable eval string."""
    if ev.mate_in is not None:
        return f"Mate in {abs(ev.mate_in)} ({'White' if ev.mate_in > 0 else 'Black'})"
    if ev.score_cp is not None:
        return f"{ev.score_cp / 100:+.2f}"
    return "?"


def _uci_to_san(fen: str, uci_moves: list[str]) -> list[str]:
    """Convert a list of UCI moves to SAN, starting from the given FEN."""
    board = chess.Board(fen)
    san_moves: list[str] = []
    for uci in uci_moves:
        try:
            move = chess.Move.from_uci(uci)
            if move in board.legal_moves:
                san_moves.append(board.san(move))
                board.push(move)
            else:
                san_moves.append(uci)  # keep as UCI if not legal
                break
        except (ValueError, chess.InvalidMoveError):
            san_moves.append(uci)
            break
    return san_moves


def _best_move_san(fen: str, best_uci: str) -> str:
    """Convert the best move from UCI to SAN."""
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(best_uci)
        if move in board.legal_moves:
            return board.san(move)
    except (ValueError, chess.InvalidMoveError):
        pass
    return best_uci


def _extract_moves_from_text(text: str) -> set[str]:
    """Extract all chess-move-like tokens from LLM output, ignoring citation brackets."""
    # Strip citation brackets like [c1] or [source-1]
    no_citations = re.sub(r'\[[^\]]*\]', '', text)
    # Normalize castling notation
    normalized = no_citations.replace("0-0-0", "O-O-O").replace("0-0", "O-O")
    return set(_MOVE_PATTERN.findall(normalized))


def _check_faithfulness(
    llm_text: str,
    fen_before: str,
    played_san: str,
    pv_uci: list[str],
    game_moves_san: list[str],
    best_move_uci: str | None = None,
) -> tuple[bool, set[str]]:
    """Check that every move in the LLM output is from known sources.

    Returns (is_faithful, set_of_hallucinated_moves).
    """
    mentioned = _extract_moves_from_text(llm_text)
    if not mentioned:
        return True, set()

    # Build allowed set
    allowed: set[str] = set()
    allowed.add(played_san)
    allowed.update(game_moves_san)

    # Convert PV to SAN and also allow PV UCI moves
    pv_san = _uci_to_san(fen_before, pv_uci)
    allowed.update(pv_san)
    allowed.update(pv_uci)

    if best_move_uci:
        best_san = _best_move_san(fen_before, best_move_uci)
        allowed.add(best_san)
        allowed.add(best_move_uci)

    # Also add castling variants
    if "O-O" in allowed:
        allowed.add("0-0")
    if "O-O-O" in allowed:
        allowed.add("0-0-0")

    # Strip check/mate symbols for comparison
    def _strip(m: str) -> str:
        return m.rstrip("+#")

    allowed_stripped = {_strip(m) for m in allowed}

    hallucinated = set()
    for m in mentioned:
        if _strip(m) not in allowed_stripped and m not in allowed:
            hallucinated.add(m)

    return len(hallucinated) == 0, hallucinated


def _engine_only_explanation(ctx: MoveContext) -> str:
    """Fallback explanation using only engine data."""
    best_san = _best_move_san(ctx.fen_before, ctx.best_move_uci)
    best_line_san = _uci_to_san(ctx.fen_before, ctx.best_line_uci)
    line_str = " ".join(best_line_san) if best_line_san else best_san

    cp_before = eval_to_cp(ctx.eval_before)
    cp_after = eval_to_cp(ctx.eval_after)
    loss = abs(cp_before - cp_after)

    return (
        f"The move {ctx.played_move_san} lost approximately {loss} centipawns. "
        f"The engine's best move was {best_san} with the line: {line_str}. "
        f"(Eval went from {_eval_to_str(ctx.eval_before)} to "
        f"{_eval_to_str(ctx.eval_after)}.) "
        "No relevant context was found in the knowledge base."
    )


def explain_move(
    ctx: MoveContext,
    chunks: list[Chunk],
    llm: LLMClientProtocol | None = None,
    max_retries: int = 1,
) -> Explanation:
    """Generate an explanation for a single mistake.

    Args:
        ctx: all context about the mistake.
        chunks: retrieved knowledge chunks.
        llm: LLM client (defaults to FakeLLMClient for safety).
        max_retries: number of retries on faithfulness failure.

    Returns:
        An ``Explanation`` with the text, sources, and faithfulness flag.
    """
    if llm is None:
        llm = FakeLLMClient()

    best_san = _best_move_san(ctx.fen_before, ctx.best_move_uci)
    best_line_san = _uci_to_san(ctx.fen_before, ctx.best_line_uci)

    chunk_dicts = [
        {"id": c.id, "text": c.text, "score": c.score}
        for c in chunks
    ]
    source_ids = [c.id for c in chunks]

    user_prompt = build_user_prompt(
        played_move=ctx.played_move_san,
        best_move=best_san,
        best_line=best_line_san,
        eval_before=_eval_to_str(ctx.eval_before),
        eval_after=_eval_to_str(ctx.eval_after),
        position_description=ctx.position_description,
        chunks=chunk_dicts,
    )

    attempts = 0
    while attempts <= max_retries:
        try:
            text = llm.generate(SYSTEM_PROMPT, user_prompt)
        except RuntimeError:
            logger.warning("LLM call failed, falling back to engine-only.")
            return Explanation(
                text=_engine_only_explanation(ctx),
                sources=[],
                faithful=True,
                engine_only=True,
            )

        faithful, hallucinated = _check_faithfulness(
            text,
            ctx.fen_before,
            ctx.played_move_san,
            ctx.best_line_uci,
            ctx.game_moves_san,
            best_move_uci=ctx.best_move_uci,
        )

        if faithful:
            return Explanation(
                text=text,
                sources=source_ids,
                faithful=True,
                engine_only=False,
            )

        attempts += 1
        logger.warning(
            "Hallucinated moves detected: %s. Retry %d/%d.",
            hallucinated, attempts, max_retries + 1,
        )
        if attempts > max_retries:
            break
        # Add warning to prompt for retry
        user_prompt += (
            f"\n\nWARNING: Your previous response contained moves not in "
            f"the engine line: {hallucinated}. Do NOT invent moves. "
            f"Only use the moves listed above."
        )

    # All retries failed — fall back to engine-only
    logger.warning("Faithfulness check failed after retries. Using engine-only.")
    return Explanation(
        text=_engine_only_explanation(ctx),
        sources=[],
        faithful=False,
        engine_only=True,
    )
