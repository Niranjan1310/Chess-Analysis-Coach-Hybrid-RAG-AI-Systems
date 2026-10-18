"""
Analysis service — orchestrates the full pipeline:

  PGN -> parse -> evaluate -> classify -> describe -> retrieve -> explain

Caches explanations by (fen, move_uci) so that repeated positions
skip the LLM call.
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.pgn_parser import parse_pgn, PositionInfo
from app.core.engine import StockfishEngine, EvalResult
from app.core.mistake_detector import classify_move, MoveClass, eval_to_cp
from app.core.position_describer import describe_position
from app.rag.vector_store import VectorStore, Chunk
from app.rag.retriever import HybridRetriever
from app.rag.generator import MoveContext, Explanation, explain_move, _eval_to_str
from app.rag.llm_client import LLMClientProtocol, FakeLLMClient
from app.schemas.analysis import MoveExplanation, AnalyzeResult
from app.config import settings

logger = logging.getLogger(__name__)

# In-memory cache: (fen, move_uci) -> Explanation
_explanation_cache: dict[tuple[str, str], Explanation] = {}


def _classify_to_str(mc: MoveClass) -> str:
    return mc.value  # "good", "inaccuracy", "mistake", "blunder"


def analyze_game(
    pgn: str,
    *,
    engine: StockfishEngine | None = None,
    retriever: HybridRetriever | None = None,
    llm: LLMClientProtocol | None = None,
    depth: int | None = None,
) -> AnalyzeResult:
    """Run the full analysis pipeline on a PGN string.

    Args:
        pgn: complete PGN game string.
        engine: Stockfish engine (if None, uses settings.stockfish_path).
        retriever: hybrid retriever (if None, retrieval is skipped).
        llm: LLM client (if None, uses FakeLLMClient).
        depth: override for engine search depth.

    Returns:
        An ``AnalyzeResult`` with explanations for flagged moves.
    """
    if depth is None:
        depth = settings.stockfish_depth
    if llm is None:
        llm = FakeLLMClient()

    # Step 1: Parse PGN
    positions = parse_pgn(pgn)
    all_san_moves = [p.move_san for p in positions]
    total_moves = len(positions)

    # Step 2: Evaluate each position and classify
    evals: list[EvalResult] = []
    classifications: list[MoveClass] = []

    manage_engine = engine is None
    if manage_engine:
        engine = StockfishEngine(settings.stockfish_path)

    try:
        if manage_engine:
            engine.__enter__()

        # Evaluate position BEFORE each move + position AFTER last move
        for pos in positions:
            ev = engine.evaluate(pos.fen_before, depth=depth)
            evals.append(ev)
        # Eval after last move
        last_eval = engine.evaluate(positions[-1].fen_after, depth=depth)
        evals.append(last_eval)

    finally:
        if manage_engine:
            engine.__exit__(None, None, None)

    # Classify each move
    for i, pos in enumerate(positions):
        ev_before = evals[i]
        ev_after = evals[i + 1]
        mc = classify_move(
            ev_before, ev_after, pos.side,
            inaccuracy_cp=settings.inaccuracy_threshold,
            mistake_cp=settings.mistake_threshold,
            blunder_cp=settings.blunder_threshold,
        )
        classifications.append(mc)

    # Step 3: For flagged moves, describe -> retrieve -> explain
    explanations: list[MoveExplanation] = []

    for i, (pos, mc) in enumerate(zip(positions, classifications)):
        if mc == MoveClass.GOOD:
            continue

        # Check cache
        cache_key = (pos.fen_before, pos.move_uci)
        if cache_key in _explanation_cache:
            cached = _explanation_cache[cache_key]
            explanations.append(MoveExplanation(
                move_number=pos.move_number,
                side=pos.side,
                played=pos.move_san,
                best=evals[i].best_move_uci,
                classification=_classify_to_str(mc),
                eval_before=_eval_to_str(evals[i]),
                eval_after=_eval_to_str(evals[i + 1]),
                explanation=cached.text,
                sources=cached.sources,
                faithful=cached.faithful,
                engine_only=cached.engine_only,
            ))
            continue

        # Describe position
        desc = describe_position(pos.fen_before)

        # Retrieve
        chunks: list[Chunk] = []
        if retriever is not None:
            query = f"{desc.summary} {mc.value} {pos.move_san}"
            chunks = retriever.retrieve(query, k=5)

        # Build context and explain
        ctx = MoveContext(
            fen_before=pos.fen_before,
            played_move_san=pos.move_san,
            played_move_uci=pos.move_uci,
            best_move_uci=evals[i].best_move_uci,
            best_line_uci=evals[i].pv,
            eval_before=evals[i],
            eval_after=evals[i + 1],
            side=pos.side,
            move_number=pos.move_number,
            position_description=desc.summary,
            game_moves_san=all_san_moves,
        )

        explanation = explain_move(ctx, chunks, llm=llm)

        # Cache
        _explanation_cache[cache_key] = explanation

        explanations.append(MoveExplanation(
            move_number=pos.move_number,
            side=pos.side,
            played=pos.move_san,
            best=evals[i].best_move_uci,
            classification=_classify_to_str(mc),
            eval_before=_eval_to_str(evals[i]),
            eval_after=_eval_to_str(evals[i + 1]),
            explanation=explanation.text,
            sources=explanation.sources,
            faithful=explanation.faithful,
            engine_only=explanation.engine_only,
        ))

    return AnalyzeResult(
        total_moves=total_moves,
        flagged_moves=len(explanations),
        explanations=explanations,
    )


def clear_cache() -> None:
    """Clear the in-memory explanation cache (for tests)."""
    _explanation_cache.clear()


import io
import chess
import chess.pgn
from sqlalchemy.orm import Session
from app.models.game import Game
from app.models.move import MoveRecord


def save_analyzed_game(
    db: Session,
    pgn: str,
    result: AnalyzeResult,
    player_analyzed: str | None = None,
    source: str = "upload",
) -> Game:
    """Save an analyzed game and its move-by-move evaluation data to the database."""
    # Read PGN headers
    stream = io.StringIO(pgn)
    chess_game = chess.pgn.read_game(stream)
    headers = chess_game.headers if chess_game else {}

    white_p = headers.get("White", "White")
    black_p = headers.get("Black", "Black")
    result_str = headers.get("Result", "*")
    eco = headers.get("ECO", "")
    opening_name = headers.get("Opening", "")

    # Create Game record
    game_obj = Game(
        source=source,
        player_analyzed=player_analyzed or white_p,
        white_player=white_p,
        black_player=black_p,
        result=result_str,
        eco=eco,
        opening_name=opening_name,
        total_moves=result.total_moves,
        flagged_moves=result.flagged_moves,
        pgn=pgn,
    )
    db.add(game_obj)
    db.flush()

    # Map explanations by (move_number, side)
    exp_map = {
        (e.move_number, e.side): e
        for e in result.explanations
    }

    positions = parse_pgn(pgn)
    for pos in positions:
        # Determine player name
        p_name = white_p if pos.side == "white" else black_p

        # Determine piece moved
        board = chess.Board(pos.fen_before)
        try:
            mv = chess.Move.from_uci(pos.move_uci)
            pc = board.piece_at(mv.from_square)
            piece_moved = pc.symbol().upper() if pc else "P"
        except Exception:
            piece_moved = "P"

        # Determine phase
        desc = describe_position(pos.fen_before)
        phase = desc.phase

        # Explanation if flagged
        exp = exp_map.get((pos.move_number, pos.side))
        move_class = exp.classification if exp else "good"
        best_mv = exp.best if exp else pos.move_san
        exp_text = exp.explanation if exp else None
        eval_bef = exp.eval_before if exp else None
        eval_aft = exp.eval_after if exp else None

        # Game context
        game_context = "equal"
        if eval_bef:
            try:
                cp = float(eval_bef) * 100
                if pos.side == "white":
                    game_context = "winning" if cp > 150 else ("losing" if cp < -150 else "equal")
                else:
                    game_context = "winning" if cp < -150 else ("losing" if cp > 150 else "equal")
            except ValueError:
                game_context = "equal"

        mr = MoveRecord(
            game_id=game_obj.id,
            player_name=p_name,
            move_number=pos.move_number,
            side=pos.side,
            move_san=pos.move_san,
            move_uci=pos.move_uci,
            fen_before=pos.fen_before,
            fen_after=pos.fen_after,
            phase=phase,
            eco=eco,
            opening_name=opening_name,
            piece_moved=piece_moved,
            centipawn_loss=0 if move_class == "good" else 100,
            move_class=move_class,
            best_move=best_mv,
            eval_before=eval_bef,
            eval_after=eval_aft,
            explanation=exp_text,
            game_result_context=game_context,
        )
        db.add(mr)

    db.commit()
    db.refresh(game_obj)
    return game_obj
