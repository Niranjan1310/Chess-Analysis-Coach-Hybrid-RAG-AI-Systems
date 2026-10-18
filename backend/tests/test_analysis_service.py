"""Tests for services/analysis_service.py."""

import pytest
from unittest.mock import MagicMock
from app.core.engine import EvalResult
from app.core.mistake_detector import MoveClass
from app.rag.vector_store import Chunk
from app.rag.llm_client import FakeLLMClient
from app.services.analysis_service import analyze_game, clear_cache


class FakeEngine:
    """Fake engine returning preconfigured evaluations."""

    def __init__(self, evals_map: dict[str, EvalResult] | None = None, default_eval: EvalResult | None = None):
        self.evals_map = evals_map or {}
        self.default_eval = default_eval or EvalResult(
            score_cp=0,
            best_move_uci="e2e4",
            pv=["e2e4", "e7e5"],
        )
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.closed = True

    def evaluate(self, fen: str, depth: int = 14) -> EvalResult:
        return self.evals_map.get(fen, self.default_eval)


@pytest.fixture(autouse=True)
def reset_cache():
    clear_cache()
    yield
    clear_cache()


def test_analyze_game_no_flagged_moves():
    # 1. e4 e5 with equal evaluations throughout
    pgn = "1. e4 e5"
    fake_engine = FakeEngine(default_eval=EvalResult(score_cp=10, best_move_uci="e2e4", pv=["e2e4"]))
    fake_llm = FakeLLMClient()

    result = analyze_game(
        pgn=pgn,
        engine=fake_engine,
        llm=fake_llm,
    )

    assert result.total_moves == 2
    assert result.flagged_moves == 0
    assert len(result.explanations) == 0
    assert fake_llm.call_count == 0


def test_analyze_game_with_blunder():
    # 1. e4 e5 2. Qh5 (blunder)
    pgn = "1. e4 e5 2. Qh5"
    # Position before 2. Qh5 has eval +20
    # Position after 2. Qh5 has eval -350 (White blundered > 300 cp)
    fen_after_e5 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    fen_after_qh5 = "rnbqkbnr/pppp1ppp/8/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 2"

    evals = {
        fen_after_e5: EvalResult(score_cp=20, best_move_uci="g1f3", pv=["g1f3", "b8c6"]),
        fen_after_qh5: EvalResult(score_cp=-350, best_move_uci="b8c6", pv=["b8c6"]),
    }
    fake_engine = FakeEngine(
        evals_map=evals,
        default_eval=EvalResult(score_cp=20, best_move_uci="e2e4", pv=["e2e4"]),
    )
    fake_llm = FakeLLMClient(response="White played Qh5 blundering material. Nf3 was best. [c1]")

    result = analyze_game(
        pgn=pgn,
        engine=fake_engine,
        llm=fake_llm,
    )

    assert result.total_moves == 3
    assert result.flagged_moves == 1
    exp = result.explanations[0]
    assert exp.played == "Qh5"
    assert exp.classification == "blunder"
    assert exp.move_number == 2
    assert exp.side == "white"
    assert fake_llm.call_count == 1


def test_analyze_game_explanation_caching():
    # Repeat the same game twice; second run should hit cache
    pgn = "1. e4 e5 2. Qh5"
    fen_after_e5 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    fen_after_qh5 = "rnbqkbnr/pppp1ppp/8/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 2"

    evals = {
        fen_after_e5: EvalResult(score_cp=20, best_move_uci="g1f3", pv=["g1f3"]),
        fen_after_qh5: EvalResult(score_cp=-350, best_move_uci="b8c6", pv=["b8c6"]),
    }
    fake_engine = FakeEngine(evals_map=evals)
    fake_llm = FakeLLMClient(response="White played Qh5. Nf3 was better.")

    # Run 1
    res1 = analyze_game(pgn=pgn, engine=fake_engine, llm=fake_llm)
    assert fake_llm.call_count == 1
    assert res1.flagged_moves == 1

    # Run 2
    res2 = analyze_game(pgn=pgn, engine=fake_engine, llm=fake_llm)
    assert fake_llm.call_count == 1  # Did NOT call LLM again
    assert res2.flagged_moves == 1
    assert res2.explanations[0].explanation == res1.explanations[0].explanation


def test_analyze_game_with_retriever():
    pgn = "1. e4 e5 2. Qh5"
    fen_after_e5 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    fen_after_qh5 = "rnbqkbnr/pppp1ppp/8/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 2"

    evals = {
        fen_after_e5: EvalResult(score_cp=20, best_move_uci="g1f3", pv=["g1f3"]),
        fen_after_qh5: EvalResult(score_cp=-350, best_move_uci="b8c6", pv=["b8c6"]),
    }
    fake_engine = FakeEngine(evals_map=evals)
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        Chunk(id="r1", text="Opening principles: avoid queen out early.", metadata={}, score=0.9)
    ]
    fake_llm = FakeLLMClient(response="Qh5 violates opening principles. Nf3 was best. [r1]")

    result = analyze_game(
        pgn=pgn,
        engine=fake_engine,
        retriever=mock_retriever,
        llm=fake_llm,
    )

    assert mock_retriever.retrieve.called
    assert result.flagged_moves == 1
    assert result.explanations[0].sources == ["r1"]
