"""Tests for rag/generator.py move-faithfulness check and explanation generation."""

import pytest
import chess
from app.core.engine import EvalResult
from app.rag.vector_store import Chunk
from app.rag.generator import (
    MoveContext,
    Explanation,
    explain_move,
    _check_faithfulness,
    _uci_to_san,
    _best_move_san,
    _engine_only_explanation,
)
from app.rag.llm_client import FakeLLMClient


class SequentialFakeLLM:
    """Returns responses in sequence on each call."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.call_count = 0
        self.last_user = ""

    def generate(self, system: str, user: str) -> str:
        self.call_count += 1
        self.last_user = user
        if self.responses:
            return self.responses.pop(0)
        return "Fallback text"


@pytest.fixture
def sample_context():
    # Starting position after 1. e4 e5
    fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    return MoveContext(
        fen_before=fen,
        played_move_san="Qh5",
        played_move_uci="d1h5",
        best_move_uci="g1f3",
        best_line_uci=["g1f3", "b8c6", "f1b5"],
        eval_before=EvalResult(score_cp=30, best_move_uci="g1f3", pv=["g1f3", "b8c6", "f1b5"]),
        eval_after=EvalResult(score_cp=-80, best_move_uci="b8c6", pv=["b8c6", "g1f3"]),
        side="white",
        move_number=2,
        position_description="Open game, King pawn opening.",
        game_moves_san=["e4", "e5", "Qh5"],
    )


@pytest.fixture
def sample_chunks():
    return [
        Chunk(id="c1", text="Developing knights before queens is recommended.", score=0.85, metadata={}),
    ]


def test_uci_to_san(sample_context):
    san_moves = _uci_to_san(sample_context.fen_before, sample_context.best_line_uci)
    assert san_moves == ["Nf3", "Nc6", "Bb5"]


def test_best_move_san(sample_context):
    san = _best_move_san(sample_context.fen_before, "g1f3")
    assert san == "Nf3"


def test_check_faithfulness_allowed_moves(sample_context):
    # Mentions played move Qh5, best move Nf3, and game move e4
    text = "Playing Qh5 was premature. White should develop with Nf3 after 1. e4."
    is_faithful, hall = _check_faithfulness(
        llm_text=text,
        fen_before=sample_context.fen_before,
        played_san=sample_context.played_move_san,
        pv_uci=sample_context.best_line_uci,
        game_moves_san=sample_context.game_moves_san,
    )
    assert is_faithful is True
    assert len(hall) == 0


def test_check_faithfulness_hallucinated_moves(sample_context):
    # Mentions Nxd5 and Bg5 which are not in the PV, played move, or game
    text = "Instead of Qh5, white had Nxd5 or Bg5 winning immediately."
    is_faithful, hall = _check_faithfulness(
        llm_text=text,
        fen_before=sample_context.fen_before,
        played_san=sample_context.played_move_san,
        pv_uci=sample_context.best_line_uci,
        game_moves_san=sample_context.game_moves_san,
    )
    assert is_faithful is False
    assert "Nxd5" in hall or "Bg5" in hall


def test_explain_move_faithful_on_first_try(sample_context, sample_chunks):
    fake = FakeLLMClient(
        response="White played Qh5 which loses tempo. Nf3 was better developing the knight. [c1]"
    )
    result = explain_move(sample_context, sample_chunks, llm=fake)
    assert result.faithful is True
    assert result.engine_only is False
    assert result.sources == ["c1"]
    assert fake.call_count == 1


def test_explain_move_retries_and_succeeds(sample_context, sample_chunks):
    # First response hallucinates Bxh7+, second is faithful
    llm = SequentialFakeLLM([
        "White should have played Bxh7+ winning the king.",
        "Qh5 was inaccurate. Better was Nf3 to develop cleanly.",
    ])
    result = explain_move(sample_context, sample_chunks, llm=llm, max_retries=1)
    assert result.faithful is True
    assert result.engine_only is False
    assert llm.call_count == 2
    assert "WARNING" in llm.last_user


def test_explain_move_falls_back_to_engine_after_retries(sample_context, sample_chunks):
    # Both responses hallucinate
    llm = SequentialFakeLLM([
        "White should have played Bxh7+.",
        "White should have played Qxe5+.",
    ])
    result = explain_move(sample_context, sample_chunks, llm=llm, max_retries=1)
    assert result.engine_only is True
    assert result.sources == []
    assert "The move Qh5 lost approximately" in result.text
    assert "Nf3" in result.text


def test_explain_move_llm_runtime_error_fallback(sample_context, sample_chunks):
    class ErrorLLM:
        def generate(self, system: str, user: str) -> str:
            raise RuntimeError("API failure")

    result = explain_move(sample_context, sample_chunks, llm=ErrorLLM())
    assert result.engine_only is True
    assert result.sources == []
    assert "The move Qh5 lost approximately" in result.text
