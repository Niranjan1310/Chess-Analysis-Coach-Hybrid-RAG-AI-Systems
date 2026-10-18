"""Tests for rag/chat.py."""

import pytest
from unittest.mock import MagicMock
from app.rag.vector_store import Chunk
from app.rag.llm_client import FakeLLMClient
from app.core.engine import EvalResult
from app.rag.chat import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    extract_fen,
    answer_chess_question,
)


def test_empty_question_raises():
    with pytest.raises(ValueError, match="empty"):
        answer_chess_question("   ")


def test_extract_fen():
    text = "What do you think about rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1 here?"
    fen = extract_fen(text)
    assert fen == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"

    no_fen = "Why is the French Defense popular?"
    assert extract_fen(no_fen) is None


def test_i_dont_know_rule_low_score():
    mock_retriever = MagicMock()
    # Chunk score is 0.1, below default min_score 0.3
    mock_retriever.retrieve.return_value = [
        Chunk(id="unrelated", text="Tennis rules and scoring.", metadata={}, score=0.1)
    ]
    resp = answer_chess_question(
        "Who won Wimbledon in 2020?",
        retriever=mock_retriever,
        min_score=0.3,
    )
    assert resp.confident is False
    assert "don't have enough information" in resp.answer
    assert resp.sources == []


def test_successful_chat_with_sources():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        Chunk(id="caro_adv", text="The Caro-Kann Advance Variation 3. e5 Bf5 gains space.", metadata={}, score=0.85)
    ]
    fake_llm = FakeLLMClient(response="The Caro-Kann Advance gains central space and restricts Black's pawns [caro_adv].")

    resp = answer_chess_question(
        "Why is the Caro-Kann Advance solid?",
        retriever=mock_retriever,
        llm=fake_llm,
        min_score=0.3,
    )
    assert resp.confident is True
    assert "caro_adv" in resp.sources
    assert "Caro-Kann" in resp.answer
    assert "[caro_adv]" in fake_llm.last_user


def test_chat_memory_trimming():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        Chunk(id="c1", text="Opening principles.", metadata={}, score=0.9)
    ]
    fake_llm = FakeLLMClient()

    # Provide 6 history messages
    history = [
        ChatMessage(role="user", content=f"Question {i}")
        for i in range(6)
    ]
    answer_chess_question(
        "Latest question?",
        history=history,
        retriever=mock_retriever,
        llm=fake_llm,
    )

    # Only last 4 should be in prompt
    assert "Question 0" not in fake_llm.last_user
    assert "Question 1" not in fake_llm.last_user
    assert "Question 2" in fake_llm.last_user
    assert "Question 5" in fake_llm.last_user


def test_fen_engine_call():
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    mock_engine = MagicMock()
    mock_engine.evaluate.return_value = EvalResult(
        score_cp=30,
        best_move_uci="c7c5",
        pv=["c7c5", "g1f3"],
    )
    fake_llm = FakeLLMClient()

    resp = answer_chess_question(
        f"What should Black play in {fen}?",
        engine=mock_engine,
        llm=fake_llm,
        retriever=None,
    )

    assert mock_engine.evaluate.called
    assert resp.confident is True
    assert "c7c5" in resp.engine_eval
    assert "c7c5" in fake_llm.last_user
