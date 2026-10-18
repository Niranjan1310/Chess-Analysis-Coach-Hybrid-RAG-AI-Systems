"""Tests for POST /api/chat endpoint."""

from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.routes_chat import get_retriever, get_llm, get_engine
from app.rag.vector_store import Chunk
from app.rag.llm_client import FakeLLMClient
from app.core.engine import EvalResult


@pytest.fixture
def client():
    return TestClient(app)


def test_chat_empty_question(client):
    resp = client.post("/api/chat", json={"question": "   ", "history": []})
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_chat_low_score_refusal(client):
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        Chunk(id="c1", text="Unrelated text.", metadata={}, score=0.1)
    ]
    app.dependency_overrides[get_retriever] = lambda: mock_retriever
    app.dependency_overrides[get_llm] = lambda: FakeLLMClient()
    app.dependency_overrides[get_engine] = lambda: None

    try:
        resp = client.post("/api/chat", json={
            "question": "What is the capital of France?",
            "history": []
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["confident"] is False
        assert "don't have enough information" in data["answer"].lower()
        assert data["sources"] == []
    finally:
        app.dependency_overrides.clear()


def test_chat_successful_answer(client):
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        Chunk(id="sicilian_najdorf", text="The Sicilian Najdorf is sharp.", metadata={}, score=0.88)
    ]
    fake_llm = FakeLLMClient(response="The Najdorf is an aggressive Sicilian choice [sicilian_najdorf].")
    app.dependency_overrides[get_retriever] = lambda: mock_retriever
    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_engine] = lambda: None

    try:
        resp = client.post("/api/chat", json={
            "question": "Why play the Najdorf?",
            "history": [{"role": "user", "content": "Hello"}]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["confident"] is True
        assert "sicilian_najdorf" in data["sources"]
        assert "Najdorf" in data["answer"]
    finally:
        app.dependency_overrides.clear()


def test_chat_fen_position_evaluation(client):
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    mock_engine = MagicMock()
    mock_engine.evaluate.return_value = EvalResult(
        score_cp=35,
        best_move_uci="c7c5",
        pv=["c7c5", "g1f3"],
    )
    fake_llm = FakeLLMClient(response="In this position, Black should play c7c5.")
    app.dependency_overrides[get_retriever] = lambda: None
    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_engine] = lambda: mock_engine

    try:
        resp = client.post("/api/chat", json={
            "question": f"What is the best move in {fen}?",
            "history": []
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["confident"] is True
        assert "c7c5" in data["engine_eval"]
        assert "c7c5" in data["answer"]
    finally:
        app.dependency_overrides.clear()
