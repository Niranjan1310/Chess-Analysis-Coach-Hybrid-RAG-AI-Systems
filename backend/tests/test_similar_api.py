"""Tests for /api/similar endpoint."""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app
from app.rag.similar_positions import SimilarPositionsResult, SimilarMoveStat


@pytest.fixture
def client():
    return TestClient(app)


def test_similar_api_success(client):
    mock_result = SimilarPositionsResult(
        status="ready",
        matches_count=10,
        side_to_move="white",
        phase="opening",
        material_signature="BNQR v BNQR",
        moves=[
            SimilarMoveStat(move="Nf3", count=6, score_pct=0.667, wins=4, draws=0, losses=2),
            SimilarMoveStat(move="d4", count=4, score_pct=0.5, wins=2, draws=0, losses=2),
        ],
        example_games=[],
        summary="Masters most often played Nf3.",
    )

    with patch("app.api.routes_similar.find_similar_positions", return_value=mock_result):
        resp = client.post(
            "/api/similar",
            json={"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "k": 10},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["matches_count"] == 10
        assert data["moves"][0]["move"] == "Nf3"
        assert data["moves"][0]["count"] == 6


def test_similar_api_invalid_fen(client):
    with patch("app.api.routes_similar.find_similar_positions", side_effect=ValueError("Invalid FEN")):
        resp = client.post("/api/similar", json={"fen": "bad-fen", "k": 10})
        assert resp.status_code == 400
        assert "Invalid FEN" in resp.json()["detail"]
