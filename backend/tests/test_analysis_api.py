"""Tests for api/routes_analysis.py."""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app
from app.api.routes_analysis import clear_jobs
from app.schemas.analysis import AnalyzeResult, MoveExplanation


@pytest.fixture(autouse=True)
def reset_job_store():
    clear_jobs()
    yield
    clear_jobs()


@pytest.fixture
def client():
    return TestClient(app)


def test_start_analysis_and_poll_completed(client):
    mock_result = AnalyzeResult(
        total_moves=4,
        flagged_moves=1,
        explanations=[
            MoveExplanation(
                move_number=2,
                side="white",
                played="Qh5",
                best="Nf3",
                classification="blunder",
                eval_before="+0.20",
                eval_after="-3.50",
                explanation="White played Qh5 prematurely.",
                sources=[],
                faithful=True,
                engine_only=False,
            )
        ],
    )

    with patch("app.api.routes_analysis.analyze_game", return_value=mock_result) as mock_analyze:
        # Submit job
        response = client.post("/api/analyze", json={"pgn": "1. e4 e5 2. Qh5", "depth": 10})
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        job_id = data["job_id"]

        # BackgroundTasks run synchronously in Starlette TestClient
        # Check job status
        poll_resp = client.get(f"/api/analyze/{job_id}")
        assert poll_resp.status_code == 200
        poll_data = poll_resp.json()
        assert poll_data["job_id"] == job_id
        assert poll_data["status"] == "completed"
        assert poll_data["result"]["total_moves"] == 4
        assert poll_data["result"]["flagged_moves"] == 1
        assert poll_data["result"]["explanations"][0]["played"] == "Qh5"
        mock_analyze.assert_called_once_with("1. e4 e5 2. Qh5", depth=10)


def test_analysis_job_failure(client):
    with patch("app.api.routes_analysis.analyze_game", side_effect=ValueError("Invalid PGN content")):
        response = client.post("/api/analyze", json={"pgn": "invalid", "depth": 8})
        assert response.status_code == 200
        job_id = response.json()["job_id"]

        poll_resp = client.get(f"/api/analyze/{job_id}")
        assert poll_resp.status_code == 200
        poll_data = poll_resp.json()
        assert poll_data["status"] == "failed"
        assert "Invalid PGN content" in poll_data["error"]


def test_get_nonexistent_job(client):
    response = client.get("/api/analyze/nonexistent-id-999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_invalid_request_body(client):
    response = client.post("/api/analyze", json={"depth": 14})
    assert response.status_code == 422
