"""
Tests for the FastAPI application (app.main) and static frontend serving.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check_returns_ok():
    """GET /api/health returns 200 with {"status": "ok"}."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_frontend_index_served():
    """GET / returns the frontend HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Chess Analysis Coach" in response.text
    assert 'id="board"' in response.text
    assert 'id="pgn-input"' in response.text


def test_frontend_static_assets_served():
    """GET /style.css and /app.js return static assets."""
    css_resp = client.get("/style.css")
    assert css_resp.status_code == 200
    assert "Chess Analysis Coach" in css_resp.text

    js_resp = client.get("/app.js")
    assert js_resp.status_code == 200
    assert "Chess Analysis Coach" in js_resp.text
