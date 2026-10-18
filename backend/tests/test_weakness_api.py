"""Tests for weakness and games API routes."""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.models.database import Base, get_db, init_db
from app.models.game import Game
from app.models.move import MoveRecord
from app.schemas.analysis import AnalyzeResult


@pytest.fixture
def db_session():
    # Use in-memory SQLite with StaticPool so all threads share the same DB
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSession()

    def override_get_db():
        s = TestingSession()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield session
    finally:
        session.close()
        app.dependency_overrides.clear()


@pytest.fixture
def client(db_session):
    return TestClient(app)


def test_get_weaknesses_not_enough_games(client, db_session):
    resp = client.get("/api/weaknesses?player=Hikaru")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "not_enough_games"
    assert "Need at least 5" in data["message"]


def test_list_games_empty(client, db_session):
    resp = client.get("/api/games")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_games_with_data(client, db_session):
    g1 = Game(
        id="g-1",
        source="upload",
        player_analyzed="Hikaru",
        white_player="Hikaru",
        black_player="Magnus",
        result="1-0",
        total_moves=40,
        flagged_moves=2,
        pgn="1. e4",
    )
    db_session.add(g1)
    db_session.commit()

    resp = client.get("/api/games?player=Hikaru")
    assert resp.status_code == 200
    games = resp.json()
    assert len(games) == 1
    assert games[0]["id"] == "g-1"
    assert games[0]["white_player"] == "Hikaru"


def test_import_lichess_games(client, db_session):
    sample_pgn = "1. e4 e5 2. Nf3 Nc6"
    mock_result = AnalyzeResult(total_moves=2, flagged_moves=0, explanations=[])

    with patch("app.services.lichess_client.LichessClient.fetch_user_games", return_value=[sample_pgn]):
        with patch("app.api.routes_games.analyze_game", return_value=mock_result):
            resp = client.post("/api/games/lichess", json={"username": "testuser", "max_games": 1})
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "success"
            assert data["imported_games"] == 1
            assert "testuser" in data["player"]

            # Verify saved in DB
            saved = db_session.query(Game).filter_by(player_analyzed="testuser").first()
            assert saved is not None
