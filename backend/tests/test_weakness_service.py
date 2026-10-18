"""Tests for services/weakness_service.py."""

import pytest
from unittest.mock import MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.database import Base, init_db
from app.models.game import Game
from app.models.move import MoveRecord
from app.rag.vector_store import Chunk
from app.rag.llm_client import FakeLLMClient
from app.services.weakness_service import compute_weakness_summary


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


def test_weakness_summary_not_enough_games(db_session):
    # Only 2 games in DB (min_games=5)
    for i in range(2):
        g = Game(id=f"g{i}", player_analyzed="Alice", pgn="1. e4")
        db_session.add(g)
    db_session.commit()

    report = compute_weakness_summary("Alice", db_session, min_games=5)
    assert report.status == "not_enough_games"
    assert report.total_games == 2
    assert "Need at least 5" in report.message


def test_weakness_summary_top_weakness_detection(db_session):
    # Create 5 games for Alice
    for i in range(5):
        g = Game(
            id=f"game_{i}",
            player_analyzed="Alice",
            white_player="Alice",
            black_player="Bob",
            pgn="1. e4",
        )
        db_session.add(g)
    db_session.commit()

    # Add 20 moves in endgame, 10 of which are blunders (50% error rate)
    for i in range(20):
        db_session.add(MoveRecord(
            game_id="game_0",
            player_name="Alice",
            move_number=i + 30,
            side="white",
            move_san="Ke3",
            move_uci="e2e3",
            fen_before="fen",
            fen_after="fen2",
            phase="endgame",
            eco="C00",
            opening_name="French Defense",
            piece_moved="K",
            centipawn_loss=0 if i < 10 else 300,
            move_class="good" if i < 10 else "blunder",
            game_result_context="equal",
        ))

    # Add 30 moves in opening, only 3 are inaccuracies/mistakes (10% error rate)
    for i in range(30):
        db_session.add(MoveRecord(
            game_id="game_1",
            player_name="Alice",
            move_number=i + 1,
            side="white",
            move_san="e4",
            move_uci="e2e4",
            fen_before="fen",
            fen_after="fen2",
            phase="opening",
            eco="B30",
            opening_name="Sicilian Defense",
            piece_moved="P",
            centipawn_loss=0 if i < 27 else 150,
            move_class="good" if i < 27 else "mistake",
            game_result_context="equal",
        ))

    db_session.commit()

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        Chunk(id="c_endgame", text="Active king and pawn promotion principles.", metadata={}, score=0.9)
    ]
    fake_llm = FakeLLMClient(response="Your top weakness is the endgame with a 50.0% error rate. [c_endgame]")

    report = compute_weakness_summary(
        "Alice",
        db_session,
        retriever=mock_retriever,
        llm=fake_llm,
        min_games=5,
        min_moves=10,
    )

    assert report.status == "ready"
    assert report.total_games == 5
    assert report.total_moves == 50
    # Top weakness should be endgame
    assert len(report.top_weaknesses) > 0
    top = report.top_weaknesses[0]
    assert top.category == "endgame"
    assert top.error_rate == 0.5
    assert "c_endgame" in report.sources
    assert "50.0%" in report.coaching_advice
