"""Tests for SQLAlchemy models and database setup."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.database import Base, init_db
from app.models.game import Game
from app.models.move import MoveRecord


@pytest.fixture
def db_session():
    # Use in-memory SQLite for test isolation
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


def test_create_game_with_moves(db_session):
    game = Game(
        id="test-game-1",
        source="upload",
        player_analyzed="Magnus",
        white_player="Magnus",
        black_player="Hikaru",
        result="1-0",
        eco="B30",
        opening_name="Sicilian Defense",
        total_moves=2,
        flagged_moves=1,
        pgn="1. e4 c5 2. Nf3 Nc6",
    )
    db_session.add(game)
    db_session.commit()

    move1 = MoveRecord(
        game_id="test-game-1",
        player_name="Magnus",
        move_number=1,
        side="white",
        move_san="e4",
        move_uci="e2e4",
        fen_before="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        fen_after="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        phase="opening",
        eco="B00",
        opening_name="King Pawn",
        piece_moved="P",
        centipawn_loss=0,
        move_class="good",
        best_move="e2e4",
        game_result_context="equal",
    )
    move2 = MoveRecord(
        game_id="test-game-1",
        player_name="Hikaru",
        move_number=1,
        side="black",
        move_san="c5",
        move_uci="c7c5",
        fen_before="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        fen_after="rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        phase="opening",
        eco="B20",
        opening_name="Sicilian Defense",
        piece_moved="P",
        centipawn_loss=120,
        move_class="mistake",
        best_move="e7e5",
        explanation="Mistake explanation",
        game_result_context="equal",
    )
    db_session.add_all([move1, move2])
    db_session.commit()

    # Query back
    saved_game = db_session.query(Game).filter_by(id="test-game-1").first()
    assert saved_game is not None
    assert saved_game.white_player == "Magnus"
    assert len(saved_game.moves) == 2

    # Query moves by classification
    mistakes = db_session.query(MoveRecord).filter_by(move_class="mistake").all()
    assert len(mistakes) == 1
    assert mistakes[0].move_san == "c5"
    assert mistakes[0].piece_moved == "P"


def test_cascade_delete_game(db_session):
    game = Game(id="game-to-delete", pgn="1. e4")
    move = MoveRecord(
        game_id="game-to-delete",
        move_number=1,
        side="white",
        move_san="e4",
        move_uci="e2e4",
        fen_before="fen1",
        fen_after="fen2",
        phase="opening",
        piece_moved="P",
        move_class="good",
    )
    db_session.add(game)
    db_session.add(move)
    db_session.commit()

    assert db_session.query(MoveRecord).filter_by(game_id="game-to-delete").count() == 1

    db_session.delete(game)
    db_session.commit()

    assert db_session.query(MoveRecord).filter_by(game_id="game-to-delete").count() == 0
