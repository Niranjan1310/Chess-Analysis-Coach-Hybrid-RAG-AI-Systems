"""Tests for rag/similar_positions.py."""

import pytest
from unittest.mock import MagicMock
from app.rag.vector_store import Chunk
from app.rag.llm_client import FakeLLMClient
from app.rag.similar_positions import (
    find_similar_positions,
    get_material_signature,
    SimilarPositionsResult,
)


def test_invalid_fen():
    with pytest.raises(ValueError, match="Invalid FEN"):
        find_similar_positions("invalid-fen-string")


def test_not_enough_matches():
    mock_store = MagicMock()
    # Returns only 2 matches (min_matches=5)
    mock_store.query.return_value = [
        Chunk(id="1", text="summary", score=0.9, metadata={"next_move_san": "Nf3", "result": "1-0"}),
        Chunk(id="2", text="summary", score=0.8, metadata={"next_move_san": "d4", "result": "1/2-1/2"}),
    ]
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    res = find_similar_positions(fen, vector_store=mock_store, min_matches=5)

    assert res.status == "not_enough_matches"
    assert res.matches_count == 2
    assert "Not enough similar positions found" in res.message


def test_similar_positions_stats_calculation():
    # 6 matches for White
    chunks = [
        Chunk(id="1", text="pos", score=0.9, metadata={"next_move_san": "Nf3", "result": "1-0", "white": "A", "black": "B", "year": "2020"}),
        Chunk(id="2", text="pos", score=0.9, metadata={"next_move_san": "Nf3", "result": "1/2-1/2", "white": "C", "black": "D", "year": "2021"}),
        Chunk(id="3", text="pos", score=0.9, metadata={"next_move_san": "Nf3", "result": "0-1", "white": "E", "black": "F", "year": "2022"}),
        Chunk(id="4", text="pos", score=0.85, metadata={"next_move_san": "d4", "result": "1-0", "white": "G", "black": "H", "year": "2023"}),
        Chunk(id="5", text="pos", score=0.85, metadata={"next_move_san": "d4", "result": "1-0", "white": "I", "black": "J", "year": "2024"}),
        Chunk(id="6", text="pos", score=0.8, metadata={"next_move_san": "c4", "result": "0-1", "white": "K", "black": "L", "year": "2025"}),
    ]
    mock_store = MagicMock()
    mock_store.query.return_value = chunks
    fake_llm = FakeLLMClient(response="Masters most often chose Nf3 (3 games) or d4 (100% win rate).")

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    res = find_similar_positions(fen, vector_store=mock_store, llm=fake_llm, min_matches=5)

    assert res.status == "ready"
    assert res.matches_count == 6
    assert res.side_to_move == "white"

    # Moves should be sorted by frequency: Nf3 (3), d4 (2), c4 (1)
    assert len(res.moves) == 3
    assert res.moves[0].move == "Nf3"
    assert res.moves[0].count == 3
    # Nf3: 1W, 1D, 1L -> (1 + 0.5) / 3 = 0.5 (50.0%)
    assert res.moves[0].score_pct == 0.5
    assert res.moves[0].wins == 1
    assert res.moves[0].draws == 1
    assert res.moves[0].losses == 1

    assert res.moves[1].move == "d4"
    assert res.moves[1].count == 2
    # d4: 2W, 0D, 0L -> 1.0 (100.0%)
    assert res.moves[1].score_pct == 1.0

    assert len(res.example_games) >= 2
    assert "Nf3" in res.summary
