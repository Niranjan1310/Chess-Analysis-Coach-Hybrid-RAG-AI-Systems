"""Tests for ingestion/build_position_index.py."""

import pytest
import chess
from app.rag.vector_store import VectorStore
from tests.test_vector_store import FakeEmbedder
from ingestion.build_position_index import (
    get_material_signature,
    sample_game_positions,
    build_position_index,
)

SAMPLE_GAME = """[Event "World Championship"]
[Site "London"]
[Date "2018.11.09"]
[White "Carlsen, Magnus"]
[Black "Caruana, Fabiano"]
[Result "1/2-1/2"]
[ECO "B31"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 g6 4. Bxc6 dxc6 5. d3 Bg7 6. h3 Nf6 7. Nc3 Nd7
8. Be3 e5 9. O-O b6 10. a3 O-O 11. b4 Qe7 12. Rb1 Rd8 13. Na4 cxb4 14. axb4 1/2-1/2
"""


def test_material_signature():
    board = chess.Board()
    sig = get_material_signature(board)
    assert "BBNNQR" in sig
    assert "v" in sig


def test_sample_game_positions():
    positions = sample_game_positions(SAMPLE_GAME, min_move=8, max_move=12)
    assert len(positions) > 0
    for p in positions:
        assert "id" in p
        assert "text" in p
        meta = p["metadata"]
        assert meta["side_to_move"] in ("white", "black")
        assert meta["phase"] in ("opening", "middlegame", "endgame")
        assert "material_signature" in meta
        assert "next_move_san" in meta
        assert meta["result"] == "1/2-1/2"
        assert meta["white"] == "Carlsen, Magnus"


def test_build_position_index(tmp_path):
    chroma_dir = tmp_path / "chroma"
    fake = FakeEmbedder()
    count = build_position_index(
        pgn_source=SAMPLE_GAME,
        persist_dir=str(chroma_dir),
        collection_name="test_positions",
        min_move=8,
        max_move=10,
        embedder=fake,
    )
    assert count > 0

    store = VectorStore(
        persist_dir=str(chroma_dir),
        collection_name="test_positions",
        embedder=fake,
    )
    assert store.count() == count
