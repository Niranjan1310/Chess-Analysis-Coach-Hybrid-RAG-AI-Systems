"""
Tests for app.core.pgn_parser – PGN parsing into PositionInfo records.
"""

import pytest

from app.core.pgn_parser import parse_pgn


class TestParsePgn:
    """Tests for the parse_pgn function."""

    def test_move_count(self, scholars_mate_pgn):
        """Scholar's Mate has exactly 7 half-moves."""
        positions = parse_pgn(scholars_mate_pgn)
        assert len(positions) == 7

    def test_first_move(self, scholars_mate_pgn):
        """First half-move is 1. e4 by White."""
        first = parse_pgn(scholars_mate_pgn)[0]

        assert first.move_san == "e4"
        assert first.move_uci == "e2e4"
        assert first.move_number == 1
        assert first.side == "white"

    def test_second_move(self, scholars_mate_pgn):
        """Second half-move is 1… e5 by Black."""
        second = parse_pgn(scholars_mate_pgn)[1]

        assert second.move_san == "e5"
        assert second.move_uci == "e7e5"
        assert second.move_number == 1
        assert second.side == "black"

    def test_last_move_is_checkmate(self, scholars_mate_pgn):
        """Last half-move is 4. Qxf7# by White."""
        last = parse_pgn(scholars_mate_pgn)[-1]

        assert last.move_san == "Qxf7#"
        assert last.move_number == 4
        assert last.side == "white"

    def test_fen_chain_is_consistent(self, scholars_mate_pgn):
        """fen_after of move N must equal fen_before of move N+1."""
        positions = parse_pgn(scholars_mate_pgn)

        for i in range(len(positions) - 1):
            assert positions[i].fen_after == positions[i + 1].fen_before, (
                f"fen_after[{i}] != fen_before[{i + 1}]"
            )

    def test_starting_fen(self, scholars_mate_pgn):
        """First fen_before is the standard starting position."""
        first = parse_pgn(scholars_mate_pgn)[0]
        assert first.fen_before.startswith(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        )

    def test_headers_extracted(self, scholars_mate_pgn):
        """All key game headers are present."""
        h = parse_pgn(scholars_mate_pgn)[0].headers

        assert h.white == "Alice"
        assert h.black == "Bob"
        assert h.result == "1-0"
        assert h.eco == "C20"
        assert h.opening == "King's Pawn Game"

    def test_headers_shared_across_moves(self, scholars_mate_pgn):
        """Every move in the game carries the same header values."""
        positions = parse_pgn(scholars_mate_pgn)
        first_headers = positions[0].headers
        for pos in positions[1:]:
            assert pos.headers == first_headers

    def test_missing_optional_headers(self):
        """ECO and Opening are None when absent from the PGN."""
        pgn = '[White "X"]\n[Black "Y"]\n[Result "0-1"]\n\n1. e4 e5 0-1\n'
        positions = parse_pgn(pgn)
        assert positions[0].headers.eco is None
        assert positions[0].headers.opening is None

    def test_empty_pgn_raises(self):
        """Empty input raises ValueError."""
        with pytest.raises(ValueError, match="Could not parse PGN"):
            parse_pgn("")

    def test_fen_after_last_move(self, scholars_mate_pgn):
        """fen_after of the final move reflects the checkmate position."""
        last = parse_pgn(scholars_mate_pgn)[-1]
        # After Qxf7#, the White queen is on f7 (shown as 'Q' in the FEN).
        assert last.fen_after.startswith("r1bqkb1r/pppp1Qpp")
