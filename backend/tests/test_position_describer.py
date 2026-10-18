"""
Tests for app.core.position_describer.
"""

import chess
import pytest

from app.core.position_describer import (
    PositionDescription,
    describe_position,
    detect_phase,
    _count_pieces,
    _material_balance_text,
    _pawn_structure,
    _king_safety,
    _center_control,
)


# -- Starting position (standard) ----------------------------------------

STARTING_FEN = chess.STARTING_FEN


class TestDetectPhase:
    """Game-phase heuristic based on piece count."""

    def test_starting_position_is_opening(self):
        board = chess.Board(STARTING_FEN)
        assert detect_phase(board) == "opening"

    def test_middlegame(self):
        # Position after several exchanges: ~18 non-king pieces
        # Both queens traded, a pair of knights traded
        fen = "r4rk1/pp3ppp/2n1pn2/8/3PP3/2N2N2/PP3PPP/R4RK1 w - - 0 1"
        board = chess.Board(fen)
        n = _count_pieces(board)
        assert 11 <= n <= 23, f"Expected middlegame piece count, got {n}"
        assert detect_phase(board) == "middlegame"

    def test_endgame(self):
        # King + rook vs king + rook = 2 non-king pieces
        fen = "8/8/4k3/8/8/4K3/4R3/4r3 w - - 0 1"
        board = chess.Board(fen)
        assert detect_phase(board) == "endgame"


class TestMaterialBalance:

    def test_starting_position_equal(self):
        board = chess.Board(STARTING_FEN)
        assert _material_balance_text(board) == "Material is equal"

    def test_white_up_a_pawn(self):
        # Remove one black pawn
        board = chess.Board(STARTING_FEN)
        board.remove_piece_at(chess.A7)
        assert "White is up a pawn" in _material_balance_text(board)

    def test_black_up_a_queen(self):
        board = chess.Board(STARTING_FEN)
        board.remove_piece_at(chess.D1)  # remove white queen
        result = _material_balance_text(board)
        assert "Black" in result


class TestPawnStructure:

    def test_no_issues_in_starting_position(self):
        board = chess.Board(STARTING_FEN)
        result = _pawn_structure(board)
        assert "No major pawn structure issues" in result

    def test_doubled_pawns(self):
        # After capturing toward center, White might have doubled c-pawns
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1"
        board = chess.Board(fen)
        # Manually put a second pawn on c-file
        board.set_piece_at(chess.C3, chess.Piece(chess.PAWN, chess.WHITE))
        result = _pawn_structure(board)
        assert "doubled" in result.lower()

    def test_isolated_pawn(self):
        # White pawn only on e-file (a, b, c, f, g, h files empty)
        board = chess.Board()
        board.clear()
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.WHITE))
        result = _pawn_structure(board)
        assert "isolated" in result.lower()


class TestKingSafety:

    def test_starting_position_can_castle(self):
        board = chess.Board(STARTING_FEN)
        result = _king_safety(board)
        assert "castle" in result.lower()

    def test_castled_king(self):
        # White king on g1 (kingside castled), pawns on f2, g2, h2
        board = chess.Board()
        board.clear()
        board.set_piece_at(chess.G1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(chess.F2, chess.Piece(chess.PAWN, chess.WHITE))
        board.set_piece_at(chess.G2, chess.Piece(chess.PAWN, chess.WHITE))
        board.set_piece_at(chess.H2, chess.Piece(chess.PAWN, chess.WHITE))
        result = _king_safety(board)
        assert "castled kingside" in result.lower()
        assert "3/3" in result


class TestCenterControl:

    def test_starting_position_roughly_equal(self):
        board = chess.Board(STARTING_FEN)
        result = _center_control(board)
        assert "equal" in result.lower() or "slight" in result.lower()


class TestDescribePosition:
    """Integration tests for the main describe_position function."""

    def test_returns_position_description(self):
        result = describe_position(STARTING_FEN)
        assert isinstance(result, PositionDescription)

    def test_opening_phase(self):
        result = describe_position(STARTING_FEN)
        assert result.phase == "opening"

    def test_summary_is_nonempty(self):
        result = describe_position(STARTING_FEN)
        assert len(result.summary) > 50

    def test_endgame_position(self):
        fen = "8/8/4k3/8/8/4K3/4R3/4r3 w - - 0 1"
        result = describe_position(fen)
        assert result.phase == "endgame"
        assert "endgame" in result.summary.lower()

    def test_material_imbalance(self):
        # White missing queen
        board = chess.Board(STARTING_FEN)
        board.remove_piece_at(chess.D1)
        result = describe_position(board.fen())
        assert "Black" in result.material_balance
