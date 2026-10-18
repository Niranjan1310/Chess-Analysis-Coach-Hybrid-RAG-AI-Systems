"""
Tests for app.core.engine – the Stockfish wrapper.

Most tests mock the chess.engine module so Stockfish is NOT required.
One integration test uses real Stockfish and is automatically skipped
when the STOCKFISH_PATH environment variable is not set.
"""

import os
from unittest.mock import MagicMock, patch

import chess
import chess.engine
import pytest

from app.core.engine import EvalResult, StockfishEngine


# ── Helper ───────────────────────────────────────────────────────


def _make_mock_engine(score, pv_moves):
    """Create a mock SimpleEngine that returns the given score & PV."""
    mock_eng = MagicMock()
    mock_eng.analyse.return_value = {
        "score": score,
        "pv": pv_moves,
    }
    return mock_eng


# ── Mocked tests ─────────────────────────────────────────────────


class TestStockfishEngineMocked:
    """Tests that use a mocked chess.engine.SimpleEngine."""

    def test_evaluate_centipawn_score(self):
        """evaluate() correctly extracts a centipawn score."""
        mock_score = chess.engine.PovScore(chess.engine.Cp(150), chess.WHITE)
        mock_pv = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("e7e5")]
        mock_eng = _make_mock_engine(mock_score, mock_pv)

        with patch("chess.engine.SimpleEngine.popen_uci", return_value=mock_eng):
            with StockfishEngine("fake") as engine:
                result = engine.evaluate(
                    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                )

        assert result.score_cp == 150
        assert result.mate_in is None
        assert result.best_move_uci == "e2e4"
        assert result.pv == ["e2e4", "e7e5"]

    def test_evaluate_mate_score(self):
        """evaluate() correctly extracts a mate-in-N score."""
        # Mate in 2 for White.
        mock_score = chess.engine.PovScore(chess.engine.Mate(2), chess.WHITE)
        mock_pv = [chess.Move.from_uci("d1h5")]
        mock_eng = _make_mock_engine(mock_score, mock_pv)

        with patch("chess.engine.SimpleEngine.popen_uci", return_value=mock_eng):
            with StockfishEngine("fake") as engine:
                result = engine.evaluate(
                    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                )

        assert result.score_cp is None
        assert result.mate_in == 2
        assert result.best_move_uci == "d1h5"

    def test_evaluate_negative_score_for_black(self):
        """When Black is better, the score is negative from White's POV."""
        # Engine says Black is +80 cp better → White's POV = -80.
        mock_score = chess.engine.PovScore(chess.engine.Cp(80), chess.BLACK)
        mock_pv = [chess.Move.from_uci("e7e5")]
        mock_eng = _make_mock_engine(mock_score, mock_pv)

        with patch("chess.engine.SimpleEngine.popen_uci", return_value=mock_eng):
            with StockfishEngine("fake") as engine:
                result = engine.evaluate(
                    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
                )

        # 80 cp for Black = -80 cp from White's perspective.
        assert result.score_cp == -80

    def test_context_manager_calls_quit(self):
        """__exit__ calls engine.quit() to clean up the process."""
        mock_eng = _make_mock_engine(
            chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE), []
        )

        with patch("chess.engine.SimpleEngine.popen_uci", return_value=mock_eng):
            with StockfishEngine("fake"):
                pass  # nothing to do

        mock_eng.quit.assert_called_once()

    def test_evaluate_without_context_manager_raises(self):
        """Calling evaluate() before entering the context raises RuntimeError."""
        engine = StockfishEngine("fake")
        with pytest.raises(RuntimeError, match="Engine not started"):
            engine.evaluate(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            )


# ── Integration test (requires real Stockfish) ──────────────────


@pytest.mark.skipif(
    not os.environ.get("STOCKFISH_PATH"),
    reason="STOCKFISH_PATH environment variable not set",
)
class TestStockfishEngineIntegration:
    """Runs against a real Stockfish binary.

    Set the STOCKFISH_PATH environment variable to enable these tests.
    """

    def test_mate_in_one(self):
        """Stockfish finds mate in 1 from a trivial position.

        Position: White Kg1 + Qg6 vs Black Kh8 (empty board).
        Best move is Qg7# (checkmate in one move).
        """
        sf_path = os.environ["STOCKFISH_PATH"]
        fen = "7k/8/6Q1/8/8/8/8/6K1 w - - 0 1"

        with StockfishEngine(sf_path) as engine:
            result = engine.evaluate(fen, depth=10)

        # Engine must report a forced mate for White.
        assert result.mate_in is not None, "Expected a mate score"
        assert result.mate_in > 0, "White should be the one mating"
        assert result.mate_in == 1, f"Expected mate in 1, got mate in {result.mate_in}"
        assert result.best_move_uci != "", "Should have a best move"
