"""
Stockfish wrapper using python-chess's UCI protocol.

All evaluation scores are stored from **White's perspective**:
  - Positive score → White is better.
  - Negative score → Black is better.

Usage::

    with StockfishEngine("/path/to/stockfish") as engine:
        result = engine.evaluate(fen, depth=14)
        print(result.score_cp, result.best_move_uci)
"""

import chess
import chess.engine
from pydantic import BaseModel


# ── Data model ───────────────────────────────────────────────────


class EvalResult(BaseModel):
    """Engine evaluation of a single position (always from White's POV).

    Exactly one of ``score_cp`` or ``mate_in`` will be non-None:

    * ``score_cp`` – evaluation in centipawns.
    * ``mate_in`` – moves until mate.
      Positive means White delivers mate; negative means Black does.
    """

    score_cp: int | None = None
    mate_in: int | None = None
    best_move_uci: str
    pv: list[str]  # principal variation (list of UCI move strings)


# ── Engine wrapper ───────────────────────────────────────────────


class StockfishEngine:
    """Context-manager wrapper around Stockfish.

    The engine process is started on ``__enter__`` and guaranteed to
    be closed on ``__exit__``, even if an exception occurs.
    """

    def __init__(self, path: str) -> None:
        """Store the path to the Stockfish binary.  The engine is NOT
        started until you enter the ``with`` block."""
        self._path = path
        self._engine: chess.engine.SimpleEngine | None = None

    # ── Context manager ──────────────────────────────────────────

    def __enter__(self) -> "StockfishEngine":
        """Start the Stockfish process."""
        self._engine = chess.engine.SimpleEngine.popen_uci(self._path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Shut down the engine process."""
        if self._engine is not None:
            self._engine.quit()
            self._engine = None
        return False  # do not suppress exceptions

    # ── Public API ───────────────────────────────────────────────

    def evaluate(self, fen: str, depth: int = 14) -> EvalResult:
        """Evaluate a position at the given depth.

        Args:
            fen:   FEN string of the position to evaluate.
            depth: Search depth (higher = stronger but slower).

        Returns:
            An ``EvalResult`` with the score from White's perspective.

        Raises:
            RuntimeError: If the engine has not been started via the
                          context manager.
        """
        if self._engine is None:
            raise RuntimeError(
                "Engine not started. Use 'with StockfishEngine(...) as engine:'"
            )

        board = chess.Board(fen)
        info = self._engine.analyse(board, chess.engine.Limit(depth=depth))

        # .white() gives the score from White's perspective regardless
        # of whose turn it is on the board.
        score = info["score"].white()
        pv_moves: list[chess.Move] = info.get("pv", [])

        # Separate centipawn score and mate score.
        score_cp: int | None = None
        mate_in: int | None = None

        if score.is_mate():
            mate_in = score.mate()  # +N → White mates in N, −N → Black mates
        else:
            score_cp = score.score()

        best_move_uci = pv_moves[0].uci() if pv_moves else ""
        pv = [m.uci() for m in pv_moves]

        return EvalResult(
            score_cp=score_cp,
            mate_in=mate_in,
            best_move_uci=best_move_uci,
            pv=pv,
        )
