"""
PGN parser – turns a PGN string into a list of per-half-move records.

Each record (``PositionInfo``) contains the FEN before and after the
move, the move in SAN and UCI notation, the move number, whose turn
it was, and the key game headers.
"""

import io

import chess
import chess.pgn
from pydantic import BaseModel


# ── Data models ──────────────────────────────────────────────────


class GameHeaders(BaseModel):
    """Key metadata extracted from PGN headers."""

    white: str = "?"
    black: str = "?"
    result: str = "*"
    eco: str | None = None
    opening: str | None = None


class PositionInfo(BaseModel):
    """One half-move in a game, with full context."""

    fen_before: str       # position before the move was played
    fen_after: str        # position after the move was played
    move_san: str         # e.g. "Nf3"
    move_uci: str         # e.g. "g1f3"
    move_number: int      # full-move counter (1, 2, 3, …)
    side: str             # "white" or "black"
    headers: GameHeaders  # game-level metadata (shared by all moves)


# ── Public API ───────────────────────────────────────────────────


def parse_pgn(pgn_string: str) -> list[PositionInfo]:
    """Parse a PGN string and return one ``PositionInfo`` per half-move.

    Args:
        pgn_string: A complete PGN game (headers + movetext).

    Returns:
        A list of ``PositionInfo`` in game order.

    Raises:
        ValueError: If the PGN cannot be parsed at all.
    """
    game = chess.pgn.read_game(io.StringIO(pgn_string))
    if game is None:
        raise ValueError("Could not parse PGN string.")

    # ── Extract headers ──────────────────────────────────────────
    headers = GameHeaders(
        white=game.headers.get("White", "?"),
        black=game.headers.get("Black", "?"),
        result=game.headers.get("Result", "*"),
        eco=game.headers.get("ECO"),
        opening=game.headers.get("Opening"),
    )

    # ── Walk through mainline moves ──────────────────────────────
    positions: list[PositionInfo] = []
    board = game.board()  # starting position (usually standard)

    for move in game.mainline_moves():
        fen_before = board.fen()
        move_san = board.san(move)       # human-readable notation
        move_uci = move.uci()            # machine-friendly notation
        move_number = board.fullmove_number
        side = "white" if board.turn == chess.WHITE else "black"

        # Apply the move to get the resulting position.
        board.push(move)
        fen_after = board.fen()

        positions.append(
            PositionInfo(
                fen_before=fen_before,
                fen_after=fen_after,
                move_san=move_san,
                move_uci=move_uci,
                move_number=move_number,
                side=side,
                headers=headers,
            )
        )

    return positions
