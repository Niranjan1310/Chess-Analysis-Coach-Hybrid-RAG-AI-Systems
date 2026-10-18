"""
Position describer -- builds a textual summary of a chess position.

Used to construct RAG queries: the description is embedded and compared
against the knowledge-base chunks so that the retriever can surface
relevant strategy notes and opening theory.

The description is deterministic (no LLM involved) and covers:
  - Game phase (opening / middlegame / endgame)
  - Material balance
  - Pawn structure weaknesses (doubled, isolated, backward)
  - King safety (castling status, pawn shield)
  - Center control
"""

from __future__ import annotations

import chess
from collections import Counter
from pydantic import BaseModel


# -- Data model ----------------------------------------------------------


class PositionDescription(BaseModel):
    """Structured textual description of a position."""

    phase: str              # "opening", "middlegame", or "endgame"
    material: str           # e.g. "White: Q R R B N  Black: Q R R B B"
    material_balance: str   # e.g. "Material is equal" / "White is up a knight"
    pawn_structure: str     # doubled / isolated / backward pawns
    king_safety: str        # castling, pawn shield
    center: str             # center control description
    summary: str            # one-paragraph human-readable summary


# -- Piece values (for material balance) ---------------------------------

_PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

_PIECE_LETTERS: dict[chess.PieceType, str] = {
    chess.PAWN: "",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK: "R",
    chess.QUEEN: "Q",
    chess.KING: "K",
}

CENTER_SQUARES = {chess.E4, chess.D4, chess.E5, chess.D5}


# -- Helpers -------------------------------------------------------------


def _count_pieces(board: chess.Board) -> int:
    """Total number of non-king pieces on the board."""
    return len(board.piece_map()) - 2  # subtract 2 kings


def detect_phase(board: chess.Board) -> str:
    """Heuristic game-phase detection based on piece count.

    - Opening: >= 24 non-king pieces (the game just started)
    - Endgame: <= 10 non-king pieces (few pieces remain)
    - Middlegame: everything in between
    """
    n = _count_pieces(board)
    if n >= 24:
        return "opening"
    if n <= 10:
        return "endgame"
    return "middlegame"


def _material_string(board: chess.Board, color: chess.Color) -> str:
    """E.g. 'Q R R B N 5P'."""
    parts: list[str] = []
    for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
        count = len(board.pieces(pt, color))
        parts.extend([_PIECE_LETTERS[pt]] * count)
    pawns = len(board.pieces(chess.PAWN, color))
    if pawns > 0:
        parts.append(f"{pawns}P")
    return " ".join(parts) if parts else "(none)"


def _material_balance_text(board: chess.Board) -> str:
    """Human-readable material-balance summary."""
    white_val = sum(
        len(board.pieces(pt, chess.WHITE)) * v
        for pt, v in _PIECE_VALUES.items()
    )
    black_val = sum(
        len(board.pieces(pt, chess.BLACK)) * v
        for pt, v in _PIECE_VALUES.items()
    )
    diff = white_val - black_val
    if diff == 0:
        return "Material is equal"
    side = "White" if diff > 0 else "Black"
    adv = abs(diff)
    if adv == 1:
        return f"{side} is up a pawn"
    if adv == 2:
        return f"{side} is up two pawns"
    if adv == 3:
        return f"{side} is up a minor piece"
    if adv == 5:
        return f"{side} is up a rook"
    if adv == 4:
        return f"{side} is up the exchange and a pawn"
    if adv == 9:
        return f"{side} is up a queen"
    return f"{side} is up {adv} pawns worth of material"


def _pawn_structure(board: chess.Board) -> str:
    """Detect doubled, isolated, and backward pawns for both sides."""
    issues: list[str] = []
    for color, name in ((chess.WHITE, "White"), (chess.BLACK, "Black")):
        pawns = board.pieces(chess.PAWN, color)
        files = [chess.square_file(sq) for sq in pawns]
        file_counts = Counter(files)
        file_set = set(files)

        # Doubled pawns
        doubled = [f for f, c in file_counts.items() if c > 1]
        if doubled:
            file_names = ", ".join(chr(ord("a") + f) for f in doubled)
            issues.append(f"{name} has doubled pawns on file(s) {file_names}")

        # Isolated pawns
        for f in sorted(file_set):
            neighbors = {f - 1, f + 1} & set(range(8))
            if not (neighbors & file_set):
                fl = chr(ord("a") + f)
                issues.append(f"{name} has an isolated pawn on the {fl}-file")

    return "; ".join(issues) if issues else "No major pawn structure issues"


def _king_safety(board: chess.Board) -> str:
    """Describe king position and castling rights."""
    parts: list[str] = []

    for color, name in ((chess.WHITE, "White"), (chess.BLACK, "Black")):
        king_sq = board.king(color)
        if king_sq is None:
            parts.append(f"{name} king not on board")
            continue

        king_file = chess.square_file(king_sq)
        king_rank = chess.square_rank(king_sq)

        can_castle_k = board.has_kingside_castling_rights(color)
        can_castle_q = board.has_queenside_castling_rights(color)

        base_rank = 0 if color == chess.WHITE else 7
        castled = False
        if king_rank == base_rank and king_file in (6, 2):  # g-file or c-file
            castled = True

        if castled:
            side_name = "kingside" if king_file == 6 else "queenside"
            shield_squares = []
            direction = 1 if color == chess.WHITE else -1
            for df in (-1, 0, 1):
                f = king_file + df
                r = king_rank + direction
                if 0 <= f < 8 and 0 <= r < 8:
                    shield_squares.append(chess.square(f, r))
            shield_pawns = sum(
                1 for sq in shield_squares
                if board.piece_at(sq) == chess.Piece(chess.PAWN, color)
            )
            parts.append(
                f"{name} has castled {side_name} "
                f"({shield_pawns}/3 pawn shield)"
            )
        elif can_castle_k or can_castle_q:
            rights = []
            if can_castle_k:
                rights.append("kingside")
            if can_castle_q:
                rights.append("queenside")
            parts.append(
                f"{name} king is in the center, "
                "can still castle " + " and ".join(rights)
            )
        else:
            sq_name = chess.square_name(king_sq)
            parts.append(f"{name} king on {sq_name}, no castling rights")

    return "; ".join(parts)


def _center_control(board: chess.Board) -> str:
    """Count which side controls more central squares."""
    white_ctrl = 0
    black_ctrl = 0
    for sq in CENTER_SQUARES:
        white_attackers = len(board.attackers(chess.WHITE, sq))
        black_attackers = len(board.attackers(chess.BLACK, sq))
        piece = board.piece_at(sq)
        if piece and piece.color == chess.WHITE:
            white_attackers += 1
        elif piece and piece.color == chess.BLACK:
            black_attackers += 1
        white_ctrl += white_attackers
        black_ctrl += black_attackers

    if white_ctrl > black_ctrl + 2:
        return "White dominates the center"
    if black_ctrl > white_ctrl + 2:
        return "Black dominates the center"
    if white_ctrl > black_ctrl:
        return "White has a slight edge in center control"
    if black_ctrl > white_ctrl:
        return "Black has a slight edge in center control"
    return "Center control is roughly equal"


# -- Public API ----------------------------------------------------------


def describe_position(fen: str) -> PositionDescription:
    """Build a PositionDescription from a FEN string.

    This is the main entry point used by the RAG pipeline to create
    a text query for retrieval.

    Args:
        fen: A valid FEN string.

    Returns:
        A PositionDescription with structured text fields.
    """
    board = chess.Board(fen)

    phase = detect_phase(board)
    mat_white = _material_string(board, chess.WHITE)
    mat_black = _material_string(board, chess.BLACK)
    material = f"White: {mat_white}  Black: {mat_black}"
    balance = _material_balance_text(board)
    pawns = _pawn_structure(board)
    king = _king_safety(board)
    center = _center_control(board)

    side_to_move = "White" if board.turn == chess.WHITE else "Black"
    summary_parts = [
        f"This is a {phase} position. {side_to_move} to move.",
        balance + ".",
        pawns + ".",
        king + ".",
        center + ".",
    ]
    summary = " ".join(summary_parts)

    return PositionDescription(
        phase=phase,
        material=material,
        material_balance=balance,
        pawn_structure=pawns,
        king_safety=king,
        center=center,
        summary=summary,
    )
