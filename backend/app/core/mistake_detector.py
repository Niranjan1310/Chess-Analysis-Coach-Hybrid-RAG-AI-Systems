"""
Move classifier – labels each move as good / inaccuracy / mistake / blunder
based on the centipawn loss between the evaluation before and after the move.

All evaluations must be from **White's perspective** (as returned by
``StockfishEngine.evaluate``).  This module flips the sign internally
when computing centipawn loss for Black moves.

Mate scores are converted to large centipawn values so that missing or
allowing a forced mate is always classified as a blunder.
"""

from enum import Enum

from app.core.engine import EvalResult

# A mate-in-N is converted to ±(MATE_CP − N) centipawns.
# This keeps shorter mates worth more and guarantees that any
# transition to/from a mate score blows past the blunder threshold.
MATE_CP = 10_000


class MoveClass(str, Enum):
    """Human-readable move classification."""

    GOOD = "good"
    INACCURACY = "inaccuracy"
    MISTAKE = "mistake"
    BLUNDER = "blunder"


# ── Helpers ──────────────────────────────────────────────────────


def eval_to_cp(eval_result: EvalResult) -> int:
    """Convert an ``EvalResult`` to a single centipawn integer.

    Normal scores are returned as-is.  Mate scores become large values:

    * White mates in N → ``+(MATE_CP − N)``   (e.g. mate in 1 → +9 999)
    * Black mates in N → ``−(MATE_CP − N)``   (e.g. mated in 2 → −9 998)
    """
    if eval_result.mate_in is not None:
        n = abs(eval_result.mate_in)
        value = MATE_CP - n
        # Positive mate_in = White mates, negative = Black mates.
        return value if eval_result.mate_in >= 0 else -value

    if eval_result.score_cp is not None:
        return eval_result.score_cp

    # Fallback (shouldn't happen with well-formed engine output).
    return 0


# ── Public API ───────────────────────────────────────────────────


def classify_move(
    eval_before: EvalResult,
    eval_after: EvalResult,
    side: str,
    *,
    inaccuracy_cp: int = 50,
    mistake_cp: int = 100,
    blunder_cp: int = 300,
) -> MoveClass:
    """Classify a move by its centipawn loss.

    Both ``eval_before`` and ``eval_after`` are from White's
    perspective.  The sign is flipped internally for Black moves so
    that centipawn *loss* is always a positive number when the moving
    side made the position worse for themselves.

    Args:
        eval_before:   Eval of the position **before** the move (White's POV).
        eval_after:    Eval of the position **after** the move (White's POV).
        side:          ``"white"`` or ``"black"`` – who made the move.
        inaccuracy_cp: CP-loss threshold for an inaccuracy  (default 50).
        mistake_cp:    CP-loss threshold for a mistake       (default 100).
        blunder_cp:    CP-loss threshold for a blunder       (default 300).

    Returns:
        A ``MoveClass`` enum value.
    """
    before_cp = eval_to_cp(eval_before)
    after_cp = eval_to_cp(eval_after)

    # CP loss from the moving side's perspective.
    # ─ White wants a HIGH eval → loss = before − after.
    # ─ Black wants a LOW  eval → loss = after − before (flipped).
    if side == "white":
        cp_loss = before_cp - after_cp
    else:
        cp_loss = after_cp - before_cp

    # Negative loss means the position improved – always "good".
    if cp_loss < 0:
        return MoveClass.GOOD

    # Compare against thresholds, largest first.
    if cp_loss >= blunder_cp:
        return MoveClass.BLUNDER
    if cp_loss >= mistake_cp:
        return MoveClass.MISTAKE
    if cp_loss >= inaccuracy_cp:
        return MoveClass.INACCURACY

    return MoveClass.GOOD
