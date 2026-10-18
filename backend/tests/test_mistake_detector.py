"""
Tests for app.core.mistake_detector – move classification by centipawn loss.

Covers:
  - eval_to_cp conversion (normal + mate scores)
  - classify_move for White and Black moves
  - Mate-related scenarios (missed mate, allowed mate)
  - Exact threshold boundaries
"""

import pytest

from app.core.engine import EvalResult
from app.core.mistake_detector import (
    MATE_CP,
    MoveClass,
    classify_move,
    eval_to_cp,
)


# ── Helpers to build EvalResult quickly ──────────────────────────


def cp_eval(cp: int) -> EvalResult:
    """EvalResult with a centipawn score (no mate)."""
    return EvalResult(score_cp=cp, mate_in=None, best_move_uci="a1a1", pv=[])


def mate_eval(n: int) -> EvalResult:
    """EvalResult with a mate-in-N score (positive = White mates)."""
    return EvalResult(score_cp=None, mate_in=n, best_move_uci="a1a1", pv=[])


# ── eval_to_cp tests ─────────────────────────────────────────────


class TestEvalToCp:
    """Verify the centipawn conversion logic."""

    def test_positive_centipawns(self):
        assert eval_to_cp(cp_eval(200)) == 200

    def test_negative_centipawns(self):
        assert eval_to_cp(cp_eval(-150)) == -150

    def test_zero_centipawns(self):
        assert eval_to_cp(cp_eval(0)) == 0

    def test_white_mates(self):
        # Mate in 3 for White → 10000 − 3 = 9997
        assert eval_to_cp(mate_eval(3)) == MATE_CP - 3

    def test_black_mates(self):
        # Black mates in 2 → −(10000 − 2) = −9998
        assert eval_to_cp(mate_eval(-2)) == -(MATE_CP - 2)

    def test_shorter_mate_is_bigger(self):
        # Mate in 1 should be worth more than mate in 5.
        assert eval_to_cp(mate_eval(1)) > eval_to_cp(mate_eval(5))

    def test_mate_in_one_value(self):
        assert eval_to_cp(mate_eval(1)) == 9999

    def test_mated_in_one_value(self):
        assert eval_to_cp(mate_eval(-1)) == -9999


# ── classify_move – White moves ──────────────────────────────────


class TestClassifyMoveWhite:
    """White's moves: cp_loss = before − after."""

    def test_good_move(self):
        """Small loss (< 50 cp) → GOOD."""
        assert classify_move(cp_eval(100), cp_eval(80), "white") == MoveClass.GOOD

    def test_inaccuracy(self):
        """50–99 cp loss → INACCURACY."""
        # loss = 100 − 30 = 70
        assert classify_move(cp_eval(100), cp_eval(30), "white") == MoveClass.INACCURACY

    def test_mistake(self):
        """100–299 cp loss → MISTAKE."""
        # loss = 200 − 50 = 150
        assert classify_move(cp_eval(200), cp_eval(50), "white") == MoveClass.MISTAKE

    def test_blunder(self):
        """≥ 300 cp loss → BLUNDER."""
        # loss = 100 − (−250) = 350
        assert classify_move(cp_eval(100), cp_eval(-250), "white") == MoveClass.BLUNDER

    def test_improvement_is_good(self):
        """Negative loss (position improved) → GOOD."""
        assert classify_move(cp_eval(0), cp_eval(200), "white") == MoveClass.GOOD


# ── classify_move – Black moves ──────────────────────────────────


class TestClassifyMoveBlack:
    """Black's moves: cp_loss = after − before  (flipped).

    Evals are from White's POV. Black wants the eval to go DOWN
    (more negative).  If it goes UP after Black's move, Black made
    an error.
    """

    def test_good_move(self):
        """Black maintains advantage – small loss → GOOD."""
        # before = −100, after = −80 → loss = −80 − (−100) = 20
        assert classify_move(cp_eval(-100), cp_eval(-80), "black") == MoveClass.GOOD

    def test_inaccuracy(self):
        """Black loses 50–99 cp → INACCURACY."""
        # before = −100, after = −30 → loss = −30 − (−100) = 70
        assert classify_move(cp_eval(-100), cp_eval(-30), "black") == MoveClass.INACCURACY

    def test_mistake(self):
        """Black loses 100–299 cp → MISTAKE."""
        # before = −200, after = 0 → loss = 0 − (−200) = 200
        assert classify_move(cp_eval(-200), cp_eval(0), "black") == MoveClass.MISTAKE

    def test_blunder(self):
        """Black loses ≥ 300 cp → BLUNDER."""
        # before = −100, after = +250 → loss = 250 − (−100) = 350
        assert classify_move(cp_eval(-100), cp_eval(250), "black") == MoveClass.BLUNDER

    def test_improvement_is_good(self):
        """Black improves the position → GOOD."""
        # before = 0, after = −200 → loss = −200 − 0 = −200 (negative)
        assert classify_move(cp_eval(0), cp_eval(-200), "black") == MoveClass.GOOD


# ── Mate-related classification ──────────────────────────────────


class TestClassifyMoveMate:
    """Mate scores should produce blunders when missed or allowed."""

    def test_white_misses_mate(self):
        """White had mate in 2 but played a normal move → BLUNDER.

        before = mate in 2 → 9998 cp
        after  = +100 cp
        loss   = 9998 − 100 = 9898
        """
        assert classify_move(mate_eval(2), cp_eval(100), "white") == MoveClass.BLUNDER

    def test_white_allows_mate(self):
        """White had +100 but now Black has mate → BLUNDER.

        before = +100 cp
        after  = Black mates in 3 → −9997 cp
        loss   = 100 − (−9997) = 10 097
        """
        assert classify_move(cp_eval(100), mate_eval(-3), "white") == MoveClass.BLUNDER

    def test_black_misses_mate(self):
        """Black had mate in 1 but played a normal move → BLUNDER.

        before = Black mates in 1 → −9999 cp
        after  = −100 cp
        Black's loss = −100 − (−9999) = 9899
        """
        assert classify_move(mate_eval(-1), cp_eval(-100), "black") == MoveClass.BLUNDER

    def test_black_allows_mate(self):
        """Black had −100 but now White has mate → BLUNDER.

        before = −100 cp
        after  = White mates in 2 → +9998 cp
        Black's loss = 9998 − (−100) = 10 098
        """
        assert classify_move(cp_eval(-100), mate_eval(2), "black") == MoveClass.BLUNDER

    def test_white_finds_faster_mate_is_good(self):
        """White had mate in 5, now has mate in 3 → GOOD.

        before = 9995, after = 9997 → loss = −2 (improvement)
        """
        assert classify_move(mate_eval(5), mate_eval(3), "white") == MoveClass.GOOD

    def test_black_finds_faster_mate_is_good(self):
        """Black had mate in 4, now has mate in 2 → GOOD.

        before = −9996, after = −9998
        Black's loss = −9998 − (−9996) = −2 (improvement)
        """
        assert classify_move(mate_eval(-4), mate_eval(-2), "black") == MoveClass.GOOD


# ── Boundary tests ───────────────────────────────────────────────


class TestClassifyMoveBoundaries:
    """Exact threshold boundaries (defaults: 50 / 100 / 300)."""

    def test_exactly_50_is_inaccuracy(self):
        assert classify_move(cp_eval(100), cp_eval(50), "white") == MoveClass.INACCURACY

    def test_exactly_100_is_mistake(self):
        assert classify_move(cp_eval(200), cp_eval(100), "white") == MoveClass.MISTAKE

    def test_exactly_300_is_blunder(self):
        assert classify_move(cp_eval(400), cp_eval(100), "white") == MoveClass.BLUNDER

    def test_49_is_good(self):
        assert classify_move(cp_eval(100), cp_eval(51), "white") == MoveClass.GOOD

    def test_99_is_inaccuracy(self):
        assert classify_move(cp_eval(200), cp_eval(101), "white") == MoveClass.INACCURACY

    def test_zero_loss_is_good(self):
        assert classify_move(cp_eval(100), cp_eval(100), "white") == MoveClass.GOOD
