"""
Shared pytest fixtures for the chess-analysis-coach test suite.
"""

import pytest


# A short, well-known game (Scholar's Mate) with full headers.
# 7 half-moves: 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6?? 4.Qxf7#
SCHOLARS_MATE_PGN = """\
[Event "Test Game"]
[Site "?"]
[Date "2024.01.01"]
[White "Alice"]
[Black "Bob"]
[Result "1-0"]
[ECO "C20"]
[Opening "King's Pawn Game"]

1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0
"""


@pytest.fixture
def scholars_mate_pgn() -> str:
    """Return a PGN string for Scholar's Mate."""
    return SCHOLARS_MATE_PGN
