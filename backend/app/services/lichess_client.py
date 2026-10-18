"""Lichess public API client for importing user games."""

from __future__ import annotations

import io
import logging
import re
from typing import Any
import httpx
import chess.pgn

logger = logging.getLogger(__name__)

LICHESS_GAMES_URL = "https://lichess.org/api/games/user/{username}"


def split_pgn_stream(text: str) -> list[str]:
    """Split a multi-game PGN stream into individual PGN strings."""
    if not text or not text.strip():
        return []

    stream = io.StringIO(text)
    games: list[str] = []

    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        # Export game back to string
        exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
        game_str = game.accept(exporter)
        if game_str.strip():
            games.append(game_str.strip())

    return games


class LichessClient:
    """Client for fetching games from Lichess public API."""

    def __init__(self, client: httpx.Client | None = None, timeout: float = 15.0) -> None:
        self._client = client
        self._timeout = timeout

    def fetch_user_games(self, username: str, max_games: int = 10) -> list[str]:
        """Fetch the most recent games for a given Lichess user.

        Args:
            username: Lichess handle.
            max_games: Maximum number of games to retrieve.

        Returns:
            List of PGN strings.

        Raises:
            ValueError: if the username does not exist.
            RuntimeError: if rate limited or if request fails.
        """
        clean_user = username.strip()
        if not clean_user:
            raise ValueError("Username cannot be empty.")

        url = LICHESS_GAMES_URL.format(username=clean_user)
        params = {
            "max": max_games,
            "pgnInBody": "true",
            "clocks": "false",
            "evals": "false",
            "opening": "true",
        }
        headers = {
            "Accept": "application/x-chess-pgn",
            "User-Agent": "ChessAnalysisCoach/0.1.0",
        }

        manage_client = self._client is None
        http = httpx.Client(timeout=self._timeout) if manage_client else self._client

        try:
            resp = http.get(url, params=params, headers=headers)
            if resp.status_code == 404:
                raise ValueError(f"Lichess user '{clean_user}' not found.")
            if resp.status_code == 429:
                raise RuntimeError("Lichess API rate limit exceeded. Please wait a moment.")
            resp.raise_for_status()

            return split_pgn_stream(resp.text)

        except httpx.HTTPStatusError as e:
            logger.error("Lichess HTTP error: %s", e)
            raise RuntimeError(f"Lichess API returned error {e.response.status_code}") from e
        except httpx.RequestError as e:
            logger.error("Lichess request error: %s", e)
            raise RuntimeError(f"Lichess connection failed: {e}") from e
        finally:
            if manage_client:
                http.close()
