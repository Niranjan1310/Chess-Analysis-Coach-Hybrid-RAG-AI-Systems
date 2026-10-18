"""Tests for services/lichess_client.py."""

import httpx
import pytest

from app.services.lichess_client import LichessClient, split_pgn_stream


SAMPLE_STREAM = """[Event "Rated Blitz game"]
[Site "https://lichess.org/abc12345"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0

[Event "Rated Classical game"]
[Site "https://lichess.org/xyz67890"]
[White "Player2"]
[Black "Player1"]
[Result "0-1"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 0-1
"""


def test_split_pgn_stream():
    games = split_pgn_stream(SAMPLE_STREAM)
    assert len(games) == 2
    assert "Player1" in games[0]
    assert "Player2" in games[1]


def test_split_empty_stream():
    assert split_pgn_stream("") == []
    assert split_pgn_stream("   \n  ") == []


def test_fetch_user_games_success():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert "magnuscarlsen" in request.url.path
        return httpx.Response(200, text=SAMPLE_STREAM)

    mock_client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    client = LichessClient(client=mock_client)
    games = client.fetch_user_games("magnuscarlsen", max_games=2)

    assert len(games) == 2
    assert "Player1" in games[0]


def test_fetch_user_games_user_not_found():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="User not found")

    mock_client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    client = LichessClient(client=mock_client)

    with pytest.raises(ValueError, match="not found"):
        client.fetch_user_games("nonexistent_user_999")


def test_fetch_user_games_rate_limit():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="Rate limit exceeded")

    mock_client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    client = LichessClient(client=mock_client)

    with pytest.raises(RuntimeError, match="rate limit"):
        client.fetch_user_games("some_user")


def test_fetch_user_games_empty_username():
    client = LichessClient()
    with pytest.raises(ValueError, match="empty"):
        client.fetch_user_games("   ")
