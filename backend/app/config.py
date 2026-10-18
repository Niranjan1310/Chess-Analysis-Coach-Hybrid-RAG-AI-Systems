"""
Application settings, loaded from environment variables / .env file.

Uses pydantic-settings so you can override any value by setting an
environment variable or adding a line to a ``.env`` file in the
working directory.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All app-level configuration.  Reads from .env automatically."""

    # ── Stockfish ────────────────────────────────────────────────
    # Full path to the Stockfish binary.
    stockfish_path: str = "stockfish"
    # Default search depth (higher = stronger but slower).
    stockfish_depth: int = 14

    # ── Move-classification thresholds (centipawns) ──────────────
    inaccuracy_threshold: int = 50
    mistake_threshold: int = 100
    blunder_threshold: int = 300

    # -- Gemini LLM -------------------------------------------------------
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # -- RAG retrieval ----------------------------------------------------
    retrieval_min_score: float = 0.3

    # -- Database (SQLite) ------------------------------------------------
    database_url: str = "sqlite:///./chessmind.db"
    min_games_for_weakness: int = 5
    min_moves_for_category: int = 10

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton instance – import this wherever you need settings.
settings = Settings()
