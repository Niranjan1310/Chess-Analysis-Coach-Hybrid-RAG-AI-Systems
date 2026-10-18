"""
Pydantic schemas for the /api/analyze endpoints.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    """POST /api/analyze request body."""
    pgn: str
    depth: int = 14


class MoveExplanation(BaseModel):
    """One flagged move with its explanation."""
    move_number: int
    side: str
    played: str
    best: str
    classification: str       # "inaccuracy", "mistake", "blunder"
    eval_before: str
    eval_after: str
    explanation: str
    sources: list[str]
    faithful: bool
    engine_only: bool


class AnalyzeResult(BaseModel):
    """Full analysis result for a game."""
    total_moves: int
    flagged_moves: int
    explanations: list[MoveExplanation]


class JobStatus(BaseModel):
    """GET /api/analyze/{job_id} response."""
    job_id: str
    status: Literal["pending", "running", "completed", "failed"]
    result: AnalyzeResult | None = None
    error: str | None = None


class JobCreated(BaseModel):
    """POST /api/analyze response (immediate)."""
    job_id: str
    status: str = "pending"
