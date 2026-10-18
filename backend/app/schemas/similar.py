"""Pydantic schemas for /api/similar endpoint."""

from __future__ import annotations
from pydantic import BaseModel
from app.rag.similar_positions import SimilarPositionsResult


class SimilarRequest(BaseModel):
    """POST /api/similar request body."""
    fen: str
    k: int = 20
