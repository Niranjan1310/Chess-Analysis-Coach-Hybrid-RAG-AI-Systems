"""
Analysis API routes.

POST /api/analyze  — submit a PGN for analysis (returns job_id immediately).
GET  /api/analyze/{job_id} — check job status and retrieve results.

Jobs run as FastAPI BackgroundTasks. Results are stored in an
in-memory dict (sufficient for single-instance use).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks

from app.schemas.analysis import (
    AnalyzeRequest,
    JobCreated,
    JobStatus,
    AnalyzeResult,
)
from app.services.analysis_service import analyze_game

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# In-memory job store: job_id -> JobStatus
_jobs: dict[str, JobStatus] = {}


def _run_analysis(job_id: str, pgn: str, depth: int) -> None:
    """Background task that runs the full analysis pipeline."""
    _jobs[job_id] = JobStatus(job_id=job_id, status="running")
    try:
        result = analyze_game(pgn, depth=depth)
        _jobs[job_id] = JobStatus(
            job_id=job_id, status="completed", result=result,
        )
        logger.info("Job %s completed: %d flagged moves.", job_id, result.flagged_moves)
    except Exception as e:
        logger.exception("Job %s failed: %s", job_id, e)
        _jobs[job_id] = JobStatus(
            job_id=job_id, status="failed", error=str(e),
        )


@router.post("/analyze", response_model=JobCreated)
async def start_analysis(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
) -> JobCreated:
    """Submit a PGN for analysis.

    Returns a ``job_id`` immediately.  Poll ``GET /api/analyze/{job_id}``
    for status and results.
    """
    job_id = str(uuid.uuid4())
    _jobs[job_id] = JobStatus(job_id=job_id, status="pending")
    background_tasks.add_task(_run_analysis, job_id, request.pgn, request.depth)
    logger.info("Job %s created.", job_id)
    return JobCreated(job_id=job_id)


@router.get("/analyze/{job_id}", response_model=JobStatus)
async def get_analysis_status(job_id: str) -> JobStatus:
    """Check the status of an analysis job."""
    if job_id not in _jobs:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return _jobs[job_id]


def clear_jobs() -> None:
    """Clear all jobs (used in tests)."""
    _jobs.clear()
