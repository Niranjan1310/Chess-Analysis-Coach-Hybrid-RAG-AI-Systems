"""API routes for similar master positions search."""

from fastapi import APIRouter, HTTPException
from app.schemas.similar import SimilarRequest
from app.rag.similar_positions import find_similar_positions, SimilarPositionsResult

router = APIRouter(prefix="/api")


@router.post("/similar", response_model=SimilarPositionsResult)
def search_similar(request: SimilarRequest) -> SimilarPositionsResult:
    """POST /api/similar — search for similar master positions and move statistics."""
    try:
        return find_similar_positions(fen=request.fen, k=request.k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similar position search failed: {e}")
