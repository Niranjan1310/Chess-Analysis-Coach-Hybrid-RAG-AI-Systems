"""
FastAPI application entry point.

Serves the API routes and mounts the static frontend.
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes_analysis import router as analysis_router
from app.api.routes_weakness import router as weakness_router
from app.api.routes_games import router as games_router
from app.api.routes_similar import router as similar_router
from app.api.routes_chat import router as chat_router

app = FastAPI(
    title="Chess Analysis Coach",
    description=(
        "Upload a PGN, find mistakes with Stockfish, "
        "and get AI-powered explanations."
    ),
    version="0.1.0",
)


@app.get("/api/health")
async def health_check():
    """Return 200 OK if the server is running."""
    return {"status": "ok"}


app.include_router(analysis_router)
app.include_router(weakness_router)
app.include_router(games_router)
app.include_router(similar_router)
app.include_router(chat_router)

# Mount frontend static files (index.html, style.css, app.js)
frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
