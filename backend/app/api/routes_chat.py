"""API routes for Ask-a-Question Chat."""

from __future__ import annotations

import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends

from app.rag.chat import ChatRequest, ChatResponse, answer_chess_question
from app.rag.retriever import HybridRetriever
from app.rag.vector_store import VectorStore
from app.rag.llm_client import LLMClientProtocol, LLMClient, FakeLLMClient
from app.core.engine import StockfishEngine
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# Cached / singleton retriever instance
_cached_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever | None:
    """FastAPI dependency for the hybrid knowledge retriever."""
    global _cached_retriever
    if _cached_retriever is not None:
        return _cached_retriever

    candidate_paths = [
        Path("data/chroma"),
        Path("../data/chroma"),
        Path("chroma_db"),
        Path("../chroma_db"),
    ]
    persist_dir = None
    for p in candidate_paths:
        if p.exists() and p.is_dir():
            persist_dir = str(p)
            break

    try:
        store = VectorStore(persist_dir=persist_dir, collection_name="chess_knowledge")
        _cached_retriever = HybridRetriever(vector_store=store)
        return _cached_retriever
    except Exception as e:
        logger.warning("Could not initialize VectorStore / HybridRetriever: %s", e)
        return None


def get_llm() -> LLMClientProtocol:
    """FastAPI dependency for the LLM client."""
    if settings.gemini_api_key:
        try:
            return LLMClient()
        except Exception as e:
            logger.warning("Failed to initialize Gemini LLM client: %s", e)
    return FakeLLMClient()


def get_engine() -> StockfishEngine | None:
    """FastAPI dependency for the Stockfish engine."""
    try:
        return StockfishEngine(settings.stockfish_path)
    except Exception:
        return None


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(
    request: ChatRequest,
    retriever: HybridRetriever | None = Depends(get_retriever),
    llm: LLMClientProtocol = Depends(get_llm),
    engine: StockfishEngine | None = Depends(get_engine),
) -> ChatResponse:
    """POST /api/chat — Ask questions from the chess knowledge base."""
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        return answer_chess_question(
            question=request.question,
            history=request.history,
            retriever=retriever,
            llm=llm,
            engine=engine,
            min_score=settings.retrieval_min_score,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Chat endpoint error: %s", e)
        raise HTTPException(status_code=500, detail=f"Chat error: {e}")
