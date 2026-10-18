"""
Ask-a-Question Chat Coach.

Goal: Answer student chess questions directly from the knowledge base with citations.

Pipeline:
  1. question -> (optional FEN extraction)
  2. hybrid retrieval (vector + BM25, top 5-8)
  3. "I don't know" rule: if top retrieval score < retrieval_min_score, return honest refusal.
  4. Chess facts rule: cannot invent concrete variations or evals without retrieved text or engine call.
     If user specifies a FEN, Stockfish evaluates it.
  5. Memory: keeps only last 3-4 turns in prompt.
"""

from __future__ import annotations

import logging
import re
from typing import Literal
import chess
from pydantic import BaseModel

from app.rag.retriever import HybridRetriever
from app.rag.vector_store import Chunk
from app.rag.llm_client import LLMClientProtocol, FakeLLMClient
from app.core.engine import StockfishEngine, EvalResult
from app.rag.generator import _eval_to_str
from app.config import settings

logger = logging.getLogger(__name__)

# FEN regex pattern (standard 6-component FEN or 4-component partial FEN)
_FEN_PATTERN = re.compile(
    r'([rnbqkpRNBQKP1-8]+/){7}[rnbqkpRNBQKP1-8]+\s+[wb]\s+[-kKqQ]{1,4}\s+[-a-h1-8]{1,2}(\s+\d+\s+\d+)?'
)


class ChatMessage(BaseModel):
    """Single chat turn."""
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """POST /api/chat request body."""
    question: str
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    """POST /api/chat response."""
    answer: str
    sources: list[str]
    confident: bool
    engine_eval: str | None = None


SYSTEM_CHAT_PROMPT = (
    "You are a knowledgeable chess coach answering student questions. "
    "Use ONLY the retrieved knowledge documents and any engine evaluations provided below. "
    "Do NOT state concrete variations or evaluations that do not appear in the provided text or engine evaluation. "
    "If the provided documents are insufficient or irrelevant, state clearly that you do not have enough "
    "information in your knowledge base. Always cite chunk IDs using [chunk-id] when making claims."
)


def extract_fen(text: str) -> str | None:
    """Extract a valid FEN string from text if present."""
    match = _FEN_PATTERN.search(text)
    if match:
        candidate = match.group(0).strip()
        try:
            chess.Board(candidate)
            return candidate
        except Exception:
            pass
    return None


def answer_chess_question(
    question: str,
    history: list[ChatMessage] | None = None,
    *,
    retriever: HybridRetriever | None = None,
    llm: LLMClientProtocol | None = None,
    engine: StockfishEngine | None = None,
    min_score: float | None = None,
    top_k: int = 5,
) -> ChatResponse:
    """Answer a user question from the chess knowledge base with RAG.

    Args:
        question: user's chess question.
        history: recent conversation turns.
        retriever: hybrid retriever.
        llm: LLM client (defaults to FakeLLMClient).
        engine: optional Stockfish engine if position analysis is needed.
        min_score: threshold for the "I don't know" rule.
        top_k: number of chunks to retrieve (5-8).

    Returns:
        A ``ChatResponse``.
    """
    clean_q = question.strip()
    if not clean_q:
        raise ValueError("Question cannot be empty.")

    if min_score is None:
        min_score = getattr(settings, "retrieval_min_score", 0.3)
    if llm is None:
        llm = FakeLLMClient()

    # Step 1: Check for FEN in question (Chess facts rule)
    fen = extract_fen(clean_q)
    engine_eval_str = None
    if fen and engine is not None:
        try:
            ev = engine.evaluate(fen, depth=14)
            engine_eval_str = (
                f"Position (FEN): {fen}\n"
                f"Best move: {ev.best_move_uci}\n"
                f"Eval: {_eval_to_str(ev)}\n"
                f"Line: {' '.join(ev.pv)}"
            )
        except Exception as e:
            logger.warning("Engine evaluation for FEN failed: %s", e)

    # Step 2: Retrieve relevant chunks
    chunks: list[Chunk] = []
    if retriever is not None:
        try:
            chunks = retriever.retrieve(query=clean_q, k=top_k)
        except Exception as e:
            logger.warning("Retrieval failed: %s", e)

    # Step 3: "I don't know" rule
    # If no chunks or top chunk score is below threshold, and no FEN engine eval
    is_low_score = not chunks or (chunks[0].score is not None and chunks[0].score < min_score)

    if is_low_score and not engine_eval_str:
        return ChatResponse(
            answer="I don't have enough information in my chess knowledge base to answer that accurately.",
            sources=[],
            confident=False,
            engine_eval=None,
        )

    # Step 4: Memory — keep only last 3-4 turns
    recent_history = history[-4:] if history else []
    hist_lines = []
    for msg in recent_history:
        prefix = "Student" if msg.role == "user" else "Coach"
        hist_lines.append(f"{prefix}: {msg.content}")
    history_block = "\n".join(hist_lines) if hist_lines else "(No prior turns)"

    # Step 5: Format context chunks
    ctx_parts = []
    for c in chunks:
        ctx_parts.append(f"[{c.id}]: {c.text}")
    context_block = "\n\n".join(ctx_parts) if ctx_parts else "(No documents retrieved)"

    engine_block = engine_eval_str if engine_eval_str else "(No specific position evaluation requested)"

    user_prompt = f"""## Conversation history
{history_block}

## Knowledge base documents
{context_block}

## Engine evaluation
{engine_block}

## Student question
{clean_q}
"""

    try:
        answer = llm.generate(SYSTEM_CHAT_PROMPT, user_prompt)
    except Exception as e:
        logger.error("LLM chat generation failed: %s", e)
        answer = "I encountered an error generating an answer. Please try again."

    source_ids = [c.id for c in chunks] if not is_low_score else []

    return ChatResponse(
        answer=answer,
        sources=source_ids,
        confident=True,
        engine_eval=engine_eval_str,
    )
