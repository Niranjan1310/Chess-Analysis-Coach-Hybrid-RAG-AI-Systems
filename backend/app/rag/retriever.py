"""
Hybrid retriever -- combines dense vector search (Chroma) with sparse
keyword search (BM25) using Reciprocal Rank Fusion (RRF).

Supports three modes, selectable via the ``mode`` parameter:
  - ``"hybrid"`` (default): fuse vector + BM25 results with RRF.
  - ``"vector"``          : vector-only search.
  - ``"bm25"``            : BM25-only search.

The ``mode`` parameter is useful for evaluation: you can compare
hybrid vs vector-only vs BM25-only retrieval quality.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Literal

from rank_bm25 import BM25Okapi

from app.rag.vector_store import Chunk, VectorStore

logger = logging.getLogger(__name__)

# RRF constant (standard value from the original paper)
RRF_K = 60

STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "for", "of", "to", "and", "or",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "what", "who", "where", "how", "why", "which",
    "can", "could", "would", "should", "you", "your", "my", "i", "we", "they",
    "this", "that", "it", "its", "with", "from", "as", "by", "about", "best"
}


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words, stripping punctuation and common stopwords."""
    words = re.findall(r"[a-zA-Z0-9_\-\.]+", text.lower())
    filtered = [w for w in words if w not in STOPWORDS]
    return filtered if filtered else words


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    """Reciprocal Rank Fusion score for a single rank (1-based)."""
    return 1.0 / (k + rank)


class HybridRetriever:
    """Hybrid retriever with RRF fusion.

    Args:
        vector_store: the Chroma-backed ``VectorStore``.
        bm25_corpus: list of (id, text) tuples for BM25 indexing.
            If empty, BM25 search returns no results.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_corpus: list[tuple[str, str, dict[str, Any]]] | None = None,
    ) -> None:
        self._vs = vector_store
        # BM25 index
        self._bm25_ids: list[str] = []
        self._bm25_texts: list[str] = []
        self._bm25_metas: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None

        if bm25_corpus:
            self._build_bm25(bm25_corpus)

    def _build_bm25(
        self, corpus: list[tuple[str, str, dict[str, Any]]]
    ) -> None:
        """Build the BM25 index from a list of (id, text, meta) tuples."""
        self._bm25_ids = [c[0] for c in corpus]
        self._bm25_texts = [c[1] for c in corpus]
        self._bm25_metas = [c[2] for c in corpus]
        tokenized = [_tokenize(text) for text in self._bm25_texts]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built with %d documents.", len(corpus))

    def retrieve(
        self,
        query: str,
        k: int = 5,
        where: dict[str, Any] | None = None,
        mode: Literal["hybrid", "vector", "bm25"] = "hybrid",
    ) -> list[Chunk]:
        """Retrieve top-k relevant chunks.

        Args:
            query: natural-language query string.
            k: number of results to return.
            where: optional Chroma metadata filter (vector mode only).
            mode: ``"hybrid"``, ``"vector"``, or ``"bm25"``.

        Returns:
            List of ``Chunk`` objects, best first.
        """
        if mode == "vector":
            return self._vector_search(query, k, where)
        if mode == "bm25":
            return self._bm25_search(query, k)
        # hybrid
        return self._hybrid_search(query, k, where)

    # -- vector only -----------------------------------------------------

    def _vector_search(
        self, query: str, k: int, where: dict[str, Any] | None
    ) -> list[Chunk]:
        return self._vs.query(query_text=query, k=k, where=where)

    # -- BM25 only -------------------------------------------------------

    def _bm25_search(self, query: str, k: int) -> list[Chunk]:
        if self._bm25 is None:
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Get top-k indices sorted by score descending
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results: list[Chunk] = []
        for idx in ranked:
            if scores[idx] <= 0.1:
                continue
            results.append(Chunk(
                id=self._bm25_ids[idx],
                text=self._bm25_texts[idx],
                metadata=self._bm25_metas[idx],
                score=float(scores[idx]),
            ))
        return results

    # -- hybrid (RRF) ----------------------------------------------------

    def _hybrid_search(
        self, query: str, k: int, where: dict[str, Any] | None
    ) -> list[Chunk]:
        # Fetch more candidates from each source, then fuse
        fetch_k = min(k * 3, 50)

        vec_results = self._vector_search(query, fetch_k, where)
        bm25_results = self._bm25_search(query, fetch_k)

        # Build RRF scores keyed by chunk id
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for rank, chunk in enumerate(vec_results, start=1):
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0) + _rrf_score(rank)
            chunk_map[chunk.id] = chunk

        for rank, chunk in enumerate(bm25_results, start=1):
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0) + _rrf_score(rank)
            if chunk.id not in chunk_map:
                chunk_map[chunk.id] = chunk

        # Sort by fused RRF score
        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:k]
        max_rrf = 2.0 / (RRF_K + 1)
        # If BM25 found zero keyword matches in the corpus, discount out-of-domain queries
        bm25_multiplier = 0.25 if not bm25_results else 1.0

        return [
            Chunk(
                id=cid,
                text=chunk_map[cid].text,
                metadata=chunk_map[cid].metadata,
                score=round(min(1.0, (rrf_scores[cid] / max_rrf) * bm25_multiplier), 4),
            )
            for cid in sorted_ids
        ]
