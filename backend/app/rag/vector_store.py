"""
Chroma-backed vector store for chess knowledge chunks.

Each document is stored with:
  - ``id``: unique chunk identifier
  - ``document``: the text content
  - ``metadata``: dict with keys like ``source``, ``type``, ``eco``,
    ``phase``, ``rating_band``
  - ``embedding``: pre-computed dense vector

Provides ``add``, ``query``, and ``count`` operations.
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.rag.embeddings import Embedder

logger = logging.getLogger(__name__)

# Pydantic model for returned chunks
from pydantic import BaseModel


class Chunk(BaseModel):
    """A single retrieved knowledge chunk."""

    id: str
    text: str
    metadata: dict[str, Any]
    score: float = 0.0


class VectorStore:
    """Chroma vector store wrapper.

    Args:
        persist_dir: directory for Chroma persistence.
            Pass ``None`` for a pure in-memory store (useful in tests).
        collection_name: Chroma collection to use.
        embedder: an ``Embedder`` instance (injected for testability).
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str = "chess_knowledge",
        embedder: Embedder | None = None,
    ) -> None:
        self._embedder = embedder or Embedder()

        if persist_dir is None:
            self._client = chromadb.Client()
        else:
            self._client = chromadb.PersistentClient(path=persist_dir)

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # -- write -----------------------------------------------------------

    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Add documents with pre-computed embeddings.

        Args:
            ids: unique identifiers for each document.
            documents: text content of each chunk.
            metadatas: metadata dicts for each chunk.
        """
        embeddings = self._embedder.embed_texts(documents)
        self._collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info("Added %d documents to collection.", len(ids))

    # -- read ------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Semantic search over the collection.

        Args:
            query_text: natural-language query.
            k: number of results to return.
            where: optional Chroma metadata filter.

        Returns:
            List of ``Chunk`` objects sorted by relevance.
        """
        query_embedding = self._embedder.embed_query(query_text)

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(k, self.count() or k),
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        chunks: list[Chunk] = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                chunks.append(Chunk(
                    id=doc_id,
                    text=results["documents"][0][i],
                    metadata=results["metadatas"][0][i],
                    score=1 - (results["distances"][0][i] if results["distances"] else 0),
                ))

        return chunks

    # -- utils -----------------------------------------------------------

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()
