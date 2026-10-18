"""
Embedding wrapper -- thin layer around sentence-transformers.

Provides a single ``Embedder`` class that lazily loads the model
on first use and exposes ``embed_texts()`` and ``embed_query()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class Embedder:
    """Lazy-loading sentence-transformer embedding client.

    The model is loaded on the first call to ``embed_texts`` or
    ``embed_query`` so that import-time is fast and tests that
    mock the embedder never trigger a real download.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents.

        Args:
            texts: list of document strings.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        model = self._load()
        embeddings = model.encode(texts, show_progress_bar=False)
        return [e.tolist() for e in embeddings]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        Args:
            query: the search query.

        Returns:
            A single embedding vector.
        """
        model = self._load()
        return model.encode(query, show_progress_bar=False).tolist()
