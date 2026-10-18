"""
Tests for app.rag.vector_store (Chroma wrapper) and app.rag.embeddings.

Uses a temporary directory for Chroma persistence to avoid polluting
the real data directory.
"""

import tempfile

import pytest

from app.rag.embeddings import Embedder
from app.rag.vector_store import Chunk, VectorStore


# -- Fixtures ------------------------------------------------------------


class FakeEmbedder(Embedder):
    """Deterministic embedder for fast, reproducible tests.

    Maps each text to a fixed-length vector based on character counts.
    This avoids downloading a real model during CI/tests.
    """

    def __init__(self) -> None:
        super().__init__(model_name="fake")
        self._dim = 32

    def _load(self):
        return None  # never load a real model

    def _text_to_vec(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for i, ch in enumerate(text.lower()):
            vec[i % self._dim] += ord(ch) / 1000.0
        # Normalize
        norm = max(sum(v ** 2 for v in vec) ** 0.5, 1e-9)
        return [v / norm for v in vec]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._text_to_vec(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._text_to_vec(query)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def store(fake_embedder, tmp_path):
    """VectorStore backed by a temp directory with a fake embedder."""
    return VectorStore(
        persist_dir=str(tmp_path / "chroma_test"),
        collection_name="test_collection",
        embedder=fake_embedder,
    )


# -- Tests ---------------------------------------------------------------


class TestVectorStoreBasics:

    def test_empty_store_count(self, store):
        assert store.count() == 0

    def test_add_and_count(self, store):
        store.add(
            ids=["doc1", "doc2"],
            documents=["chess opening theory", "endgame technique"],
            metadatas=[{"type": "article"}, {"type": "article"}],
        )
        assert store.count() == 2

    def test_add_and_query(self, store):
        store.add(
            ids=["pawn_structure", "king_safety", "tactics"],
            documents=[
                "Doubled pawns are a structural weakness in chess positions",
                "King safety requires a solid pawn shield after castling",
                "Tactical motifs include forks, pins, and skewers",
            ],
            metadatas=[
                {"type": "article", "section": "pawns"},
                {"type": "article", "section": "king"},
                {"type": "article", "section": "tactics"},
            ],
        )
        results = store.query("pawn weakness", k=3)
        assert len(results) >= 1
        assert all(isinstance(r, Chunk) for r in results)
        # All three documents should be returned when k=3
        result_ids = {r.id for r in results}
        assert result_ids == {"pawn_structure", "king_safety", "tactics"}

    def test_query_with_where_filter(self, store):
        store.add(
            ids=["open1", "art1"],
            documents=[
                "Sicilian Defense 1.e4 c5",
                "Strategy notes about pawn play",
            ],
            metadatas=[
                {"type": "opening", "eco": "B20"},
                {"type": "article", "section": "pawns"},
            ],
        )
        results = store.query("chess strategy", k=5, where={"type": "opening"})
        assert all(r.metadata["type"] == "opening" for r in results)


class TestInMemoryStore:
    """Test that persist_dir=None gives a working in-memory store."""

    def test_in_memory(self, fake_embedder):
        store = VectorStore(
            persist_dir=None,
            collection_name="mem_test",
            embedder=fake_embedder,
        )
        store.add(
            ids=["m1"],
            documents=["test document"],
            metadatas=[{"type": "test"}],
        )
        assert store.count() == 1
        results = store.query("test", k=1)
        assert len(results) == 1
        assert results[0].id == "m1"
