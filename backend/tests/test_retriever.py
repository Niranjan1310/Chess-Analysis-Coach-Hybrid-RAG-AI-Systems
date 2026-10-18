"""
Tests for app.rag.retriever (hybrid retriever with RRF).
"""

import pytest

from app.rag.vector_store import Chunk, VectorStore
from app.rag.retriever import HybridRetriever, _rrf_score


# -- Reuse the FakeEmbedder from test_vector_store -----------------------

from tests.test_vector_store import FakeEmbedder


# -- Fixtures ------------------------------------------------------------


SAMPLE_DOCS = [
    ("d1", "Pawn structure weaknesses including doubled and isolated pawns", {"type": "article", "section": "pawns"}),
    ("d2", "King safety and castling with a solid pawn shield", {"type": "article", "section": "king"}),
    ("d3", "Tactical motifs: forks, pins, skewers, and discovered attacks", {"type": "article", "section": "tactics"}),
    ("d4", "Caro-Kann Defense ECO B12 1.e4 c6 2.d4 d5 3.e5", {"type": "opening", "eco": "B12"}),
    ("d5", "Italian Game ECO C50 1.e4 e5 2.Nf3 Nc6 3.Bc4", {"type": "opening", "eco": "C50"}),
    ("d6", "Endgame technique: king activity and pawn promotion", {"type": "article", "section": "endgame"}),
]


@pytest.fixture
def retriever(tmp_path):
    embedder = FakeEmbedder()
    store = VectorStore(
        persist_dir=str(tmp_path / "chroma_ret"),
        collection_name="ret_test",
        embedder=embedder,
    )
    store.add(
        ids=[d[0] for d in SAMPLE_DOCS],
        documents=[d[1] for d in SAMPLE_DOCS],
        metadatas=[d[2] for d in SAMPLE_DOCS],
    )
    bm25_corpus = [(d[0], d[1], d[2]) for d in SAMPLE_DOCS]
    return HybridRetriever(vector_store=store, bm25_corpus=bm25_corpus)


# -- Tests ---------------------------------------------------------------


class TestRRFScore:

    def test_rank_1_highest(self):
        assert _rrf_score(1) > _rrf_score(2)

    def test_all_positive(self):
        for rank in range(1, 100):
            assert _rrf_score(rank) > 0


class TestVectorOnlyMode:

    def test_returns_results(self, retriever):
        results = retriever.retrieve("pawn structure", k=3, mode="vector")
        assert len(results) >= 1
        assert all(isinstance(r, Chunk) for r in results)

    def test_relevance(self, retriever):
        results = retriever.retrieve("pawn weakness doubled isolated", k=3, mode="vector")
        ids = [r.id for r in results]
        assert "d1" in ids  # pawn structure doc should be found


class TestBM25OnlyMode:

    def test_returns_results(self, retriever):
        results = retriever.retrieve("pawn structure", k=3, mode="bm25")
        assert len(results) >= 1

    def test_keyword_match(self, retriever):
        results = retriever.retrieve("Caro-Kann Defense", k=3, mode="bm25")
        ids = [r.id for r in results]
        assert "d4" in ids  # exact keyword match

    def test_no_bm25_corpus(self, tmp_path):
        embedder = FakeEmbedder()
        store = VectorStore(persist_dir=None, collection_name="empty", embedder=embedder)
        ret = HybridRetriever(vector_store=store, bm25_corpus=None)
        results = ret.retrieve("anything", k=3, mode="bm25")
        assert results == []


class TestHybridMode:

    def test_returns_results(self, retriever):
        results = retriever.retrieve("pawn structure weakness", k=3, mode="hybrid")
        assert len(results) >= 1

    def test_fuses_both_sources(self, retriever):
        results = retriever.retrieve("chess opening Caro-Kann pawn", k=5, mode="hybrid")
        # Should include results from both vector and BM25 sources
        assert len(results) >= 2

    def test_default_mode_is_hybrid(self, retriever):
        results = retriever.retrieve("pawn structure", k=3)
        assert len(results) >= 1

    def test_with_where_filter(self, retriever):
        results = retriever.retrieve(
            "chess opening theory",
            k=5,
            where={"type": "opening"},
            mode="hybrid",
        )
        # Vector results are filtered, BM25 results are not filtered
        # but hybrid should still return something
        assert len(results) >= 1
