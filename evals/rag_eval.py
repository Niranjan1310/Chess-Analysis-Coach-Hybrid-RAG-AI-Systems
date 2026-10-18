"""
Comprehensive RAG and Pipeline Evaluation Suite.

Metrics & Evaluations:
  1. Retrieval Ablations: Recall@k, MRR, nDCG@k across modes (Dense Baseline vs Sparse BM25 vs Hybrid RRF).
  2. Latency Benchmarks: p50 and p95 retrieval latency in milliseconds.
  3. Out-of-Scope Refusal Guardrail: Classification Precision, Recall, and F1 score.
  4. Move Mistake Classification: Precision, Recall, and F1 across centipawn mistake categories.
  5. Generation & Faithfulness Guardrail: Move hallucination detection accuracy with python-chess.

Usage::
    python -m evals.rag_eval
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
import statistics
import time
from typing import Any

from app.core.engine import EvalResult
from app.core.mistake_detector import classify_move, MoveClass
from app.rag.retriever import HybridRetriever
from app.rag.vector_store import VectorStore, Chunk
from app.rag.chat import answer_chess_question
from app.rag.generator import _check_faithfulness

logger = logging.getLogger(__name__)


class FakeEmbedder:
    """Deterministic bag-of-words embedder for fast evaluation without heavy models."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed_text(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in text.lower().replace("-", " ").replace(":", " ").replace(".", " ").split():
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            for i in range(self.dim):
                if (h >> (i % 32)) & 1:
                    vec[i] += 1.0
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self.embed_text(query)


def matches_expected(chunk: Chunk, expected_keywords: list[str], expected_ids: list[str]) -> bool:
    """Check if a retrieved chunk matches expected keywords or source IDs."""
    chunk_id = chunk.id.lower()
    chunk_text = chunk.text.lower()

    for exp_id in expected_ids:
        if exp_id.lower() in chunk_id:
            return True

    matched_kw = 0
    for kw in expected_keywords:
        if kw.lower() in chunk_id or kw.lower() in chunk_text:
            matched_kw += 1

    return matched_kw >= max(1, len(expected_keywords) // 2)


def compute_dcg(relevances: list[float], k: int) -> float:
    """Discounted Cumulative Gain at rank k."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        dcg += rel / math.log2(i + 1)
    return dcg


def compute_ndcg(relevances: list[float], k: int) -> float:
    """Normalized Discounted Cumulative Gain at rank k."""
    dcg = compute_dcg(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = compute_dcg(ideal_relevances, k)
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def evaluate_retrieval(
    retriever: HybridRetriever,
    cases: list[dict],
    k: int = 5,
    mode: str = "hybrid",
) -> dict[str, float]:
    """Compute Recall@k, MRR, nDCG@k, and p50/p95 latency for a list of evaluation cases."""
    if not cases:
        return {
            "recall@k": 0.0,
            "mrr": 0.0,
            "ndcg@k": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_mean_ms": 0.0,
            "total": 0,
        }

    hits = 0
    rr_sum = 0.0
    ndcg_sum = 0.0
    latencies: list[float] = []

    for case in cases:
        query = case["query"]
        expected_keywords = case.get("expected_source_keywords", [])
        expected_ids = case.get("expected_sources", [])

        t0 = time.perf_counter()
        try:
            chunks = retriever.retrieve(query=query, k=k, mode=mode)
        except Exception:
            chunks = []
        latencies.append((time.perf_counter() - t0) * 1000.0)

        hit_rank = None
        relevances: list[float] = []
        for rank, chunk in enumerate(chunks, start=1):
            is_rel = matches_expected(chunk, expected_keywords, expected_ids)
            relevances.append(1.0 if is_rel else 0.0)
            if is_rel and hit_rank is None:
                hit_rank = rank

        # Pad relevances to k if fewer chunks returned
        while len(relevances) < k:
            relevances.append(0.0)

        if hit_rank is not None:
            hits += 1
            rr_sum += 1.0 / hit_rank

        ndcg_sum += compute_ndcg(relevances, k)

    total = len(cases)
    sorted_latencies = sorted(latencies)
    idx_95 = max(0, int(math.ceil(0.95 * len(sorted_latencies))) - 1)

    return {
        "recall@k": round(hits / total, 3),
        "mrr": round(rr_sum / total, 3),
        "ndcg@k": round(ndcg_sum / total, 3),
        "latency_p50_ms": round(statistics.median(sorted_latencies), 2),
        "latency_p95_ms": round(sorted_latencies[idx_95], 2),
        "latency_mean_ms": round(statistics.mean(sorted_latencies), 2),
        "total": total,
    }


def compare_retrieval_modes(
    retriever: HybridRetriever,
    cases: list[dict],
    k: int = 5,
) -> dict[str, dict[str, float]]:
    """Compare baseline vector vs BM25 vs Hybrid retrieval per Section 10 ablation."""
    return {
        "vector": evaluate_retrieval(retriever, cases, k=k, mode="vector"),
        "bm25": evaluate_retrieval(retriever, cases, k=k, mode="bm25"),
        "hybrid": evaluate_retrieval(retriever, cases, k=k, mode="hybrid"),
    }


def evaluate_chat_refusals(
    cases: list[dict],
    in_scope_cases: list[dict] | None = None,
    retriever: HybridRetriever | None = None,
    min_score: float = 0.3,
) -> dict[str, Any]:
    """Evaluate 'I don't know' refusal rate as a binary classification with Precision, Recall, and F1."""
    if not cases and not in_scope_cases:
        return {"accuracy": 0.0, "refusals": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "total": 0}

    # Positive class = Out-of-Scope (should refuse)
    # Negative class = In-Scope (should answer)
    tp = 0  # correctly refused out-of-scope question
    fn = 0  # failed to refuse out-of-scope question
    tn = 0  # correctly answered in-scope question
    fp = 0  # falsely refused in-scope question

    for case in cases:
        question = case["question"]
        resp = answer_chess_question(
            question=question,
            retriever=retriever,
            min_score=min_score,
        )
        is_refused = (not resp.confident) and ("don't have enough information" in resp.answer.lower())
        if is_refused:
            tp += 1
        else:
            fn += 1

    if in_scope_cases:
        for case in in_scope_cases:
            question = case["question"]
            resp = answer_chess_question(
                question=question,
                retriever=retriever,
                min_score=min_score,
            )
            is_refused = (not resp.confident) and ("don't have enough information" in resp.answer.lower())
            if is_refused:
                fp += 1
            else:
                tn += 1
    else:
        # Default balance if in_scope_cases not passed separately
        tn = len(cases)

    total_eval = tp + fn + fp + tn
    accuracy = (tp + tn) / total_eval if total_eval > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "refusals": tp,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "total": len(cases),
    }


def evaluate_faithfulness(cases: list[dict]) -> dict[str, Any]:
    """Evaluate move faithfulness detection against ground truth."""
    if not cases:
        return {"accuracy": 0.0, "total": 0}

    correct = 0
    default_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    for case in cases:
        fen = case.get("fen_before", default_fen)
        is_faithful, _ = _check_faithfulness(
            llm_text=case["text"],
            fen_before=fen,
            played_san=case["played_san"],
            pv_uci=case["engine_pv"],
            game_moves_san=case.get("game_moves_san", []),
        )
        if is_faithful == case["expected_faithful"]:
            correct += 1

    total = len(cases)
    return {
        "accuracy": round(correct / total, 3),
        "correct": correct,
        "total": total,
    }


def evaluate_move_classification(cases: list[dict]) -> dict[str, Any]:
    """Evaluate mistake detector classification across centipawn threshold categories."""
    if not cases:
        return {"accuracy": 0.0, "macro_f1": 0.0, "total": 0}

    correct = 0
    categories = ["good", "inaccuracy", "mistake", "blunder"]
    tp_map = {c: 0 for c in categories}
    fp_map = {c: 0 for c in categories}
    fn_map = {c: 0 for c in categories}

    for case in cases:
        eval_before = EvalResult(score_cp=case["eval_before"], best_move_uci="e2e4", pv=["e2e4"])
        eval_after = EvalResult(score_cp=case["eval_after"], best_move_uci="e2e4", pv=["e2e4"])
        predicted = classify_move(eval_before, eval_after, side=case["side"]).value
        expected = case["expected_class"]

        if predicted == expected:
            correct += 1
            tp_map[expected] += 1
        else:
            fp_map[predicted] += 1
            fn_map[expected] += 1

    f1_list = []
    for c in categories:
        p = tp_map[c] / (tp_map[c] + fp_map[c]) if (tp_map[c] + fp_map[c]) > 0 else 0.0
        r = tp_map[c] / (tp_map[c] + fn_map[c]) if (tp_map[c] + fn_map[c]) > 0 else 0.0
        f1_c = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        f1_list.append(f1_c)

    total = len(cases)
    macro_f1 = statistics.mean(f1_list) if f1_list else 0.0

    return {
        "accuracy": round(correct / total, 3),
        "macro_f1": round(macro_f1, 3),
        "correct": correct,
        "total": total,
    }


def build_eval_retriever() -> HybridRetriever:
    """Build a standard in-memory hybrid retriever with representative chess knowledge."""
    store = VectorStore(embedder=FakeEmbedder(dim=64))
    corpus_docs = [
        ("article_pawn_structure", "Pawn structure fundamentals: skeleton of a chess position.", {"type": "article"}),
        ("article_pawn_structure_doubled", "Doubled pawns occur when two pawns of the same color are on the same file. They cannot protect each other and reduce mobility.", {"type": "article"}),
        ("article_pawn_structure_isolated", "An isolated pawn (IQP) has no friendly pawns on adjacent files. Natural outpost for enemy pieces, dynamic chances in middlegame, weakness in endgame.", {"type": "article"}),
        ("article_pawn_structure_backward", "A backward pawn cannot advance because the square in front is controlled by enemy pawn. Strong outpost for opponent.", {"type": "article"}),
        ("opening_B12", "Opening: Caro-Kann Defense Advance Variation (ECO B12). Moves: 1. e4 c6 2. d4 d5 3. e5", {"type": "opening", "eco": "B12"}),
        ("opening_C50", "Opening: Italian Game (ECO C50). Moves: 1. e4 e5 2. Nf3 Nc6 3. Bc4", {"type": "opening", "eco": "C50"}),
        ("opening_D06", "Opening: Queen's Gambit (ECO D06). Moves: 1. d4 d5 2. c4", {"type": "opening", "eco": "D06"}),
        ("opening_A00", "Opening: Van't Kruijs Opening (ECO A00). Moves: 1. e3", {"type": "opening", "eco": "A00"}),
    ]
    store.add(
        ids=[d[0] for d in corpus_docs],
        documents=[d[1] for d in corpus_docs],
        metadatas=[d[2] for d in corpus_docs],
    )
    return HybridRetriever(vector_store=store, bm25_corpus=corpus_docs)


def run_all_evaluations(eval_path: Path | None = None) -> dict[str, Any]:
    """Run the entire evaluation suite and return all metrics."""
    if eval_path is None:
        eval_path = Path(__file__).resolve().parent / "eval_set.json"

    data = json.loads(eval_path.read_text(encoding="utf-8"))
    retrieval_cases = data.get("retrieval_cases", [])
    oos_cases = data.get("out_of_scope_cases", [])
    in_scope_cases = data.get("in_scope_cases", [])
    faith_cases = data.get("faithfulness_cases", [])
    move_class_cases = data.get("move_classification_cases", [])

    retriever = build_eval_retriever()

    # 1. Retrieval Ablation Comparison
    retrieval_metrics = compare_retrieval_modes(retriever, retrieval_cases, k=5)

    # 2. Out-of-Scope Refusals (Precision, Recall, F1)
    refusal_metrics = evaluate_chat_refusals(
        cases=oos_cases,
        in_scope_cases=in_scope_cases,
        retriever=retriever,
        min_score=0.3,
    )

    # 3. Move Faithfulness Guardrail
    faithfulness_metrics = evaluate_faithfulness(faith_cases)

    # 4. Move Mistake Classification F1
    move_metrics = evaluate_move_classification(move_class_cases)

    results = {
        "retrieval": retrieval_metrics,
        "chat_refusals": refusal_metrics,
        "move_faithfulness": faithfulness_metrics,
        "move_classification": move_metrics,
    }

    # Print summary
    print("=" * 72)
    print("           CHESS RAG COMPREHENSIVE EVALUATION REPORT           ")
    print("=" * 72)
    print(f"Retrieval Cases Evaluated: {len(retrieval_cases)}")
    print(f"  • Dense Vector (Baseline): Recall@5 = {retrieval_metrics['vector']['recall@k']}, MRR = {retrieval_metrics['vector']['mrr']}, nDCG@5 = {retrieval_metrics['vector']['ndcg@k']}, Latency (p50/p95) = {retrieval_metrics['vector']['latency_p50_ms']}ms / {retrieval_metrics['vector']['latency_p95_ms']}ms")
    print(f"  • Sparse BM25 Only:        Recall@5 = {retrieval_metrics['bm25']['recall@k']}, MRR = {retrieval_metrics['bm25']['mrr']}, nDCG@5 = {retrieval_metrics['bm25']['ndcg@k']}, Latency (p50/p95) = {retrieval_metrics['bm25']['latency_p50_ms']}ms / {retrieval_metrics['bm25']['latency_p95_ms']}ms")
    print(f"  • Hybrid (RRF Fusion):     Recall@5 = {retrieval_metrics['hybrid']['recall@k']}, MRR = {retrieval_metrics['hybrid']['mrr']}, nDCG@5 = {retrieval_metrics['hybrid']['ndcg@k']}, Latency (p50/p95) = {retrieval_metrics['hybrid']['latency_p50_ms']}ms / {retrieval_metrics['hybrid']['latency_p95_ms']}ms")
    print("-" * 72)
    print(f"Out-of-Scope Refusal Guardrail (Precision / Recall / F1):")
    print(f"  • Precision: {refusal_metrics['precision']:.3f} | Recall: {refusal_metrics['recall']:.3f} | F1-Score: {refusal_metrics['f1']:.3f} | Accuracy: {refusal_metrics['accuracy']:.1%}")
    print("-" * 72)
    print(f"Move Classification (Mistake Detector):")
    print(f"  • Accuracy: {move_metrics['accuracy']:.1%} | Macro F1: {move_metrics['macro_f1']:.3f}")
    print("-" * 72)
    print(f"Move Faithfulness Accuracy (python-chess guardrail):")
    print(f"  • Accuracy: {faithfulness_metrics['accuracy']:.1%} ({faithfulness_metrics['correct']}/{faithfulness_metrics['total']})")
    print("=" * 72)

    return results


if __name__ == "__main__":
    run_all_evaluations()
