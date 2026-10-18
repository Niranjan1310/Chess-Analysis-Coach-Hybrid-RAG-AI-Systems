"""Tests for evals suite."""

import json
from pathlib import Path
from evals.rag_eval import (
    build_eval_retriever,
    evaluate_retrieval,
    compare_retrieval_modes,
    evaluate_chat_refusals,
    evaluate_faithfulness,
    run_all_evaluations,
)


def test_eval_set_json_exists_and_valid():
    eval_file = Path("D:/NST/evals/eval_set.json")
    assert eval_file.exists()
    data = json.loads(eval_file.read_text(encoding="utf-8"))
    assert len(data["retrieval_cases"]) >= 20
    assert len(data["out_of_scope_cases"]) >= 8
    assert len(data["faithfulness_cases"]) >= 5


def test_retrieval_evaluations():
    retriever = build_eval_retriever()
    cases = [
        {
            "query": "doubled pawns weakness",
            "expected_source_keywords": ["doubled"],
            "expected_sources": ["article_pawn_structure_doubled"]
        },
        {
            "query": "Caro-Kann Advance B12",
            "expected_source_keywords": ["caro-kann", "b12"],
            "expected_sources": ["opening_B12"]
        }
    ]
    res = compare_retrieval_modes(retriever, cases, k=3)
    assert res["hybrid"]["recall@k"] > 0
    assert res["hybrid"]["mrr"] > 0


def test_out_of_scope_refusal_eval():
    oos_cases = [
        {"question": "How to bake a sourdough bread?"},
        {"question": "What is the capital of Mongolia?"}
    ]
    metrics = evaluate_chat_refusals(oos_cases, retriever=None, min_score=0.3)
    assert metrics["accuracy"] == 1.0
    assert metrics["refusals"] == 2


def test_faithfulness_eval():
    cases = [
        {
            "played_san": "f3",
            "played_uci": "f2f3",
            "engine_pv": ["d2d4"],
            "text": "White played f3 instead of d4.",
            "expected_faithful": True
        },
        {
            "played_san": "f3",
            "played_uci": "f2f3",
            "engine_pv": ["d2d4"],
            "text": "White played f3. Better was Qh5 winning a queen.",
            "expected_faithful": False
        }
    ]
    metrics = evaluate_faithfulness(cases)
    assert metrics["accuracy"] == 1.0


def test_run_all_evaluations_end_to_end():
    results = run_all_evaluations()
    assert "retrieval" in results
    assert "chat_refusals" in results
    assert "move_faithfulness" in results
    assert results["chat_refusals"]["accuracy"] >= 0.8
    assert results["move_faithfulness"]["accuracy"] >= 0.8
