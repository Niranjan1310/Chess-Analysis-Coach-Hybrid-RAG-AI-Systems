"""Tests for rag/prompts.py."""

from app.rag.prompts import SYSTEM_PROMPT, build_user_prompt


def test_system_prompt_content():
    assert "chess coach" in SYSTEM_PROMPT.lower()
    assert "ONLY" in SYSTEM_PROMPT
    assert "invent" in SYSTEM_PROMPT.lower()


def test_build_user_prompt_with_chunks():
    chunks = [
        {"id": "opening-1", "text": "The Sicilian Defense begins 1. e4 c5."},
        {"id": "tactic-2", "text": "Pins restrict piece movement."},
    ]
    prompt = build_user_prompt(
        played_move="Nf3",
        best_move="Nc3",
        best_line=["Nc3", "d6", "d4"],
        eval_before="+0.40",
        eval_after="-0.60",
        position_description="White has slight advantage in the opening.",
        chunks=chunks,
    )

    assert "Nf3" in prompt
    assert "Nc3" in prompt
    assert "Nc3 d6 d4" in prompt
    assert "+0.40" in prompt
    assert "-0.60" in prompt
    assert "White has slight advantage in the opening." in prompt
    assert "[opening-1]: The Sicilian Defense begins 1. e4 c5." in prompt
    assert "[tactic-2]: Pins restrict piece movement." in prompt
    assert "Do not invent moves" in prompt


def test_build_user_prompt_empty_chunks():
    prompt = build_user_prompt(
        played_move="Qh5",
        best_move="Nf3",
        best_line=[],
        eval_before="0.00",
        eval_after="-1.50",
        position_description="Standard opening.",
        chunks=[],
    )
    assert "(No relevant context documents found.)" in prompt
    assert "Qh5" in prompt
    assert "Nf3" in prompt
