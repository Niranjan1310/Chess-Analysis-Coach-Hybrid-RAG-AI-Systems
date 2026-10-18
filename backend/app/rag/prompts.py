"""
Prompt templates for the chess coach RAG pipeline.

Design Principles:
  - System prompt: "You are a chess coach. Use ONLY the engine data
    and context provided. Do not calculate or invent variations."
  - Required output: short explanation, the better move with its line
    from the engine, one concept to learn, sources.
  - If retrieved context is irrelevant, explain using engine data only
    and say no extra context was found.
"""

from __future__ import annotations


SYSTEM_PROMPT = (
    "You are a chess coach helping a student understand their mistakes. "
    "Use ONLY the engine evaluation data and the context documents provided below. "
    "Do not calculate or invent new variations that are not given to you. "
    "If the context documents are not relevant to the mistake, explain "
    "using the engine data only and say that no extra context was found."
)


def build_user_prompt(
    *,
    played_move: str,
    best_move: str,
    best_line: list[str],
    eval_before: str,
    eval_after: str,
    position_description: str,
    chunks: list[dict],
) -> str:
    """Build the user-facing prompt for explaining a single mistake.

    Args:
        played_move: the move played (SAN).
        best_move: the engine's best move (SAN if available, else UCI).
        best_line: the principal variation as a list of SAN/UCI moves.
        eval_before: evaluation string before the move (e.g. "+1.2" or "M3").
        eval_after: evaluation string after the move.
        position_description: text summary from position_describer.
        chunks: list of dicts with keys ``id``, ``text``, ``score``.

    Returns:
        A formatted prompt string.
    """
    line_str = " ".join(best_line) if best_line else best_move

    # Build context block
    if chunks:
        ctx_parts = []
        for i, c in enumerate(chunks, 1):
            ctx_parts.append(f"[{c['id']}]: {c['text']}")
        context_block = "\n\n".join(ctx_parts)
    else:
        context_block = "(No relevant context documents found.)"

    return f"""## Position
{position_description}

## Engine evaluation
- Before the move: {eval_before}
- After the move: {eval_after}
- The move played: **{played_move}**
- Best move: **{best_move}**
- Best line: {line_str}

## Context documents
{context_block}

## Your task
1. In 2-3 sentences, explain WHY the played move ({played_move}) was a mistake and what it allowed the opponent to do.
2. Show the better move ({best_move}) and briefly explain the engine's main line.
3. Name ONE chess concept the student should study to avoid this type of mistake.
4. List the source document IDs you used (e.g. [source-id]).

Important: only reference moves that appear in the engine line above or the played move. Do not invent moves."""
