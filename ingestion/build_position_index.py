"""
Build a vector search index of master game positions in Chroma.

Indexing Pipeline:
  - Sample positions from master games at moves 8-40.
  - For each position, store:
      - FEN
      - side_to_move ("white" or "black")
      - game phase ("opening", "middlegame", "endgame")
      - opening ECO
      - material signature (e.g. "RRBN v RRBB")
      - text description (from position_describer)
      - next_move_san (the move played next)
      - result (game result)
      - white, black, year
  - Chroma collection: "positions"
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path
from typing import Any
import chess
import chess.pgn

# Add backend to sys.path for position_describer and vector_store
backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.position_describer import describe_position

logger = logging.getLogger(__name__)


def get_material_signature(board: chess.Board) -> str:
    """Return compact non-pawn material signature (e.g. 'BNQR v BNQR')."""
    w_pieces = sorted([
        p.symbol().upper()
        for p in board.piece_map().values()
        if p.color == chess.WHITE and p.piece_type not in (chess.PAWN, chess.KING)
    ])
    b_pieces = sorted([
        p.symbol().upper()
        for p in board.piece_map().values()
        if p.color == chess.BLACK and p.piece_type not in (chess.PAWN, chess.KING)
    ])
    w_str = "".join(w_pieces) if w_pieces else "-"
    b_str = "".join(b_pieces) if b_pieces else "-"
    return f"{w_str} v {b_str}"


def sample_game_positions(
    pgn_text: str,
    min_move: int = 8,
    max_move: int = 40,
) -> list[dict[str, Any]]:
    """Extract sample positions and metadata from a single PGN game.

    Args:
        pgn_text: complete single game PGN.
        min_move: earliest move number to sample (inclusive).
        max_move: latest move number to sample (inclusive).

    Returns:
        List of dicts ready for insertion into VectorStore.
    """
    stream = io.StringIO(pgn_text)
    game = chess.pgn.read_game(stream)
    if game is None:
        return []

    headers = game.headers
    white = headers.get("White", "Unknown")
    black = headers.get("Black", "Unknown")
    year = headers.get("Date", "????")[:4]
    result = headers.get("Result", "*")
    eco = headers.get("ECO", "")

    board = game.board()
    positions: list[dict[str, Any]] = []

    move_count = 0
    node = game
    while node.variations:
        next_node = node.variation(0)
        move = next_node.move
        move_number = (move_count // 2) + 1

        if min_move <= move_number <= max_move:
            fen = board.fen()
            side_to_move = "white" if board.turn == chess.WHITE else "black"
            desc = describe_position(fen)
            mat_sig = get_material_signature(board)
            san_played = board.san(move)

            pos_id = f"pos_{white}_{black}_{move_number}_{side_to_move}_{san_played}".replace(" ", "_")[:80]
            pos_id = f"{pos_id}_{move_count}"

            positions.append({
                "id": pos_id,
                "text": desc.summary,
                "metadata": {
                    "fen": fen,
                    "side_to_move": side_to_move,
                    "phase": desc.phase,
                    "eco": eco,
                    "material_signature": mat_sig,
                    "next_move_san": san_played,
                    "result": result,
                    "white": white,
                    "black": black,
                    "year": year,
                    "move_number": move_number,
                },
            })

        board.push(move)
        move_count += 1
        node = next_node

    return positions


def build_position_index(
    pgn_source: str | Path,
    persist_dir: str,
    collection_name: str = "positions",
    min_move: int = 8,
    max_move: int = 40,
    max_games: int | None = None,
    embedder: Any | None = None,
) -> int:
    """Build a Chroma collection of master game positions.

    Args:
        pgn_source: path to PGN file or string containing PGN games.
        persist_dir: Chroma storage directory.
        collection_name: target Chroma collection (default: "positions").
        min_move: start move for sampling.
        max_move: end move for sampling.
        max_games: optional limit on games to parse.
        embedder: optional custom Embedder (e.g. FakeEmbedder for tests).

    Returns:
        Number of positions indexed.
    """
    from app.rag.vector_store import VectorStore

    if isinstance(pgn_source, Path) or (isinstance(pgn_source, str) and Path(pgn_source).is_file()):
        pgn_text = Path(pgn_source).read_text(encoding="utf-8")
    else:
        pgn_text = str(pgn_source)

    stream = io.StringIO(pgn_text)
    all_positions: list[dict[str, Any]] = []

    game_idx = 0
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        exporter = chess.pgn.StringExporter(headers=True, variations=True)
        game_str = game.accept(exporter)

        positions = sample_game_positions(game_str, min_move=min_move, max_move=max_move)
        all_positions.extend(positions)

        game_idx += 1
        if max_games and game_idx >= max_games:
            break

    if not all_positions:
        logger.warning("No positions sampled to index.")
        return 0

    store = VectorStore(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedder=embedder,
    )
    store.add(
        ids=[p["id"] for p in all_positions],
        documents=[p["text"] for p in all_positions],
        metadatas=[p["metadata"] for p in all_positions],
    )
    logger.info("Indexed %d positions into collection '%s'.", len(all_positions), collection_name)
    return len(all_positions)
