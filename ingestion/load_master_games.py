"""
Load and index master games into the Chroma positions collection.

Convenience wrapper around ``ingestion.build_position_index``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ingestion.build_position_index import index_master_games

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest master games into Chroma position index.")
    parser.add_argument("--pgn", type=str, required=True, help="Path to PGN file with master games")
    parser.add_argument("--persist-dir", type=str, default="data/chroma", help="Chroma persistence directory")
    parser.add_argument("--max-games", type=int, default=100, help="Maximum number of games to sample")
    parser.add_argument("--min-move", type=int, default=8, help="Start move number")
    parser.add_argument("--max-move", type=int, default=40, help="End move number")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    pgn_path = Path(args.pgn)
    if not pgn_path.exists():
        logger.error("PGN file not found: %s", pgn_path)
        return

    count = index_master_games(
        pgn_source=pgn_path,
        persist_dir=args.persist_dir,
        max_games=args.max_games,
        min_move=args.min_move,
        max_move=args.max_move,
    )
    print(f"Successfully indexed {count} master positions into '{args.persist_dir}'.")


if __name__ == "__main__":
    main()
