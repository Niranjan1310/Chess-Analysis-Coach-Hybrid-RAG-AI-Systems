"""
Ingest opening data into the vector store.

Reads TSV files from the lichess-org/chess-openings repository format:
    eco \t name \t pgn

Each opening becomes one chunk with metadata:
  - type: "opening"
  - eco: the ECO code (e.g. "B12")
  - name: opening name

Usage::

    python -m ingestion.load_openings --data-dir data/openings --persist-dir data/chroma
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_openings_tsv(filepath: Path) -> list[dict]:
    """Parse a single TSV file with columns: eco, name, pgn."""
    entries: list[dict] = []
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eco = row.get("eco", "").strip()
            name = row.get("name", "").strip()
            pgn = row.get("pgn", "").strip()
            if not (eco and name and pgn):
                continue
            # Build a readable text chunk
            text = f"Opening: {name} (ECO {eco}). Moves: {pgn}"
            entries.append({
                "id": f"opening_{eco}_{name}".replace(" ", "_").replace("/", "_")[:80],
                "text": text,
                "metadata": {
                    "type": "opening",
                    "eco": eco,
                    "source": filepath.name,
                },
            })
    return entries


def load_openings_dir(data_dir: Path) -> list[dict]:
    """Load all .tsv opening files from a directory."""
    all_entries: list[dict] = []
    tsv_files = sorted(data_dir.glob("*.tsv"))
    if not tsv_files:
        logger.warning("No .tsv files found in %s", data_dir)
        return all_entries
    for tsv_file in tsv_files:
        entries = parse_openings_tsv(tsv_file)
        logger.info("Loaded %d openings from %s", len(entries), tsv_file.name)
        all_entries.extend(entries)
    return all_entries


def ingest_openings(data_dir: str, persist_dir: str) -> int:
    """Load openings from TSV files and add them to the vector store.

    Args:
        data_dir: path to directory containing .tsv opening files.
        persist_dir: path to Chroma persistence directory.

    Returns:
        Number of openings ingested.
    """
    # Import here to avoid circular imports at module level
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
    from app.rag.vector_store import VectorStore

    entries = load_openings_dir(Path(data_dir))
    if not entries:
        logger.warning("No openings to ingest.")
        return 0

    store = VectorStore(persist_dir=persist_dir, collection_name="chess_knowledge")
    store.add(
        ids=[e["id"] for e in entries],
        documents=[e["text"] for e in entries],
        metadatas=[e["metadata"] for e in entries],
    )
    logger.info("Ingested %d openings into vector store.", len(entries))
    return len(entries)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ingest chess openings into vector store.")
    parser.add_argument("--data-dir", default="data/openings", help="Directory with .tsv files")
    parser.add_argument("--persist-dir", default="data/chroma", help="Chroma persistence directory")
    args = parser.parse_args()
    count = ingest_openings(args.data_dir, args.persist_dir)
    print(f"Done. Ingested {count} openings.")
