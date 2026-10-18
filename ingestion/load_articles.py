"""
Ingest strategy notes (markdown files) into the vector store.

Reads ``.md`` files from ``data/notes/``.  Each file is split by
top-level headings (``# ...`` or ``## ...``).  Each section becomes
one chunk (200-400 token target, but we don't hard-truncate — we
trust the author to keep sections reasonably sized).

Metadata on every chunk:
  - type: "article"
  - source: filename
  - section: heading text

Usage::

    python -m ingestion.load_articles --data-dir data/notes --persist-dir data/chroma
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """Split markdown text into (heading, body) pairs.

    A heading is any line starting with ``#`` or ``##``.
    The body is all text until the next heading (or end of file).
    """
    sections: list[tuple[str, str]] = []
    current_heading = "Introduction"
    current_body: list[str] = []

    for line in text.splitlines():
        if re.match(r"^#{1,2}\s+", line):
            # Save the previous section
            body = "\n".join(current_body).strip()
            if body:
                sections.append((current_heading, body))
            current_heading = re.sub(r"^#+\s*", "", line).strip()
            current_body = []
        else:
            current_body.append(line)

    # Last section
    body = "\n".join(current_body).strip()
    if body:
        sections.append((current_heading, body))

    return sections


def parse_article(filepath: Path) -> list[dict]:
    """Parse a markdown file into chunks (one per heading section)."""
    text = filepath.read_text(encoding="utf-8")
    sections = _split_by_headings(text)
    entries: list[dict] = []

    for heading, body in sections:
        chunk_text = f"{heading}\n\n{body}"
        # Stable ID from file + heading
        raw_id = f"{filepath.stem}_{heading}"
        chunk_id = "article_" + hashlib.md5(raw_id.encode()).hexdigest()[:12]

        entries.append({
            "id": chunk_id,
            "text": chunk_text,
            "metadata": {
                "type": "article",
                "source": filepath.name,
                "section": heading,
            },
        })

    return entries


def load_articles_dir(data_dir: Path) -> list[dict]:
    """Load all .md article files from a directory."""
    all_entries: list[dict] = []
    md_files = sorted(data_dir.glob("*.md"))
    if not md_files:
        logger.warning("No .md files found in %s", data_dir)
        return all_entries
    for md_file in md_files:
        entries = parse_article(md_file)
        logger.info("Loaded %d sections from %s", len(entries), md_file.name)
        all_entries.extend(entries)
    return all_entries


def ingest_articles(data_dir: str, persist_dir: str) -> int:
    """Load articles from markdown files and add them to the vector store.

    Args:
        data_dir: path to directory containing .md files.
        persist_dir: path to Chroma persistence directory.

    Returns:
        Number of article chunks ingested.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
    from app.rag.vector_store import VectorStore

    entries = load_articles_dir(Path(data_dir))
    if not entries:
        logger.warning("No articles to ingest.")
        return 0

    store = VectorStore(persist_dir=persist_dir, collection_name="chess_knowledge")
    store.add(
        ids=[e["id"] for e in entries],
        documents=[e["text"] for e in entries],
        metadatas=[e["metadata"] for e in entries],
    )
    logger.info("Ingested %d article chunks into vector store.", len(entries))
    return len(entries)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ingest strategy articles into vector store.")
    parser.add_argument("--data-dir", default="data/notes", help="Directory with .md files")
    parser.add_argument("--persist-dir", default="data/chroma", help="Chroma persistence directory")
    args = parser.parse_args()
    count = ingest_articles(args.data_dir, args.persist_dir)
    print(f"Done. Ingested {count} article chunks.")
