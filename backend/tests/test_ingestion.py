"""
Tests for ingestion scripts (load_openings, load_articles).
"""

import tempfile
from pathlib import Path

import pytest

# Add ingestion to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ingestion"))

from ingestion.load_openings import parse_openings_tsv, load_openings_dir
from ingestion.load_articles import parse_article, load_articles_dir, _split_by_headings


# -- load_openings -------------------------------------------------------


class TestParseOpeningsTsv:

    def test_parse_valid_tsv(self, tmp_path):
        tsv = tmp_path / "test.tsv"
        tsv.write_text(
            "eco\tname\tpgn\n"
            "B12\tCaro-Kann Defense\t1. e4 c6 2. d4 d5 3. e5\n"
            "C50\tItalian Game\t1. e4 e5 2. Nf3 Nc6 3. Bc4\n",
            encoding="utf-8",
        )
        entries = parse_openings_tsv(tsv)
        assert len(entries) == 2
        assert entries[0]["metadata"]["eco"] == "B12"
        assert entries[0]["metadata"]["type"] == "opening"
        assert "Caro-Kann" in entries[0]["text"]

    def test_empty_tsv(self, tmp_path):
        tsv = tmp_path / "empty.tsv"
        tsv.write_text("eco\tname\tpgn\n", encoding="utf-8")
        entries = parse_openings_tsv(tsv)
        assert entries == []

    def test_load_dir(self, tmp_path):
        tsv = tmp_path / "openings.tsv"
        tsv.write_text(
            "eco\tname\tpgn\n"
            "A00\tTest Opening\t1. e3\n",
            encoding="utf-8",
        )
        entries = load_openings_dir(tmp_path)
        assert len(entries) == 1


# -- load_articles -------------------------------------------------------


class TestSplitByHeadings:

    def test_basic_split(self):
        text = "# Heading 1\nBody one.\n## Heading 2\nBody two."
        sections = _split_by_headings(text)
        assert len(sections) == 2
        assert sections[0][0] == "Heading 1"
        assert sections[1][0] == "Heading 2"

    def test_no_headings(self):
        text = "Just some text without any headings."
        sections = _split_by_headings(text)
        assert len(sections) == 1
        assert sections[0][0] == "Introduction"

    def test_empty_sections_skipped(self):
        text = "# H1\n# H2\nContent here."
        sections = _split_by_headings(text)
        # H1 has no body so is skipped
        assert len(sections) == 1
        assert sections[0][0] == "H2"


class TestParseArticle:

    def test_parse_md(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text(
            "# Topic A\nSome content about topic A.\n"
            "## Subtopic B\nMore content about B.\n",
            encoding="utf-8",
        )
        entries = parse_article(md)
        assert len(entries) == 2
        assert entries[0]["metadata"]["type"] == "article"
        assert entries[0]["metadata"]["source"] == "test.md"

    def test_load_dir(self, tmp_path):
        md = tmp_path / "notes.md"
        md.write_text("# Title\nBody text.", encoding="utf-8")
        entries = load_articles_dir(tmp_path)
        assert len(entries) == 1
