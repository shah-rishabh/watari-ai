"""Heading-aware chunking."""

from __future__ import annotations

from watari.core.tokens import count_tokens
from watari.rag.chunking import chunk_markdown


def test_splits_on_headings_with_heading_path() -> None:
    md = "# Root\n## A\nalpha text\n## B\nbravo text\n"
    chunks = chunk_markdown(md, source_path="doc.md")
    paths = [c.heading_path for c in chunks]
    assert "Root > A" in paths
    assert "Root > B" in paths


def test_nested_heading_trail() -> None:
    md = "# Top\n## Mid\n### Leaf\ncontent here\n"
    chunks = chunk_markdown(md, source_path="doc.md")
    assert chunks[0].heading_path == "Top > Mid > Leaf"


def test_preamble_before_first_heading_has_empty_path() -> None:
    md = "intro paragraph\n# Heading\nbody\n"
    chunks = chunk_markdown(md, source_path="doc.md")
    assert chunks[0].heading_path == ""
    assert "intro" in chunks[0].text


def test_large_section_is_split_with_overlap() -> None:
    body = " ".join(f"word{i}" for i in range(2000))
    md = f"# Big\n{body}\n"
    chunks = chunk_markdown(md, source_path="doc.md", target_tokens=100, overlap_ratio=0.2)
    assert len(chunks) > 1
    # Every chunk stays near the token budget.
    assert all(count_tokens(c.text) <= 100 for c in chunks)
    # Overlap: the tail of chunk 0 reappears at the head of chunk 1.
    first_tail = chunks[0].text.split()[-3:]
    assert any(w in chunks[1].text for w in first_tail)


def test_chunk_indices_are_sequential() -> None:
    md = "# A\nx\n## B\ny\n## C\nz\n"
    chunks = chunk_markdown(md, source_path="doc.md")
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_empty_document_yields_no_chunks() -> None:
    assert chunk_markdown("   \n\n", source_path="doc.md") == []
