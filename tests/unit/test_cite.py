"""Citation extraction, validation, and rendering."""

from __future__ import annotations

from watari.rag.cite import (
    extract_citations,
    format_context_block,
    render_sources,
    strip_invalid_citations,
    validate_citations,
)
from watari.rag.models import RetrievedChunk


def _chunks(n: int) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=i,
            source_path=f"doc{i}.md",
            heading_path=f"H{i}",
            chunk_index=i,
            text=f"text {i}",
        )
        for i in range(1, n + 1)
    ]


def test_extract_deduplicates_and_preserves_order() -> None:
    assert extract_citations("a [2] b [1] c [2] d") == [2, 1]


def test_validate_splits_valid_and_hallucinated() -> None:
    valid, hallucinated = validate_citations("see [1] and [5] and [2]", n_chunks=2)
    assert valid == [1, 2]
    assert hallucinated == [5]


def test_strip_removes_only_out_of_range_markers() -> None:
    stripped = strip_invalid_citations("keep [1] drop [9] keep [2]", n_chunks=2)
    assert "[1]" in stripped
    assert "[2]" in stripped
    assert "[9]" not in stripped


def test_render_sources_lists_only_cited_chunks() -> None:
    chunks = _chunks(3)
    out = render_sources("grounded in [1] and [3]", chunks)
    assert "[1] doc1.md — H1" in out
    assert "[3] doc3.md — H3" in out
    assert "doc2.md" not in out


def test_render_sources_empty_when_nothing_cited() -> None:
    assert render_sources("no citations here", _chunks(2)) == ""


def test_context_block_numbers_chunks_from_one() -> None:
    block = format_context_block(_chunks(2))
    assert "[1] (doc1.md — H1)" in block
    assert "[2] (doc2.md — H2)" in block
