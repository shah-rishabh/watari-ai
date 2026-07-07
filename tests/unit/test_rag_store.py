"""RagStore roundtrip, hybrid search, and incremental ingest.

Uses the real fastembed embedder (module-scoped so the ONNX model loads once)
and a temp-file store. Marked as the RAG store's integration surface; fast
enough (single-digit seconds) to run in the default unit suite.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from watari.config import Settings
from watari.rag.chunking import chunk_markdown
from watari.rag.embeddings import FastEmbedEmbedder
from watari.rag.ingest import content_hash
from watari.rag.service import IngestService
from watari.rag.store import RagStore


@pytest.fixture(scope="module")
def embedder() -> FastEmbedEmbedder:
    return FastEmbedEmbedder(Settings())


@pytest.fixture
def store(tmp_path: Path) -> Iterator[RagStore]:
    s = RagStore(Settings(data_dir=tmp_path))
    s.connect()
    try:
        yield s
    finally:
        s.close()


DOC = """# Watari
## Design
The vector store uses sqlite-vec with FTS5 for hybrid retrieval.
## Privacy
All your data stays on your own machine and is never sent to the cloud.
"""


def _ingest(store: RagStore, embedder: FastEmbedEmbedder, text: str) -> int:
    chunks = chunk_markdown(text, source_path="watari.md")
    embeddings = embedder.embed([c.text for c in chunks])
    return store.upsert_document(
        source_path="watari.md",
        content_hash=content_hash(text),
        chunks=chunks,
        embeddings=embeddings,
    )


def test_ingest_and_stats(store: RagStore, embedder: FastEmbedEmbedder) -> None:
    n = _ingest(store, embedder, DOC)
    assert n == 2
    assert store.stats() == {"documents": 1, "chunks": 2}


def test_semantic_query_finds_paraphrased_chunk(
    store: RagStore, embedder: FastEmbedEmbedder
) -> None:
    _ingest(store, embedder, DOC)
    q = "is my information kept confidential?"  # no lexical overlap
    results = store.hybrid_search(q, embedder.embed_query(q), top_k=1)
    assert results
    assert "Privacy" in results[0].heading_path


def test_keyword_query_finds_exact_term(store: RagStore, embedder: FastEmbedEmbedder) -> None:
    _ingest(store, embedder, DOC)
    q = "sqlite-vec FTS5"
    results = store.hybrid_search(q, embedder.embed_query(q), top_k=1)
    assert results
    assert "Design" in results[0].heading_path


def test_reingest_unchanged_is_skipped(store: RagStore, embedder: FastEmbedEmbedder) -> None:
    _ingest(store, embedder, DOC)
    assert store.needs_ingest("watari.md", content_hash(DOC)) is False


def test_reingest_changed_replaces_chunks(store: RagStore, embedder: FastEmbedEmbedder) -> None:
    _ingest(store, embedder, DOC)
    changed = DOC + "\n## Extra\nAn additional section about backups.\n"
    assert store.needs_ingest("watari.md", content_hash(changed)) is True
    _ingest(store, embedder, changed)
    # Still one document (replaced, not duplicated), now with more chunks.
    stats = store.stats()
    assert stats["documents"] == 1
    assert stats["chunks"] == 3


def test_ingest_service_is_idempotent(tmp_path: Path, embedder: FastEmbedEmbedder) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text(DOC, encoding="utf-8")

    store = RagStore(Settings(data_dir=tmp_path))
    store.connect()
    try:
        svc = IngestService(store, embedder, Settings(data_dir=tmp_path))
        first = svc.ingest_path(docs)
        assert first.ingested_files == 1
        second = svc.ingest_path(docs)
        assert second.ingested_files == 0
        assert second.skipped_files == 1
    finally:
        store.close()
