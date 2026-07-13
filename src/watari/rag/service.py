"""Ingestion orchestration: discover → chunk → embed → store, incrementally."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from watari.config import Settings
from watari.obs.logging import get_logger
from watari.rag.embeddings import FastEmbedEmbedder
from watari.rag.ingest import chunk_file, discover
from watari.rag.store import RagStore

logger = get_logger(__name__)


@dataclass
class IngestResult:
    ingested_files: int
    skipped_files: int
    total_chunks: int


class IngestService:
    def __init__(self, store: RagStore, embedder: FastEmbedEmbedder, settings: Settings) -> None:
        self._store = store
        self._embedder = embedder
        self._settings = settings

    def ingest_path(self, root: Path) -> IngestResult:
        ingested = skipped = total_chunks = 0
        for path in discover(root):
            digest, chunks = chunk_file(
                path,
                target_tokens=self._settings.chunk_target_tokens,
                overlap_ratio=self._settings.chunk_overlap_ratio,
            )
            source = str(path)
            if not self._store.needs_ingest(source, digest):
                skipped += 1
                logger.info("ingest.skip_unchanged", source=source)
                continue
            if not chunks:
                skipped += 1
                continue
            embeddings = self._embedder.embed([c.text for c in chunks])
            n = self._store.upsert_document(
                source_path=source,
                content_hash=digest,
                chunks=chunks,
                embeddings=embeddings,
            )
            ingested += 1
            total_chunks += n
            logger.info("ingest.file", source=source, chunks=n)
        return IngestResult(ingested, skipped, total_chunks)
