"""Document loading and ingestion.

Loaders normalise every source to markdown so there is a single downstream path:
markdown files pass through as-is; PDFs are converted with ``pymupdf4llm`` (which
preserves headings, giving the chunker a real heading trail to work with).

Ingestion is incremental: each file's content hash is stored, and unchanged
files are skipped on re-ingest, so ``watari ingest`` is idempotent.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import cast

from watari.obs.logging import get_logger
from watari.rag.chunking import chunk_markdown
from watari.rag.models import Chunk

logger = get_logger(__name__)

MARKDOWN_SUFFIXES = {".md", ".markdown", ".txt"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = MARKDOWN_SUFFIXES | PDF_SUFFIXES


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_as_markdown(path: Path) -> str:
    """Load a supported file, returning its markdown representation."""
    suffix = path.suffix.lower()
    if suffix in MARKDOWN_SUFFIXES:
        return path.read_text(encoding="utf-8")
    if suffix in PDF_SUFFIXES:
        import pymupdf4llm  # imported lazily; heavy dependency

        # Without page_chunks, to_markdown returns a single markdown string
        # (the list form only occurs when page_chunks=True, which we don't set).
        # pymupdf4llm ships no type stubs, hence the cast.
        to_markdown = cast("object", pymupdf4llm.to_markdown)  # pyright: ignore[reportUnknownMemberType]
        assert callable(to_markdown)
        result: object = to_markdown(str(path))
        if not isinstance(result, str):
            raise TypeError("expected markdown string from pymupdf4llm")
        return result
    raise ValueError(f"unsupported file type: {path.suffix}")


def discover(root: Path) -> Iterator[Path]:
    """Yield supported files under ``root`` (or ``root`` itself if it's a file)."""
    if root.is_file():
        if root.suffix.lower() in SUPPORTED_SUFFIXES:
            yield root
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def chunk_file(path: Path, *, target_tokens: int, overlap_ratio: float) -> tuple[str, list[Chunk]]:
    """Load and chunk one file; returns (content_hash, chunks)."""
    markdown = load_as_markdown(path)
    digest = content_hash(markdown)
    chunks = chunk_markdown(
        markdown,
        source_path=str(path),
        target_tokens=target_tokens,
        overlap_ratio=overlap_ratio,
    )
    return digest, chunks


def iter_chunkable(paths: Iterable[Path]) -> Iterator[Path]:
    for path in paths:
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path
