"""Heading-aware markdown chunking.

Strategy: split the document on markdown headings so each section keeps its
semantic boundary and heading trail, then recursively split any section that
exceeds the token budget, carrying a small overlap so a fact spanning a split
point is still retrievable from either side. Hand-rolled (~150 lines) rather
than pulling in LangChain's splitter for a portfolio of this size.

Everything is token-budgeted via the same tiktoken proxy used for context
assembly, so "400 tokens" means the same thing everywhere.
"""

from __future__ import annotations

import re

from watari.core.tokens import count_tokens, decode, encode
from watari.rag.models import Chunk

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _split_sections(markdown: str) -> list[tuple[str, str]]:
    """Split markdown into (heading_path, section_text) pairs.

    The heading path is the trail of ancestor headings joined by " > ". Text
    before the first heading is attached to an empty heading path.
    """
    lines = markdown.splitlines()
    sections: list[tuple[str, str]] = []
    heading_stack: list[tuple[int, str]] = []  # (level, title)
    buf: list[str] = []

    def current_path() -> str:
        return " > ".join(title for _, title in heading_stack)

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            sections.append((current_path(), text))
        buf.clear()

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            # Pop headings at the same or deeper level, then push this one.
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
        else:
            buf.append(line)
    flush()
    return sections


def _split_by_tokens(text: str, target: int, overlap: int) -> list[str]:
    """Split text into token-bounded windows with overlap, on token boundaries."""
    token_ids = encode(text)
    if len(token_ids) <= target:
        return [text]

    step = max(1, target - overlap)
    windows: list[str] = []
    for start in range(0, len(token_ids), step):
        window_ids = token_ids[start : start + target]
        if not window_ids:
            break
        windows.append(decode(window_ids))
        if start + target >= len(token_ids):
            break
    return windows


def chunk_markdown(
    markdown: str,
    *,
    source_path: str,
    target_tokens: int = 400,
    overlap_ratio: float = 0.15,
) -> list[Chunk]:
    """Chunk a markdown document into retrievable :class:`Chunk` objects."""
    overlap = int(target_tokens * overlap_ratio)
    chunks: list[Chunk] = []
    index = 0

    for heading_path, section in _split_sections(markdown):
        if count_tokens(section) <= target_tokens:
            pieces = [section]
        else:
            pieces = _split_by_tokens(section, target_tokens, overlap)

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    source_path=source_path,
                    heading_path=heading_path,
                    chunk_index=index,
                    text=piece,
                )
            )
            index += 1

    return chunks
