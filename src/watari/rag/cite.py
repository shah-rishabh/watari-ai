"""Citation formatting and validation.

Retrieved chunks are numbered ``[1]..[k]`` in the context, and the system prompt
requires the model to cite with those markers. After generation we parse the
markers back out, **validate that every cited index actually exists** (a marker
like ``[7]`` when only 5 chunks were shown is a hallucinated citation — logged
and stripped), and render the used sources as footnotes.

The fraction of cited markers that are valid is a first-class eval metric
(citation validity rate), so the parsing lives here as one tested unit.
"""

from __future__ import annotations

import re

from watari.obs.logging import get_logger
from watari.rag.models import RetrievedChunk

logger = get_logger(__name__)

_CITE_RE = re.compile(r"\[(\d+)\]")


def format_context_block(chunks: list[RetrievedChunk]) -> str:
    """Render numbered, delimited chunks for injection into the prompt."""
    if not chunks:
        return ""
    lines = ["Retrieved context (cite with [n]):", ""]
    for i, chunk in enumerate(chunks, start=1):
        location = chunk.source_path
        if chunk.heading_path:
            location += f" — {chunk.heading_path}"
        lines.append(f"[{i}] ({location})")
        lines.append(chunk.text.strip())
        lines.append("")
    return "\n".join(lines).strip()


def extract_citations(answer: str) -> list[int]:
    """Return the distinct 1-based indices cited in the answer, in order."""
    seen: set[int] = set()
    ordered: list[int] = []
    for m in _CITE_RE.finditer(answer):
        n = int(m.group(1))
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def validate_citations(answer: str, n_chunks: int) -> tuple[list[int], list[int]]:
    """Split cited indices into (valid, hallucinated) against the chunk count."""
    valid: list[int] = []
    hallucinated: list[int] = []
    for n in extract_citations(answer):
        if 1 <= n <= n_chunks:
            valid.append(n)
        else:
            hallucinated.append(n)
    return valid, hallucinated


def strip_invalid_citations(answer: str, n_chunks: int) -> str:
    """Remove out-of-range ``[n]`` markers from the answer text."""

    def repl(m: re.Match[str]) -> str:
        n = int(m.group(1))
        return m.group(0) if 1 <= n <= n_chunks else ""

    return _CITE_RE.sub(repl, answer)


def render_sources(answer: str, chunks: list[RetrievedChunk]) -> str:
    """Render a footnote list for the chunks actually cited by the answer."""
    valid, hallucinated = validate_citations(answer, len(chunks))
    if hallucinated:
        logger.warning("cite.hallucinated_markers", indices=hallucinated)
    if not valid:
        return ""
    lines = ["", "Sources:"]
    for n in valid:
        chunk = chunks[n - 1]
        location = chunk.source_path
        if chunk.heading_path:
            location += f" — {chunk.heading_path}"
        lines.append(f"[{n}] {location}")
    return "\n".join(lines)
