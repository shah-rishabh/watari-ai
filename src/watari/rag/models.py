"""Shared RAG domain models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A retrievable unit of a document.

    ``heading_path`` records the markdown heading trail (e.g.
    ``"Projects > Watari > Design"``) so citations can point at a meaningful
    location, not just a file.
    """

    source_path: str
    heading_path: str = ""
    chunk_index: int = 0
    text: str


class RetrievedChunk(BaseModel):
    """A chunk returned from search, with its provenance and fused score."""

    chunk_id: int
    source_path: str
    heading_path: str = ""
    chunk_index: int = 0
    text: str
    score: float = Field(default=0.0, description="Fused RRF score; higher is better.")
