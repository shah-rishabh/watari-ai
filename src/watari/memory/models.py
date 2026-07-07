"""Memory domain models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Category(StrEnum):
    PREFERENCE = "preference"
    BIOGRAPHICAL = "biographical"
    PROJECT = "project"
    OTHER = "other"


class Fact(BaseModel):
    """An atomic fact about the user, as extracted or stored."""

    fact: str = Field(min_length=1, max_length=500)
    category: Category = Category.OTHER
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class StoredMemory(BaseModel):
    """A persisted memory row."""

    id: int
    fact: str
    category: Category
    active: bool
    source: str | None = None


class RecalledMemory(BaseModel):
    """A memory returned from similarity retrieval."""

    id: int
    fact: str
    category: Category
    score: float = 0.0
