"""Eval domain models.

Golden datasets are JSONL: one :class:`EvalCase` per line. Cases identify their
expected chunks by ``(source_path, heading_path)`` — stable, human-readable
coordinates — rather than by chunk id, which depends on ingestion order. The
runner resolves these to concrete chunk ids at eval time.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkRef(BaseModel):
    """A stable reference to an expected chunk."""

    source_path: str
    heading_path: str


class EvalCase(BaseModel):
    """One golden case. ``tags`` includes "smoke" for the CI subset."""

    id: str
    schema_version: int = 1
    question: str
    # Retrieval cases: the chunks that should be retrieved.
    relevant: list[ChunkRef] = Field(default_factory=list[ChunkRef])
    # RAG-QA cases: a reference answer and the facts a faithful answer must state.
    reference_answer: str = ""
    must_include: list[str] = Field(default_factory=list[str])
    tags: list[str] = Field(default_factory=list[str])


class MetricResult(BaseModel):
    """A single named metric value with the sample size it was computed over."""

    name: str
    value: float
    n: int


class SuiteResult(BaseModel):
    """All metrics for one suite run, plus provenance."""

    suite: str
    model: str
    n_cases: int
    metrics: list[MetricResult]

    def metric(self, name: str) -> float:
        for m in self.metrics:
            if m.name == name:
                return m.value
        raise KeyError(name)
