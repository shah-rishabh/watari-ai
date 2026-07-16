"""In-process metrics endpoint (JSON)."""

from __future__ import annotations

from fastapi import APIRouter

from watari.obs.metrics import METRICS

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def metrics() -> dict[str, object]:
    """Token counters and latency percentiles (p50 TTFT, reply latency, spans)."""
    return METRICS.snapshot()
