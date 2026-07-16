"""Lightweight span tracing.

A `@traced` decorator and a `span()` context manager emit span start/end with
timing into the structured log, giving per-request waterfalls (retrieval →
LLM TTFT → tool) without an OTel SDK, collector, and exporter that a single-user
local app has no consumer for.

Span/attribute names follow OpenTelemetry semantic conventions (``duration_ms``,
``span.name``, ``span.status``) so the seam maps 1:1 onto a real OTel exporter if
one is ever added — a documented, deliberate compromise (see
``docs/architecture.md``). Timings also feed the metrics registry.
"""

from __future__ import annotations

import functools
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import ParamSpec, TypeVar

from watari.obs.logging import get_logger
from watari.obs.metrics import METRICS, Metrics

logger = get_logger("trace")

P = ParamSpec("P")
R = TypeVar("R")


@asynccontextmanager
async def span(
    name: str, *, metrics: Metrics = METRICS, **attributes: object
) -> AsyncGenerator[None]:
    """Time an async block and emit a span record."""
    start = time.perf_counter()
    status = "ok"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        duration_ms = (time.perf_counter() - start) * 1000.0
        metrics.observe(f"span.{name}", duration_ms)
        logger.info(
            "span",
            **{"span.name": name, "span.status": status, "duration_ms": round(duration_ms, 1)},
            **attributes,
        )


def traced(
    name: str | None = None,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Decorate an async function to emit a span around each call."""

    def decorator(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        span_name = name or fn.__name__

        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            async with span(span_name):
                return await fn(*args, **kwargs)

        return wrapper

    return decorator
