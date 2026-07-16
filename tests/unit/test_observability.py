"""Metrics counters/latency and span tracing."""

from __future__ import annotations

from watari.obs.metrics import Metrics
from watari.obs.tracing import span, traced


class TestMetrics:
    def test_counters_accumulate(self) -> None:
        m = Metrics()
        m.incr("requests")
        m.incr("requests", 2)
        assert m.snapshot()["counters"]["requests"] == 3  # type: ignore[index]

    def test_usage_records_tokens_and_requests(self) -> None:
        m = Metrics()
        m.record_usage(100, 20)
        counters = m.snapshot()["counters"]
        assert counters["tokens_prompt"] == 100  # type: ignore[index]
        assert counters["tokens_completion"] == 20  # type: ignore[index]
        assert counters["requests"] == 1  # type: ignore[index]

    def test_latency_percentiles(self) -> None:
        m = Metrics()
        for v in [10.0, 20.0, 30.0, 40.0, 100.0]:
            m.observe("op", v)
        lat = m.snapshot()["latency"]["op"]  # type: ignore[index]
        assert lat["count"] == 5
        assert lat["p50_ms"] > 0

    def test_reset_clears(self) -> None:
        m = Metrics()
        m.incr("x")
        m.reset()
        assert m.snapshot()["counters"] == {}


class TestTracing:
    async def test_span_records_duration(self) -> None:
        m = Metrics()
        async with span("work", metrics=m):
            pass
        assert "span.work" in m.snapshot()["latency"]  # type: ignore[operator]

    async def test_span_records_on_error(self) -> None:
        m = Metrics()
        try:
            async with span("boom", metrics=m):
                raise ValueError("x")
        except ValueError:
            pass
        assert "span.boom" in m.snapshot()["latency"]  # type: ignore[operator]

    async def test_traced_decorator(self) -> None:
        @traced("decorated")
        async def fn() -> int:
            return 42

        assert await fn() == 42
