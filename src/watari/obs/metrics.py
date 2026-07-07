"""In-process metrics.

A single-user local app doesn't need Prometheus + a scrape target. We keep
lightweight in-memory counters and latency samples, expose them at
``GET /metrics`` (JSON) and ``watari stats``, and record the same numbers during
eval runs so figures like p50 TTFT land in the README.

Thread-safe via a lock; percentiles computed on demand from retained samples
(bounded to avoid unbounded growth).
"""

from __future__ import annotations

import threading
from bisect import insort
from dataclasses import dataclass, field

_MAX_SAMPLES = 1024


@dataclass
class _Latency:
    samples: list[float] = field(default_factory=list[float])

    def record(self, value_ms: float) -> None:
        if len(self.samples) >= _MAX_SAMPLES:
            self.samples.pop(0)
        insort(self.samples, value_ms)

    def percentile(self, p: float) -> float:
        if not self.samples:
            return 0.0
        idx = min(len(self.samples) - 1, int(p / 100.0 * len(self.samples)))
        return self.samples[idx]


class Metrics:
    """Process-wide counters and latency histograms."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._latencies: dict[str, _Latency] = {}

    def incr(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def observe(self, name: str, value_ms: float) -> None:
        with self._lock:
            self._latencies.setdefault(name, _Latency()).record(value_ms)

    def record_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.incr("tokens_prompt", prompt_tokens)
        self.incr("tokens_completion", completion_tokens)
        self.incr("requests")

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            counters = dict(self._counters)
            latencies = {
                name: {
                    "count": len(lat.samples),
                    "p50_ms": round(lat.percentile(50), 1),
                    "p95_ms": round(lat.percentile(95), 1),
                }
                for name, lat in self._latencies.items()
            }
        return {"counters": counters, "latency": latencies}

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._latencies.clear()


# Process-wide default instance. Injected where needed; tests can construct
# their own Metrics() for isolation.
METRICS = Metrics()
