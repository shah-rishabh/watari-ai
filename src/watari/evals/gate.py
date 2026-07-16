"""Threshold gating for CI.

Compares a run's metrics against absolute floors declared in
``evals/thresholds.json``. Floors (not exact-match against a baseline) are
deliberate: small local models are nondeterministic, so an exact-match gate would
be permanently flaky. A regression is a metric dropping *below its floor* — a
real quality drop, not run-to-run noise. Baselines are recorded separately (for
humans to see trends); the gate itself only enforces floors.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from watari.evals.models import SuiteResult


@dataclass
class GateViolation:
    suite: str
    metric: str
    value: float
    floor: float

    def __str__(self) -> str:
        return f"{self.suite}.{self.metric} = {self.value:.3f} < floor {self.floor:.3f}"


def load_thresholds(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {suite: dict(metrics) for suite, metrics in data.items()}


def check_gate(
    results: Sequence[SuiteResult], thresholds: dict[str, dict[str, float]]
) -> list[GateViolation]:
    """Return the list of metrics that fell below their floor (empty = pass)."""
    violations: list[GateViolation] = []
    for r in results:
        floors = thresholds.get(r.suite, {})
        for m in r.metrics:
            floor = floors.get(m.name)
            if floor is not None and m.value < floor:
                violations.append(
                    GateViolation(suite=r.suite, metric=m.name, value=m.value, floor=floor)
                )
    return violations
