"""Render suite results as JSON and as a markdown table."""

from __future__ import annotations

import json
from collections.abc import Sequence

from watari.evals.models import SuiteResult


def results_to_json(results: Sequence[SuiteResult]) -> str:
    return json.dumps([r.model_dump() for r in results], indent=2, sort_keys=True)


def results_to_markdown(results: Sequence[SuiteResult]) -> str:
    """A compact markdown table: one row per (suite, metric)."""
    lines = [
        "| Suite | Model | Cases | Metric | Value |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in results:
        for m in r.metrics:
            lines.append(f"| {r.suite} | {r.model} | {r.n_cases} | {m.name} | {m.value:.3f} |")
    return "\n".join(lines)


def parse_results_json(text: str) -> list[SuiteResult]:
    data = json.loads(text)
    return [SuiteResult.model_validate(item) for item in data]
