"""Gate logic, report round-trip, and dataset loading."""

from __future__ import annotations

from pathlib import Path

from watari.evals.gate import check_gate
from watari.evals.harness import DATASETS_DIR
from watari.evals.models import MetricResult, SuiteResult
from watari.evals.report import (
    parse_results_json,
    results_to_json,
    results_to_markdown,
)
from watari.evals.runner import load_dataset


def _result(**metrics: float) -> SuiteResult:
    return SuiteResult(
        suite="retrieval",
        model="m",
        n_cases=10,
        metrics=[MetricResult(name=k, value=v, n=10) for k, v in metrics.items()],
    )


def test_gate_passes_when_above_floor() -> None:
    results = [_result(recall_5=0.9, mrr=0.8)]
    thresholds = {"retrieval": {"recall_5": 0.75, "mrr": 0.65}}
    assert check_gate(results, thresholds) == []


def test_gate_fails_below_floor() -> None:
    results = [_result(recall_5=0.5)]
    thresholds = {"retrieval": {"recall_5": 0.75}}
    violations = check_gate(results, thresholds)
    assert len(violations) == 1
    assert violations[0].metric == "recall_5"
    assert violations[0].value == 0.5


def test_gate_ignores_metrics_without_a_floor() -> None:
    results = [_result(untracked=0.0)]
    thresholds = {"retrieval": {"recall_5": 0.75}}
    assert check_gate(results, thresholds) == []


def test_report_json_roundtrips() -> None:
    results = [_result(recall_5=0.9)]
    restored = parse_results_json(results_to_json(results))
    assert restored[0].metric("recall_5") == 0.9


def test_markdown_has_a_row_per_metric() -> None:
    md = results_to_markdown([_result(a=0.1, b=0.2)])
    assert md.count("\n") == 3  # header + separator + 2 metric rows


def test_smoke_filter_selects_subset() -> None:
    path = DATASETS_DIR / "retrieval_v1.jsonl"
    everything = load_dataset(path)
    smoke = load_dataset(path, smoke_only=True)
    assert 0 < len(smoke) < len(everything)
    assert all("smoke" in c.tags for c in smoke)


def test_committed_datasets_are_valid_and_nonempty() -> None:
    for name in ("retrieval_v1.jsonl", "rag_qa_v1.jsonl"):
        cases = load_dataset(DATASETS_DIR / name)
        assert cases, f"{name} is empty"
        assert all(c.question for c in cases)
        assert len({c.id for c in cases}) == len(cases), "duplicate ids"


def test_agent_and_injection_datasets_load() -> None:
    from watari.evals.agent_eval import load_agent
    from watari.evals.injection import load_injection

    agent = load_agent(DATASETS_DIR / "agent_v1.jsonl")
    assert agent and all(c.expected_tools for c in agent)
    assert all(c.assert_file or c.assert_task for c in agent)

    inj = load_injection(DATASETS_DIR / "injection_v1.jsonl")
    assert inj and all(c.canary in c.attack or c.canary for c in inj)


def test_memory_dataset_loads() -> None:
    from watari.evals.memory_eval import load_memory

    cases = load_memory(DATASETS_DIR / "memory_v1.jsonl")
    assert cases
    assert all(c.expected_facts and c.recall_query and c.recall_expect for c in cases)


def test_golden_heading_paths_resolve_to_real_corpus_chunks() -> None:
    # Guards against dataset drift: every relevant chunk ref in the golden data
    # must correspond to an actual chunk the corpus produces. A typo here would
    # otherwise silently tank recall.
    from watari.evals.harness import CORPORA_DIR
    from watari.rag.chunking import chunk_markdown

    real: set[tuple[str, str]] = set()
    for f in sorted(Path(CORPORA_DIR).glob("*.md")):
        for c in chunk_markdown(f.read_text(encoding="utf-8"), source_path=f.name):
            real.add((c.source_path, c.heading_path))

    for name in ("retrieval_v1.jsonl", "rag_qa_v1.jsonl"):
        for case in load_dataset(DATASETS_DIR / name):
            for ref in case.relevant:
                key = (Path(ref.source_path).name, ref.heading_path)
                assert key in real, f"{name} {case.id}: {key} not in corpus"
