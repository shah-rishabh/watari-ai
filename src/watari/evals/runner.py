"""Eval suite runner.

Loads a golden JSONL dataset, executes each case against the live RAG stack, and
computes the suite's metrics. Two suites are wired here:

- ``retrieval`` — deterministic: retrieve for each question, score recall@k and
  MRR against the golden chunk refs (resolved to chunk ids via heading path).
- ``rag-qa``    — generation quality: answer each question with RAG, then judge
  faithfulness (against retrieved context) and answer relevance with the LLM
  judge, plus deterministic citation validity.

Cases run with bounded concurrency. The runner returns :class:`SuiteResult`
objects; report/gate handle formatting and CI enforcement.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from watari.config import Settings
from watari.core.chat import ChatService
from watari.evals.metrics.judge import Judge
from watari.evals.metrics.retrieval import (
    citation_validity,
    mean_recall_at_k,
    mean_reciprocal_rank,
)
from watari.evals.models import ChunkRef, EvalCase, MetricResult, SuiteResult
from watari.obs.logging import get_logger
from watari.rag.cite import extract_citations, format_context_block
from watari.rag.retrieve import Retriever

logger = get_logger(__name__)

_CONCURRENCY = 4


def load_dataset(path: Path, *, smoke_only: bool = False) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = EvalCase.model_validate_json(line)
        if smoke_only and "smoke" not in case.tags:
            continue
        cases.append(case)
    return cases


def _key(source_path: str, heading_path: str) -> str:
    # Match on the file's basename so golden data (relative names like
    # "profile.md") lines up with retrieved chunks whose source_path includes
    # the ingestion directory prefix.
    return f"{Path(source_path).name}::{heading_path}"


def _ref_key(ref: ChunkRef) -> str:
    return _key(ref.source_path, ref.heading_path)


async def _gather_bounded[T](coros: list[Coroutine[Any, Any, T]]) -> list[T]:
    sem = asyncio.Semaphore(_CONCURRENCY)

    async def run(coro: Coroutine[Any, Any, T]) -> T:
        async with sem:
            return await coro

    return await asyncio.gather(*(run(c) for c in coros))


async def run_retrieval_suite(
    cases: list[EvalCase], retriever: Retriever, settings: Settings, model: str
) -> SuiteResult:
    async def eval_case(case: EvalCase) -> tuple[list[str], set[str]]:
        chunks = await retriever.retrieve(case.question, top_k=10)
        retrieved_keys = [_key(c.source_path, c.heading_path) for c in chunks]
        relevant_keys = {_ref_key(r) for r in case.relevant}
        return retrieved_keys, relevant_keys

    rankings = await _gather_bounded([eval_case(c) for c in cases])

    metrics = [
        MetricResult(name="recall@3", value=mean_recall_at_k(rankings, 3), n=len(cases)),
        MetricResult(name="recall@5", value=mean_recall_at_k(rankings, 5), n=len(cases)),
        MetricResult(name="recall@10", value=mean_recall_at_k(rankings, 10), n=len(cases)),
        MetricResult(name="mrr", value=mean_reciprocal_rank(rankings), n=len(cases)),
    ]
    return SuiteResult(suite="retrieval", model=model, n_cases=len(cases), metrics=metrics)


async def run_rag_qa_suite(
    cases: list[EvalCase],
    chat: ChatService,
    retriever: Retriever,
    judge: Judge,
    settings: Settings,
    model: str,
) -> SuiteResult:
    async def eval_case(case: EvalCase) -> tuple[float, float, float]:
        chunks = await retriever.retrieve(case.question)
        context = format_context_block(chunks)
        # Generate an answer grounded in the same chunks.
        session_id = await chat.create_session()
        parts: list[str] = []
        async for delta in chat.stream_reply(session_id, case.question, use_rag=True):
            if delta.content:
                parts.append(delta.content)
        answer = "".join(parts)

        cited = extract_citations(answer)
        cite_score = citation_validity(cited, len(chunks))
        faith = await judge.faithfulness(answer, context)
        relevance = await judge.answer_relevance(case.question, answer)
        return faith.score, relevance, cite_score

    triples = await _gather_bounded([eval_case(c) for c in cases])
    n = len(triples) or 1
    faith_avg = sum(t[0] for t in triples) / n
    rel_avg = sum(t[1] for t in triples) / n
    cite_avg = sum(t[2] for t in triples) / n

    metrics = [
        MetricResult(name="faithfulness", value=faith_avg, n=len(cases)),
        MetricResult(name="answer_relevance", value=rel_avg, n=len(cases)),
        MetricResult(name="citation_validity", value=cite_avg, n=len(cases)),
    ]
    return SuiteResult(suite="rag-qa", model=model, n_cases=len(cases), metrics=metrics)


def dataset_path(suite: str, datasets_dir: Path) -> Path:
    filename = {"retrieval": "retrieval_v1.jsonl", "rag-qa": "rag_qa_v1.jsonl"}[suite]
    return datasets_dir / filename
