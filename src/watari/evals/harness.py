"""End-to-end eval orchestration.

Builds an *isolated* eval environment (a temp RAG store seeded from
``evals/corpora``) so retrieval is deterministic and never touches the user's
real data, runs the requested suites, and returns results plus a markdown table.
The CLI (`watari evals run`) is a thin wrapper over :func:`run_suites`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from watari.config import Settings
from watari.core.chat import ChatService
from watari.core.llm import OpenAICompatibleProvider
from watari.core.session import SessionStore
from watari.evals.agent_eval import load_agent, run_agent_suite
from watari.evals.injection import load_injection, run_injection_suite
from watari.evals.metrics.judge import Judge
from watari.evals.models import SuiteResult
from watari.evals.runner import (
    dataset_path,
    load_dataset,
    run_rag_qa_suite,
    run_retrieval_suite,
)
from watari.obs.logging import get_logger
from watari.rag.embeddings import FastEmbedEmbedder
from watari.rag.retrieve import Retriever
from watari.rag.service import IngestService
from watari.rag.store import RagStore

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
CORPORA_DIR = REPO_ROOT / "evals" / "corpora"
DATASETS_DIR = REPO_ROOT / "evals" / "datasets"

SUITES = ("retrieval", "rag-qa", "agent", "injection")
_RAG_SUITES = {"retrieval", "rag-qa"}


async def run_suites(
    suites: list[str],
    *,
    settings: Settings,
    smoke_only: bool = False,
    corpora_dir: Path = CORPORA_DIR,
    datasets_dir: Path = DATASETS_DIR,
) -> list[SuiteResult]:
    """Run the named suites, each in isolation.

    RAG suites share a temp store seeded from the corpus; agent and injection
    suites self-isolate (fresh per-case data dirs / their own provider), so they
    are dispatched separately and need no corpus.
    """
    results: list[SuiteResult] = []

    rag_requested = [s for s in suites if s in _RAG_SUITES]
    if rag_requested:
        results.extend(
            await _run_rag_suites(
                rag_requested,
                settings=settings,
                smoke_only=smoke_only,
                corpora_dir=corpora_dir,
                datasets_dir=datasets_dir,
            )
        )

    for suite in suites:
        if suite == "agent":
            cases = load_agent(datasets_dir / "agent_v1.jsonl", smoke_only=smoke_only)
            results.append(await run_agent_suite(cases, settings=settings))
        elif suite == "injection":
            cases = load_injection(datasets_dir / "injection_v1.jsonl", smoke_only=smoke_only)
            results.append(await run_injection_suite(cases, settings=settings))
        elif suite not in _RAG_SUITES:
            raise ValueError(f"unknown suite: {suite}")

    return results


async def _run_rag_suites(
    suites: list[str],
    *,
    settings: Settings,
    smoke_only: bool,
    corpora_dir: Path,
    datasets_dir: Path,
) -> list[SuiteResult]:
    with tempfile.TemporaryDirectory(prefix="watari-evals-") as tmp:
        eval_settings = settings.model_copy(update={"data_dir": Path(tmp)})

        rag_store = RagStore(eval_settings)
        rag_store.connect()
        embedder = FastEmbedEmbedder(eval_settings)
        retriever = Retriever(rag_store, embedder, eval_settings)

        # Seed the corpus.
        ingest = IngestService(rag_store, embedder, eval_settings)
        result = ingest.ingest_path(corpora_dir)
        logger.info("evals.seeded", chunks=result.total_chunks)

        provider = OpenAICompatibleProvider(eval_settings)
        session_store = SessionStore(eval_settings.db_path)
        await session_store.connect()
        chat = ChatService(provider, session_store, eval_settings, retriever=retriever)
        judge = Judge(provider, eval_settings.judge_model)

        results: list[SuiteResult] = []
        try:
            for suite in suites:
                cases = load_dataset(dataset_path(suite, datasets_dir), smoke_only=smoke_only)
                if suite == "retrieval":
                    results.append(
                        await run_retrieval_suite(
                            cases, retriever, eval_settings, eval_settings.chat_model
                        )
                    )
                elif suite == "rag-qa":
                    results.append(
                        await run_rag_qa_suite(
                            cases,
                            chat,
                            retriever,
                            judge,
                            eval_settings,
                            eval_settings.chat_model,
                        )
                    )
        finally:
            await provider.aclose()
            await session_store.close()
            rag_store.close()

        return results
