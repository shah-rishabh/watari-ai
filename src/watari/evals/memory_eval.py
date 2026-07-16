"""Memory eval suite — extraction quality and recall-in-context.

Two metrics:
- **extraction_recall**: fraction of golden facts that the extractor captured
  (matched by keyword against the extracted facts). Extraction is fuzzy, so a
  substring/keyword match is the pragmatic golden check.
- **recall_in_context**: given the extracted facts stored, does querying the
  store surface a fact containing the expected keyword? This tests the full
  extract -> embed -> store -> recall path end to end.

Each case runs in an isolated store so facts don't bleed across cases.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic import BaseModel

from watari.config import Settings
from watari.core.llm import OpenAICompatibleProvider
from watari.core.models import ChatMessage, Role
from watari.evals.models import MetricResult, SuiteResult
from watari.memory.extract import FactExtractor
from watari.memory.service import MemoryService
from watari.memory.store import MemoryStore
from watari.rag.embeddings import FastEmbedEmbedder


class MemoryCase(BaseModel):
    id: str
    transcript: str
    expected_facts: list[str] = []
    recall_query: str
    recall_expect: str
    tags: list[str] = []


def load_memory(path: Path, *, smoke_only: bool = False) -> list[MemoryCase]:
    cases: list[MemoryCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = MemoryCase.model_validate_json(line)
        if smoke_only and "smoke" not in case.tags:
            continue
        cases.append(case)
    return cases


def _transcript_to_messages(transcript: str) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    for line in transcript.splitlines():
        if line.startswith("user:"):
            messages.append(ChatMessage(role=Role.USER, content=line[5:].strip()))
        elif line.startswith("assistant:"):
            messages.append(ChatMessage(role=Role.ASSISTANT, content=line[10:].strip()))
    return messages


def _keyword(expected: str) -> str:
    # Use the most distinctive token of the expected fact for fuzzy matching.
    tokens = [t for t in expected.split() if len(t) > 3]
    return (tokens[-1] if tokens else expected).lower()


async def run_memory_suite(cases: list[MemoryCase], *, settings: Settings) -> SuiteResult:
    extraction_hits = 0
    extraction_total = 0
    recall_hits = 0

    for case in cases:
        with tempfile.TemporaryDirectory(prefix="watari-mem-eval-") as tmp:
            case_settings = settings.model_copy(update={"data_dir": Path(tmp)})
            provider = OpenAICompatibleProvider(case_settings)
            store = MemoryStore(case_settings)
            store.connect()
            embedder = FastEmbedEmbedder(case_settings)
            extractor = FactExtractor(provider, case_settings.extract_model)
            service = MemoryService(store, embedder, extractor, case_settings)

            try:
                messages = _transcript_to_messages(case.transcript)
                facts = await service.remember_from_transcript(messages, source=case.id)
                extracted_text = " ".join(f.fact.lower() for f in facts)

                for expected in case.expected_facts:
                    extraction_total += 1
                    if _keyword(expected) in extracted_text:
                        extraction_hits += 1

                recalled = await service.recall(case.recall_query)
                recalled_text = " ".join(m.fact.lower() for m in recalled)
                if case.recall_expect.lower() in recalled_text:
                    recall_hits += 1
            finally:
                await provider.aclose()
                store.close()

    n = len(cases) or 1
    extraction_recall = extraction_hits / (extraction_total or 1)
    metrics = [
        MetricResult(name="extraction_recall", value=extraction_recall, n=extraction_total),
        MetricResult(name="recall_in_context", value=recall_hits / n, n=len(cases)),
    ]
    return SuiteResult(
        suite="memory", model=settings.chat_model, n_cases=len(cases), metrics=metrics
    )
