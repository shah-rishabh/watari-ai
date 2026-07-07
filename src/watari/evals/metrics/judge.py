"""LLM-as-judge metrics for generation quality.

Two metrics, both designed to be reliable on a *small local* judge model:

- **faithfulness** — decompose the answer into atomic claims, then judge each
  claim against the retrieved context (is it supported?). Faithfulness is the
  fraction of claims that are grounded. Decomposition + per-claim binary
  judgments are far more reliable on a 3B model than asking for a single 1-10
  score, because each sub-judgment is a simple yes/no.
- **answer_relevance** — a single 3-point rubric (0 / 0.5 / 1) on whether the
  answer actually addresses the question.

Judge prompts are versioned here in code and calibrated against human labels
(see ``evals/datasets/judge_calibration_v1.jsonl`` and ``docs/evals.md``). We
force low temperature and parse a small JSON object; malformed output is retried
once, then scored as the conservative value.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import cast

from watari.core.llm import LLMProvider
from watari.core.models import ChatMessage, Role
from watari.obs.logging import get_logger

logger = get_logger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_DECOMPOSE_PROMPT = """Break the ANSWER into a list of atomic factual claims.
Each claim must be a single, self-contained statement. Ignore hedges, opinions,
and filler. Respond with JSON only: {{"claims": ["claim 1", "claim 2"]}}.

ANSWER:
{answer}"""

_FAITHFUL_PROMPT = """You are checking whether a CLAIM is supported by CONTEXT.
Answer "yes" only if the CONTEXT directly supports the CLAIM. If the CONTEXT is
silent or contradicts it, answer "no". Respond with JSON only:
{{"supported": "yes"}} or {{"supported": "no"}}.

CONTEXT:
{context}

CLAIM:
{claim}"""

_RELEVANCE_PROMPT = """Rate how well the ANSWER addresses the QUESTION, ignoring
whether it is factually correct. Use exactly one of:
2 = fully addresses the question,
1 = partially addresses it,
0 = does not address it.
Respond with JSON only: {{"score": 2}}.

QUESTION:
{question}

ANSWER:
{answer}"""


@dataclass
class FaithfulnessResult:
    n_claims: int
    n_supported: int

    @property
    def score(self) -> float:
        if self.n_claims == 0:
            return 1.0
        return self.n_supported / self.n_claims


def _parse_json(text: str) -> dict[str, object] | None:
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        obj: object = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    return {str(k): v for k, v in obj.items()}  # type: ignore[misc]


class Judge:
    """Runs judge prompts through an LLMProvider."""

    def __init__(self, provider: LLMProvider, model: str) -> None:
        self._provider = provider
        self._model = model

    async def _complete(self, prompt: str) -> str:
        messages = [ChatMessage(role=Role.USER, content=prompt)]
        parts: list[str] = []
        async for delta in self._provider.stream(messages, model=self._model, temperature=0.0):
            if delta.content:
                parts.append(delta.content)
        return "".join(parts)

    async def _complete_json(self, prompt: str) -> dict[str, object] | None:
        for _ in range(2):  # one retry on malformed output
            obj = _parse_json(await self._complete(prompt))
            if obj is not None:
                return obj
        return None

    async def decompose_claims(self, answer: str) -> list[str]:
        obj = await self._complete_json(_DECOMPOSE_PROMPT.format(answer=answer))
        if obj is None:
            return []
        claims = obj.get("claims")
        if not isinstance(claims, list):
            return []
        items = cast("list[object]", claims)
        return [str(c) for c in items if str(c).strip()]

    async def claim_supported(self, claim: str, context: str) -> bool:
        obj = await self._complete_json(_FAITHFUL_PROMPT.format(context=context, claim=claim))
        if obj is None:
            return False  # conservative: unparseable == unsupported
        return str(obj.get("supported", "")).strip().lower() == "yes"

    async def faithfulness(self, answer: str, context: str) -> FaithfulnessResult:
        claims = await self.decompose_claims(answer)
        supported = 0
        for claim in claims:
            if await self.claim_supported(claim, context):
                supported += 1
        return FaithfulnessResult(n_claims=len(claims), n_supported=supported)

    async def answer_relevance(self, question: str, answer: str) -> float:
        obj = await self._complete_json(_RELEVANCE_PROMPT.format(question=question, answer=answer))
        if obj is None:
            return 0.0
        try:
            raw = int(obj.get("score", 0))  # type: ignore[arg-type]
        except (ValueError, TypeError):
            return 0.0
        return max(0.0, min(1.0, raw / 2.0))
