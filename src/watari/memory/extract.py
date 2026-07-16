"""Fact extraction from a conversation transcript.

A constrained LLM call turns a transcript into atomic, durable facts about the
user (preferences, biographical details, projects) as validated JSON. Ephemeral
chit-chat and the assistant's own statements are ignored. Malformed output is
retried once, then dropped — never crashes the caller.
"""

from __future__ import annotations

import json
import re
from typing import cast

from watari.core.llm import LLMProvider
from watari.core.models import ChatMessage, Role
from watari.memory.models import Category, Fact
from watari.obs.logging import get_logger

logger = get_logger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_EXTRACT_PROMPT = """Extract durable facts about the USER from the transcript.
Include only stable facts worth remembering across sessions: preferences,
biographical details, and ongoing projects. Ignore small talk, one-off requests,
and anything the assistant said about itself.

Each fact must be atomic (one statement) and phrased about the user. Categorise
each as one of: preference, biographical, project, other.

Respond with JSON only:
{{"facts": [{{"fact": "...", "category": "preference", "confidence": 0.9}}]}}
If there is nothing worth remembering, respond with {{"facts": []}}.

TRANSCRIPT:
{transcript}"""


def _parse(text: str) -> dict[str, object] | None:
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


class FactExtractor:
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

    async def extract(self, transcript: str) -> list[Fact]:
        prompt = _EXTRACT_PROMPT.format(transcript=transcript)
        for _ in range(2):  # one retry on malformed output
            obj = _parse(await self._complete(prompt))
            if obj is not None:
                return self._coerce(obj)
        logger.warning("memory.extract_failed")
        return []

    @staticmethod
    def _coerce(obj: dict[str, object]) -> list[Fact]:
        raw = obj.get("facts")
        if not isinstance(raw, list):
            return []
        items = cast("list[object]", raw)
        facts: list[Fact] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            entry = cast("dict[str, object]", item)
            text = str(entry.get("fact", "")).strip()
            if not text:
                continue
            category = _coerce_category(str(entry.get("category", "other")))
            confidence = _coerce_confidence(entry.get("confidence", 1.0))
            facts.append(Fact(fact=text, category=category, confidence=confidence))
        return facts


def _coerce_category(value: str) -> Category:
    try:
        return Category(value.strip().lower())
    except ValueError:
        return Category.OTHER


def _coerce_confidence(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1.0
