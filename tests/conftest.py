"""Shared test fixtures."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from watari.config import Settings
from watari.core.models import ChatDelta, ChatMessage, Usage
from watari.core.session import SessionStore


class FakeProvider:
    """An :class:`LLMProvider` that echoes a scripted reply word-by-word.

    Records the messages it was called with so tests can assert on the assembled
    context without a live model server.
    """

    def __init__(self, reply: str = "hello there", reasoning: str = "") -> None:
        self.reply = reply
        self.reasoning = reasoning
        self.calls: list[list[ChatMessage]] = []

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ChatDelta]:
        self.calls.append(list(messages))
        for word in self.reasoning.split():
            yield ChatDelta(reasoning=word + " ")
        for word in self.reply.split():
            yield ChatDelta(content=word + " ")
        yield ChatDelta(done=True, usage=Usage(prompt_tokens=10, completion_tokens=2))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path, max_context_tokens=1024, max_response_tokens=128)


@pytest.fixture
async def store(settings: Settings) -> AsyncIterator[SessionStore]:
    s = SessionStore(settings.db_path)
    await s.connect()
    try:
        yield s
    finally:
        await s.close()


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()
