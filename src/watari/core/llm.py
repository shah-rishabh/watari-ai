"""The LLM provider seam.

We use the official ``openai`` async client pointed at any OpenAI-compatible
endpoint (Ollama's ``/v1`` on a laptop, a 0.5B model in CI, or a hosted API).
The :class:`LLMProvider` Protocol is the seam that keeps ``openai`` types out
of the rest of the codebase and gives us a single place to attach metrics and a
single thing to mock in tests. See ``docs/adr/000-provider-abstraction.md``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
)

from watari.config import Settings
from watari.core.models import ChatDelta, ChatMessage, Usage


class LLMProvider(Protocol):
    """A minimal streaming chat interface.

    Any implementation yields :class:`ChatDelta` chunks and emits a final delta
    with ``done=True`` and populated ``usage``.
    """

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ChatDelta]:
        """Stream a completion for ``messages``.

        Implementations are async generators, so the *call* returns an
        ``AsyncIterator`` directly (not a coroutine wrapping one).
        """
        ...


class OpenAICompatibleProvider:
    """:class:`LLMProvider` backed by an OpenAI-compatible HTTP endpoint."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.request_timeout_s,
        )

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ChatDelta]:
        wire_messages: list[ChatCompletionMessageParam] = [
            {"role": m.role.value, "content": m.content}  # type: ignore[misc]
            for m in messages
        ]

        stream = await self._client.chat.completions.create(
            model=model or self._settings.chat_model,
            messages=wire_messages,
            temperature=(temperature if temperature is not None else self._settings.temperature),
            max_tokens=max_tokens or self._settings.max_response_tokens,
            stream=True,
            stream_options={"include_usage": True},
            # Control reasoning-model "thinking". "none" disables it for a snappy
            # assistant; servers without reasoning support ignore this field.
            # Passed via extra_body so the SDK's typed return isn't widened.
            extra_body={"reasoning_effort": self._settings.reasoning_effort},
        )

        usage: Usage | None = None
        chunk: ChatCompletionChunk
        async for chunk in stream:
            if chunk.usage is not None:
                usage = Usage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                )
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            # Reasoning models stream their chain of thought on a separate
            # `reasoning` field (not part of the typed SDK model); surface it
            # distinctly so it is shown/logged but never mixed into the answer.
            reasoning = getattr(delta, "reasoning", None) or ""
            if reasoning:
                yield ChatDelta(reasoning=reasoning)
            piece = delta.content or ""
            if piece:
                yield ChatDelta(content=piece)

        yield ChatDelta(done=True, usage=usage or Usage())

    async def aclose(self) -> None:
        await self._client.close()
