"""Chat orchestration shared by the CLI and the API.

This service is the single place that turns "a user message in a session" into
"a streamed assistant reply, persisted". Both surfaces (typer CLI, FastAPI SSE)
consume the same :meth:`ChatService.stream_reply` async iterator, proving the
thin-adapter design.

When a retriever is configured and RAG is enabled for the turn, the service
retrieves chunks, injects them as numbered context with citation instructions,
and appends a validated sources footnote to the persisted answer.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Protocol

from watari.config import Settings
from watari.core.context import assemble_context
from watari.core.llm import LLMProvider
from watari.core.models import ChatDelta, ChatMessage, Role
from watari.core.session import SessionStore
from watari.memory.models import RecalledMemory
from watari.obs.logging import get_logger
from watari.obs.metrics import METRICS, Metrics
from watari.rag.cite import (
    format_context_block,
    render_sources,
    strip_invalid_citations,
    validate_citations,
)
from watari.rag.retrieve import RetrieverProtocol

logger = get_logger(__name__)


class MemoryRecaller(Protocol):
    """The recall seam ChatService depends on (faked in tests)."""

    async def recall(self, query: str) -> list[RecalledMemory]: ...


_CITE_INSTRUCTION = (
    "Answer using only the retrieved context below. Cite every claim with the "
    "corresponding [n] marker. If the context does not contain the answer, say "
    "so plainly instead of guessing."
)


class ChatService:
    def __init__(
        self,
        provider: LLMProvider,
        store: SessionStore,
        settings: Settings,
        retriever: RetrieverProtocol | None = None,
        memory: MemoryRecaller | None = None,
        metrics: Metrics | None = None,
    ) -> None:
        self._provider = provider
        self._store = store
        self._settings = settings
        self._retriever = retriever
        self._memory = memory
        self._metrics = metrics or METRICS

    async def create_session(self, title: str | None = None) -> str:
        return await self._store.create_session(title)

    async def stream_reply(
        self,
        session_id: str,
        user_text: str,
        *,
        use_rag: bool = False,
        use_memory: bool = True,
    ) -> AsyncIterator[ChatDelta]:
        """Persist the user turn, stream the reply, then persist the reply."""
        user_msg = ChatMessage(role=Role.USER, content=user_text)
        await self._store.add_message(session_id, user_msg)

        blocks: list[str] = []

        # Long-term memory recall (on by default; independent of RAG).
        if use_memory and self._memory is not None:
            from watari.memory.service import format_memory_block

            recalled = await self._memory.recall(user_text)
            memory_block = format_memory_block(recalled)
            if memory_block:
                blocks.append(memory_block)

        chunks = []
        if use_rag and self._retriever is not None:
            chunks = await self._retriever.retrieve(user_text)
            if chunks:
                blocks.append(f"{_CITE_INSTRUCTION}\n\n{format_context_block(chunks)}")

        context_block = "\n\n".join(blocks) if blocks else None
        history = await self._store.get_history(session_id)
        messages = assemble_context(
            history,
            context_block=context_block,
            max_context_tokens=self._settings.max_context_tokens,
            reserved_response_tokens=self._settings.max_response_tokens,
        )

        parts: list[str] = []
        final_usage = None
        start = time.perf_counter()
        ttft_ms: float | None = None
        async for delta in self._provider.stream(messages):
            if delta.content:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - start) * 1000.0
                    self._metrics.observe("ttft_ms", ttft_ms)
                parts.append(delta.content)
            if delta.done:
                final_usage = delta.usage
                continue
            yield delta

        self._metrics.observe("reply_latency_ms", (time.perf_counter() - start) * 1000.0)
        if final_usage is not None:
            self._metrics.record_usage(final_usage.prompt_tokens, final_usage.completion_tokens)

        answer = "".join(parts)
        # Validate citations, strip hallucinated markers, append a sources block.
        if chunks:
            valid, hallucinated = validate_citations(answer, len(chunks))
            answer = strip_invalid_citations(answer, len(chunks))
            sources = render_sources(answer, chunks)
            if sources:
                answer = answer.rstrip() + "\n" + sources
                # Emit the footnote to the live stream too.
                yield ChatDelta(content="\n" + sources)
            logger.info(
                "chat.citations",
                session_id=session_id,
                cited_valid=len(valid),
                cited_hallucinated=len(hallucinated),
                chunks_retrieved=len(chunks),
            )

        # Re-emit the terminal delta (with usage) after any appended sources.
        yield ChatDelta(done=True, usage=final_usage)

        await self._store.add_message(session_id, ChatMessage(role=Role.ASSISTANT, content=answer))
        logger.info(
            "chat.reply_complete",
            session_id=session_id,
            reply_chars=len(answer),
            rag=bool(chunks),
            prompt_tokens=final_usage.prompt_tokens if final_usage else None,
            completion_tokens=final_usage.completion_tokens if final_usage else None,
        )
