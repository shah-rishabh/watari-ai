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

from collections.abc import AsyncIterator

from watari.config import Settings
from watari.core.context import assemble_context
from watari.core.llm import LLMProvider
from watari.core.models import ChatDelta, ChatMessage, Role
from watari.core.session import SessionStore
from watari.obs.logging import get_logger
from watari.rag.cite import (
    format_context_block,
    render_sources,
    strip_invalid_citations,
    validate_citations,
)
from watari.rag.retrieve import RetrieverProtocol

logger = get_logger(__name__)

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
    ) -> None:
        self._provider = provider
        self._store = store
        self._settings = settings
        self._retriever = retriever

    async def stream_reply(
        self, session_id: str, user_text: str, *, use_rag: bool = False
    ) -> AsyncIterator[ChatDelta]:
        """Persist the user turn, stream the reply, then persist the reply."""
        user_msg = ChatMessage(role=Role.USER, content=user_text)
        await self._store.add_message(session_id, user_msg)

        context_block = None
        chunks = []
        if use_rag and self._retriever is not None:
            chunks = await self._retriever.retrieve(user_text)
            if chunks:
                context_block = f"{_CITE_INSTRUCTION}\n\n{format_context_block(chunks)}"

        history = await self._store.get_history(session_id)
        messages = assemble_context(
            history,
            context_block=context_block,
            max_context_tokens=self._settings.max_context_tokens,
            reserved_response_tokens=self._settings.max_response_tokens,
        )

        parts: list[str] = []
        final_usage = None
        async for delta in self._provider.stream(messages):
            if delta.content:
                parts.append(delta.content)
            if delta.done:
                final_usage = delta.usage
                continue
            yield delta

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
