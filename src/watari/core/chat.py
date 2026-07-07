"""Chat orchestration shared by the CLI and the API.

This service is the single place that turns "a user message in a session" into
"a streamed assistant reply, persisted". Both surfaces (typer CLI, FastAPI SSE)
consume the same :meth:`ChatService.stream_reply` async iterator, proving the
thin-adapter design.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from watari.config import Settings
from watari.core.context import assemble_context
from watari.core.llm import LLMProvider
from watari.core.models import ChatDelta, ChatMessage, Role
from watari.core.session import SessionStore
from watari.obs.logging import get_logger

logger = get_logger(__name__)


class ChatService:
    def __init__(
        self,
        provider: LLMProvider,
        store: SessionStore,
        settings: Settings,
    ) -> None:
        self._provider = provider
        self._store = store
        self._settings = settings

    async def stream_reply(self, session_id: str, user_text: str) -> AsyncIterator[ChatDelta]:
        """Persist the user turn, stream the reply, then persist the reply."""
        user_msg = ChatMessage(role=Role.USER, content=user_text)
        await self._store.add_message(session_id, user_msg)

        history = await self._store.get_history(session_id)
        messages = assemble_context(
            history,
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
            yield delta

        reply_text = "".join(parts)
        await self._store.add_message(
            session_id, ChatMessage(role=Role.ASSISTANT, content=reply_text)
        )
        logger.info(
            "chat.reply_complete",
            session_id=session_id,
            reply_chars=len(reply_text),
            prompt_tokens=final_usage.prompt_tokens if final_usage else None,
            completion_tokens=final_usage.completion_tokens if final_usage else None,
        )
