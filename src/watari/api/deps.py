"""Shared application state and FastAPI dependencies.

The provider, session store, and chat service are constructed once at startup
and stashed on ``app.state``; request handlers pull them via these accessors.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from watari.config import Settings
from watari.core.chat import ChatService
from watari.core.llm import OpenAICompatibleProvider
from watari.core.session import SessionStore


@dataclass
class AppState:
    settings: Settings
    provider: OpenAICompatibleProvider
    store: SessionStore
    chat: ChatService


async def build_state(settings: Settings) -> AppState:
    provider = OpenAICompatibleProvider(settings)
    store = SessionStore(settings.db_path)
    await store.connect()
    chat = ChatService(provider, store, settings)
    return AppState(settings=settings, provider=provider, store=store, chat=chat)


async def teardown_state(state: AppState) -> None:
    await state.provider.aclose()
    await state.store.close()


def get_state(request: Request) -> AppState:
    return request.app.state.watari
