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
from watari.rag.embeddings import FastEmbedEmbedder
from watari.rag.retrieve import Retriever
from watari.rag.service import IngestService
from watari.rag.store import RagStore


@dataclass
class AppState:
    settings: Settings
    provider: OpenAICompatibleProvider
    store: SessionStore
    chat: ChatService
    rag_store: RagStore
    embedder: FastEmbedEmbedder
    retriever: Retriever
    ingest: IngestService


async def build_state(settings: Settings) -> AppState:
    provider = OpenAICompatibleProvider(settings)
    store = SessionStore(settings.db_path)
    await store.connect()

    rag_store = RagStore(settings)
    await rag_store.aconnect()
    embedder = FastEmbedEmbedder(settings)
    retriever = Retriever(rag_store, embedder, settings)
    ingest = IngestService(rag_store, embedder, settings)

    chat = ChatService(provider, store, settings, retriever=retriever)
    return AppState(
        settings=settings,
        provider=provider,
        store=store,
        chat=chat,
        rag_store=rag_store,
        embedder=embedder,
        retriever=retriever,
        ingest=ingest,
    )


async def teardown_state(state: AppState) -> None:
    await state.provider.aclose()
    await state.store.close()
    await state.rag_store.aclose()


def get_state(request: Request) -> AppState:
    return request.app.state.watari
