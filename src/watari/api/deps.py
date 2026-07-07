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
from watari.memory.extract import FactExtractor
from watari.memory.service import MemoryService
from watari.memory.store import MemoryStore
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
    memory_store: MemoryStore
    memory: MemoryService


async def build_state(settings: Settings) -> AppState:
    provider = OpenAICompatibleProvider(settings)
    store = SessionStore(settings.db_path)
    await store.connect()

    embedder = FastEmbedEmbedder(settings)

    rag_store = RagStore(settings)
    await rag_store.aconnect()
    retriever = Retriever(rag_store, embedder, settings)
    ingest = IngestService(rag_store, embedder, settings)

    memory_store = MemoryStore(settings)
    await memory_store.aconnect()
    extractor = FactExtractor(provider, settings.extract_model)
    memory = MemoryService(memory_store, embedder, extractor, settings)

    chat = ChatService(provider, store, settings, retriever=retriever, memory=memory)
    return AppState(
        settings=settings,
        provider=provider,
        store=store,
        chat=chat,
        rag_store=rag_store,
        embedder=embedder,
        retriever=retriever,
        ingest=ingest,
        memory_store=memory_store,
        memory=memory,
    )


async def teardown_state(state: AppState) -> None:
    await state.provider.aclose()
    await state.store.close()
    await state.rag_store.aclose()
    await state.memory_store.aclose()


def get_state(request: Request) -> AppState:
    return request.app.state.watari
