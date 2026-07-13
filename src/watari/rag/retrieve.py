"""Retrieval: embed the query, then hybrid-search the store."""

from __future__ import annotations

from typing import Protocol

from watari.config import Settings
from watari.rag.embeddings import FastEmbedEmbedder
from watari.rag.models import RetrievedChunk
from watari.rag.store import RagStore


class RetrieverProtocol(Protocol):
    """The seam ChatService depends on, so it can be faked in tests."""

    async def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]: ...


class Retriever:
    def __init__(self, store: RagStore, embedder: FastEmbedEmbedder, settings: Settings) -> None:
        self._store = store
        self._embedder = embedder
        self._settings = settings

    async def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or self._settings.retrieval_top_k
        query_embedding = await self._embedder.aembed_query(query)
        return await self._store.ahybrid_search(query, query_embedding, top_k=k)
