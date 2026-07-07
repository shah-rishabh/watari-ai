"""Memory orchestration: remember (extract→embed→store) and recall."""

from __future__ import annotations

from watari.config import Settings
from watari.core.models import ChatMessage
from watari.memory.extract import FactExtractor
from watari.memory.models import Fact, RecalledMemory, StoredMemory
from watari.memory.store import MemoryStore
from watari.obs.logging import get_logger
from watari.rag.embeddings import FastEmbedEmbedder

logger = get_logger(__name__)


class MemoryService:
    def __init__(
        self,
        store: MemoryStore,
        embedder: FastEmbedEmbedder,
        extractor: FactExtractor,
        settings: Settings,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._extractor = extractor
        self._settings = settings

    async def remember_fact(self, fact: Fact, *, source: str | None = None) -> int:
        """Embed and store a single fact (dedup handled by the store)."""
        embedding = await self._embedder.aembed_query(fact.fact)
        return await self._store.aadd(fact, embedding, source=source)

    async def remember_from_transcript(
        self, messages: list[ChatMessage], *, source: str | None = None
    ) -> list[Fact]:
        """Extract facts from a transcript and store the novel ones."""
        transcript = "\n".join(f"{m.role.value}: {m.content}" for m in messages)
        facts = await self._extractor.extract(transcript)
        for fact in facts:
            await self.remember_fact(fact, source=source)
        logger.info("memory.remembered", count=len(facts), source=source)
        return facts

    async def recall(self, query: str) -> list[RecalledMemory]:
        embedding = await self._embedder.aembed_query(query)
        return await self._store.arecall(embedding, top_k=self._settings.memory_recall_top_k)

    def list_active(self) -> list[StoredMemory]:
        return self._store.list_active()

    def forget(self, memory_id: int) -> bool:
        return self._store.forget(memory_id)

    def wipe(self) -> int:
        return self._store.wipe()


def format_memory_block(memories: list[RecalledMemory]) -> str:
    """Render recalled facts as a labeled context section."""
    if not memories:
        return ""
    lines = ["Relevant things you remember about the user:"]
    lines.extend(f"- {m.fact}" for m in memories)
    return "\n".join(lines)
