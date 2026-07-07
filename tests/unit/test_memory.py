"""Memory store, extraction parsing, and chat wiring."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path

import pytest

from watari.config import Settings
from watari.core.models import ChatDelta, ChatMessage, Usage
from watari.memory.extract import FactExtractor
from watari.memory.models import Category, Fact
from watari.memory.store import MemoryStore


@pytest.fixture(scope="module")
def _embedder():  # type: ignore[no-untyped-def]
    from watari.rag.embeddings import FastEmbedEmbedder

    return FastEmbedEmbedder(Settings())


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(Settings(data_dir=tmp_path))
    s.connect()
    try:
        yield s
    finally:
        s.close()


def _add(store: MemoryStore, embedder, text: str, cat=Category.OTHER) -> int:  # type: ignore[no-untyped-def]
    return store.add(Fact(fact=text, category=cat), embedder.embed_query(text), source="t")


class TestMemoryStore:
    def test_add_and_list(self, store: MemoryStore, _embedder) -> None:  # type: ignore[no-untyped-def]
        _add(store, _embedder, "The user likes tea.")
        active = store.list_active()
        assert len(active) == 1
        assert active[0].fact == "The user likes tea."

    def test_near_identical_fact_supersedes(self, store: MemoryStore, _embedder) -> None:  # type: ignore[no-untyped-def]
        _add(store, _embedder, "The user prefers morning meetings.")
        _add(store, _embedder, "The user prefers morning meetings")  # no period
        # Superseded, not duplicated.
        assert len(store.list_active()) == 1

    def test_distinct_facts_coexist(self, store: MemoryStore, _embedder) -> None:  # type: ignore[no-untyped-def]
        _add(store, _embedder, "The user is allergic to peanuts.")
        _add(store, _embedder, "The user works on a drone project.")
        assert len(store.list_active()) == 2

    def test_recall_surfaces_relevant_fact(self, store: MemoryStore, _embedder) -> None:  # type: ignore[no-untyped-def]
        _add(store, _embedder, "The user is lactose intolerant.")
        _add(store, _embedder, "The user drives an electric car.")
        recalled = store.recall(_embedder.embed_query("dietary restrictions"), top_k=1)
        assert recalled
        assert "lactose" in recalled[0].fact.lower()

    def test_forget_deactivates(self, store: MemoryStore, _embedder) -> None:  # type: ignore[no-untyped-def]
        mid = _add(store, _embedder, "temporary fact")
        assert store.forget(mid) is True
        assert store.list_active() == []
        assert store.forget(mid) is False  # already inactive

    def test_wipe_clears_all(self, store: MemoryStore, _embedder) -> None:  # type: ignore[no-untyped-def]
        _add(store, _embedder, "fact one")
        _add(store, _embedder, "fact two")
        assert store.wipe() == 2
        assert store.list_active() == []


class ScriptedProvider:
    def __init__(self, response: str) -> None:
        self._response = response

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ChatDelta]:
        yield ChatDelta(content=self._response)
        yield ChatDelta(done=True, usage=Usage())


class TestExtraction:
    async def test_parses_facts(self) -> None:
        provider = ScriptedProvider(
            '{"facts": [{"fact": "The user likes tea", "category": "preference", '
            '"confidence": 0.9}]}'
        )
        facts = await FactExtractor(provider, "m").extract("user: I love tea")
        assert len(facts) == 1
        assert facts[0].category is Category.PREFERENCE

    async def test_empty_facts_ok(self) -> None:
        provider = ScriptedProvider('{"facts": []}')
        assert await FactExtractor(provider, "m").extract("user: hi") == []

    async def test_malformed_json_returns_empty(self) -> None:
        provider = ScriptedProvider("not json at all")
        assert await FactExtractor(provider, "m").extract("user: hi") == []

    async def test_unknown_category_falls_back_to_other(self) -> None:
        provider = ScriptedProvider(
            '{"facts": [{"fact": "x", "category": "bogus", "confidence": 1}]}'
        )
        facts = await FactExtractor(provider, "m").extract("t")
        assert facts[0].category is Category.OTHER
