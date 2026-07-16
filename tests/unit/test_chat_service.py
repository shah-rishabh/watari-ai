"""ChatService orchestration against a fake provider."""

from __future__ import annotations

from tests.conftest import FakeProvider, FakeRetriever, make_chunks
from watari.config import Settings
from watari.core.chat import ChatService
from watari.core.models import Role
from watari.core.session import SessionStore
from watari.memory.models import Category, RecalledMemory


class FakeRecaller:
    def __init__(self, facts: list[str]) -> None:
        self._facts = facts

    async def recall(self, query: str) -> list[RecalledMemory]:
        return [
            RecalledMemory(id=i, fact=f, category=Category.OTHER, score=0.9)
            for i, f in enumerate(self._facts, start=1)
        ]


async def test_stream_reply_persists_both_turns(
    store: SessionStore, settings: Settings, fake_provider: FakeProvider
) -> None:
    service = ChatService(fake_provider, store, settings)
    session_id = await store.create_session()

    deltas = [d async for d in service.stream_reply(session_id, "hi")]
    text = "".join(d.content for d in deltas)
    assert "hello there" in text
    assert deltas[-1].done

    history = await store.get_history(session_id)
    assert history[0].role is Role.USER
    assert history[0].content == "hi"
    assert history[1].role is Role.ASSISTANT
    assert "hello there" in history[1].content


async def test_context_includes_system_prompt(
    store: SessionStore, settings: Settings, fake_provider: FakeProvider
) -> None:
    service = ChatService(fake_provider, store, settings)
    session_id = await store.create_session()

    _ = [d async for d in service.stream_reply(session_id, "hi")]

    sent = fake_provider.calls[0]
    assert sent[0].role is Role.SYSTEM
    assert sent[-1].content == "hi"


async def test_reasoning_is_not_persisted_into_the_answer(
    store: SessionStore, settings: Settings
) -> None:
    # A reasoning model streams thinking on the `reasoning` field; only the
    # visible answer must be saved as the assistant turn.
    provider = FakeProvider(reply="the answer", reasoning="secret chain of thought")
    service = ChatService(provider, store, settings)
    session_id = await store.create_session()

    deltas = [d async for d in service.stream_reply(session_id, "hi")]
    assert any(d.reasoning for d in deltas)

    history = await store.get_history(session_id)
    assert history[1].role is Role.ASSISTANT
    assert history[1].content.strip() == "the answer"
    assert "secret" not in history[1].content


async def test_rag_injects_context_and_appends_sources(
    store: SessionStore, settings: Settings
) -> None:
    provider = FakeProvider(reply="grounded reply [1]")
    retriever = FakeRetriever(make_chunks(2))
    service = ChatService(provider, store, settings, retriever=retriever)
    session_id = await store.create_session()

    deltas = [d async for d in service.stream_reply(session_id, "q", use_rag=True)]
    text = "".join(d.content for d in deltas)
    # The numbered context reached the model.
    sent = provider.calls[0]
    assert "[1] (doc1.md" in sent[0].content
    # A sources footnote was appended and persisted.
    assert "Sources:" in text
    history = await store.get_history(session_id)
    assert "Sources:" in history[1].content
    assert "[1] doc1.md" in history[1].content


async def test_rag_strips_hallucinated_citation(store: SessionStore, settings: Settings) -> None:
    # Model cites [5] but only 2 chunks were provided.
    provider = FakeProvider(reply="claim [5] here")
    retriever = FakeRetriever(make_chunks(2))
    service = ChatService(provider, store, settings, retriever=retriever)
    session_id = await store.create_session()

    _ = [d async for d in service.stream_reply(session_id, "q", use_rag=True)]
    history = await store.get_history(session_id)
    # The out-of-range marker is stripped; no bogus source list.
    assert "[5]" not in history[1].content
    assert "Sources:" not in history[1].content


async def test_rag_off_by_default_does_not_retrieve(
    store: SessionStore, settings: Settings
) -> None:
    provider = FakeProvider(reply="plain reply")
    retriever = FakeRetriever(make_chunks(2))
    service = ChatService(provider, store, settings, retriever=retriever)
    session_id = await store.create_session()

    _ = [d async for d in service.stream_reply(session_id, "q")]  # use_rag defaults False
    sent = provider.calls[0]
    assert "doc1.md" not in sent[0].content


async def test_memory_facts_injected_into_context(store: SessionStore, settings: Settings) -> None:
    provider = FakeProvider(reply="ok")
    recaller = FakeRecaller(["The user is allergic to peanuts."])
    service = ChatService(provider, store, settings, memory=recaller)
    session_id = await store.create_session()

    _ = [d async for d in service.stream_reply(session_id, "what should I avoid?")]
    system_content = provider.calls[0][0].content
    assert "allergic to peanuts" in system_content


async def test_memory_can_be_disabled_per_turn(store: SessionStore, settings: Settings) -> None:
    provider = FakeProvider(reply="ok")
    recaller = FakeRecaller(["The user likes tea."])
    service = ChatService(provider, store, settings, memory=recaller)
    session_id = await store.create_session()

    _ = [d async for d in service.stream_reply(session_id, "q", use_memory=False)]
    assert "tea" not in provider.calls[0][0].content
