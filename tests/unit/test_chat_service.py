"""ChatService orchestration against a fake provider."""

from __future__ import annotations

from tests.conftest import FakeProvider
from watari.config import Settings
from watari.core.chat import ChatService
from watari.core.models import Role
from watari.core.session import SessionStore


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
