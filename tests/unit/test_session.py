"""Session persistence and the migration runner."""

from __future__ import annotations

from watari.core.models import ChatMessage, Role
from watari.core.session import SCHEMA_VERSION, SessionStore


async def test_create_and_roundtrip_messages(store: SessionStore) -> None:
    session_id = await store.create_session(title="test")
    assert await store.session_exists(session_id)

    await store.add_message(session_id, ChatMessage(role=Role.USER, content="hi"))
    await store.add_message(session_id, ChatMessage(role=Role.ASSISTANT, content="hello"))

    history = await store.get_history(session_id)
    assert [(m.role, m.content) for m in history] == [
        (Role.USER, "hi"),
        (Role.ASSISTANT, "hello"),
    ]


async def test_unknown_session_reports_missing(store: SessionStore) -> None:
    assert not await store.session_exists("does-not-exist")


async def test_migration_sets_schema_version(store: SessionStore) -> None:
    cur = await store._db.execute("SELECT version FROM schema_version")
    row = await cur.fetchone()
    assert row is not None
    assert row["version"] == SCHEMA_VERSION


async def test_reconnect_is_idempotent(store: SessionStore) -> None:
    # Re-running connect()/_migrate() must not error or duplicate version rows.
    await store._migrate()
    cur = await store._db.execute("SELECT COUNT(*) AS n FROM schema_version")
    row = await cur.fetchone()
    assert row is not None
    assert row["n"] == 1
