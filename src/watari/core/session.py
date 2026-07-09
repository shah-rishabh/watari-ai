"""Conversation session persistence on SQLite (via aiosqlite).

A deliberately tiny migration runner keeps a single ``schema_version`` row and
applies ``schema.sql`` on startup. For a single-user local app this beats
pulling in Alembic; the schema is append-only and idempotent.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from watari.core.models import ChatMessage, Role

SCHEMA_VERSION = 1
_SCHEMA_SQL = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def new_session_id() -> str:
    return uuid.uuid4().hex


class SessionStore:
    """Async CRUD over sessions and their messages."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    @property
    def _db(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("SessionStore used before connect()")
        return self._conn

    async def connect(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA foreign_keys = ON")
        await self._migrate()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def _migrate(self) -> None:
        await self._db.executescript(_SCHEMA_SQL)
        cur = await self._db.execute("SELECT version FROM schema_version LIMIT 1")
        row = await cur.fetchone()
        if row is None:
            await self._db.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
            )
        await self._db.commit()

    async def create_session(self, title: str | None = None) -> str:
        session_id = new_session_id()
        ts = _now_iso()
        await self._db.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, title, ts, ts),
        )
        await self._db.commit()
        return session_id

    async def session_exists(self, session_id: str) -> bool:
        cur = await self._db.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,))
        return await cur.fetchone() is not None

    async def add_message(self, session_id: str, message: ChatMessage) -> None:
        ts = _now_iso()
        await self._db.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, message.role.value, message.content, ts),
        )
        await self._db.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (ts, session_id))
        await self._db.commit()

    async def get_history(self, session_id: str) -> list[ChatMessage]:
        cur = await self._db.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
        rows = await cur.fetchall()
        return [ChatMessage(role=Role(r["role"]), content=r["content"]) for r in rows]
