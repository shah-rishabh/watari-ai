"""Memory store: embedded facts with cosine-dedup and similarity recall.

Mirrors the RAG store's sqlite-vec pattern (same DB file, same lock discipline).
On insert, a new fact whose cosine similarity to an existing active fact exceeds
a threshold **supersedes** the old one (the old row is deactivated, not deleted —
an audit trail). Recall returns the top-k active facts most similar to a query,
above a floor.

Kept intentionally simple: no decay curves, no graph memory, no summarisation
hierarchies. Simple *and* evaluable.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

import sqlite_vec

from watari.config import Settings
from watari.memory.models import Category, Fact, RecalledMemory, StoredMemory
from watari.obs.logging import get_logger

logger = get_logger(__name__)

_SCHEMA_SQL = (Path(__file__).parents[1] / "core" / "schema.sql").read_text(encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class MemoryStore:
    """Synchronous memory store with async wrappers."""

    def __init__(self, settings: Settings) -> None:
        self._db_path = settings.db_path
        self._dim = settings.embed_dim
        self._dedup_threshold = settings.memory_dedup_threshold
        self._recall_floor = settings.memory_recall_floor
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    @property
    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("MemoryStore used before connect()")
        return self._conn

    def connect(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(_SCHEMA_SQL)
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories "
            f"USING vec0(embedding float[{self._dim}])"
        )
        conn.commit()
        self._conn = conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def add(self, fact: Fact, embedding: list[float], *, source: str | None) -> int:
        """Insert a fact, superseding any near-duplicate active fact."""
        with self._lock:
            superseded = self._nearest_active(embedding)
            cur = self._db.execute(
                "INSERT INTO memories (fact, category, active, source, created_at) "
                "VALUES (?, ?, 1, ?, ?)",
                (fact.fact, fact.category.value, source, _now_iso()),
            )
            new_id = int(cur.lastrowid or 0)
            self._db.execute(
                "INSERT INTO vec_memories (rowid, embedding) VALUES (?, ?)",
                (new_id, sqlite_vec.serialize_float32(embedding)),
            )
            if superseded is not None:
                old_id, sim = superseded
                self._db.execute(
                    "UPDATE memories SET active = 0, superseded_by = ? WHERE id = ?",
                    (new_id, old_id),
                )
                self._db.execute("DELETE FROM vec_memories WHERE rowid = ?", (old_id,))
                logger.info("memory.superseded", old=old_id, new=new_id, sim=round(sim, 3))
            self._db.commit()
            return new_id

    @staticmethod
    def _similarity(distance: float) -> float:
        # sqlite-vec returns L2 distance; for (roughly) unit vectors from a
        # sentence embedder, cosine ~ 1 - d^2/2.
        return 1.0 - (distance * distance) / 2.0

    def _knn(self, embedding: list[float], k: int) -> list[tuple[int, float]]:
        # vec0 requires the LIMIT/k directly on the virtual table; joins/filters
        # must happen after this KNN, not inside its query.
        rows = self._db.execute(
            "SELECT rowid AS rid, distance FROM vec_memories "
            "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (sqlite_vec.serialize_float32(embedding), k),
        ).fetchall()
        return [(int(r["rid"]), float(r["distance"])) for r in rows]

    def _active_facts(self, ids: list[int]) -> dict[int, sqlite3.Row]:
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self._db.execute(
            f"SELECT id, fact, category FROM memories WHERE active = 1 AND id IN ({placeholders})",
            ids,
        ).fetchall()
        return {int(r["id"]): r for r in rows}

    def _nearest_active(self, embedding: list[float]) -> tuple[int, float] | None:
        """Return (id, similarity) of the closest active fact if above threshold."""
        # Over-fetch, then keep the closest that is still active.
        knn = self._knn(embedding, 10)
        active = self._active_facts([rid for rid, _ in knn])
        for rid, distance in knn:
            if rid in active:
                similarity = self._similarity(distance)
                if similarity >= self._dedup_threshold:
                    return rid, similarity
                return None
        return None

    def recall(self, embedding: list[float], *, top_k: int) -> list[RecalledMemory]:
        with self._lock:
            knn = self._knn(embedding, max(top_k * 4, 20))
            active = self._active_facts([rid for rid, _ in knn])
        out: list[RecalledMemory] = []
        for rid, distance in knn:
            row = active.get(rid)
            if row is None:
                continue
            score = self._similarity(distance)
            if score < self._recall_floor:
                continue
            out.append(
                RecalledMemory(
                    id=rid,
                    fact=row["fact"],
                    category=Category(row["category"]),
                    score=score,
                )
            )
            if len(out) >= top_k:
                break
        return out

    def list_active(self) -> list[StoredMemory]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, fact, category, active, source FROM memories "
                "WHERE active = 1 ORDER BY id"
            ).fetchall()
        return [
            StoredMemory(
                id=int(r["id"]),
                fact=r["fact"],
                category=Category(r["category"]),
                active=bool(r["active"]),
                source=r["source"],
            )
            for r in rows
        ]

    def forget(self, memory_id: int) -> bool:
        with self._lock:
            cur = self._db.execute(
                "UPDATE memories SET active = 0 WHERE id = ? AND active = 1",
                (memory_id,),
            )
            self._db.execute("DELETE FROM vec_memories WHERE rowid = ?", (memory_id,))
            self._db.commit()
            return cur.rowcount > 0

    def wipe(self) -> int:
        with self._lock:
            n = self._db.execute("SELECT COUNT(*) AS n FROM memories WHERE active = 1").fetchone()[
                "n"
            ]
            self._db.execute("UPDATE memories SET active = 0 WHERE active = 1")
            self._db.execute("DELETE FROM vec_memories")
            self._db.commit()
            return int(n)

    # --- async wrappers ---

    async def aconnect(self) -> None:
        await asyncio.to_thread(self.connect)

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)

    async def arecall(self, embedding: list[float], *, top_k: int) -> list[RecalledMemory]:
        return await asyncio.to_thread(self.recall, embedding, top_k=top_k)

    async def aadd(self, fact: Fact, embedding: list[float], *, source: str | None) -> int:
        return await asyncio.to_thread(self.add, fact, embedding, source=source)
