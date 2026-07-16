"""Hybrid vector + keyword store on SQLite.

The distinctive part of the RAG stack. A single SQLite file holds three linked
tables that share one integer id space (the ``chunks`` rowid):

- ``chunks``      — chunk text + provenance metadata
- ``chunks_fts``  — FTS5 BM25 keyword index (external content over ``chunks``)
- ``vec_chunks``  — sqlite-vec vec0 table for cosine KNN over embeddings

Search runs both a vector KNN and a BM25 query, then fuses their rankings with
Reciprocal Rank Fusion (RRF). Hybrid-with-RRF captures both semantic and lexical
matches and is ~100 lines of SQL + Python we own — no server, no LangChain.

sqlite-vec needs a synchronous connection with ``enable_load_extension``; all
blocking work is offloaded to threads so async callers never stall.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

import sqlite_vec

from watari.config import Settings
from watari.obs.logging import get_logger
from watari.rag.models import Chunk, RetrievedChunk

logger = get_logger(__name__)

_SCHEMA_SQL = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class RagStore:
    """Synchronous hybrid store with async wrappers."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._db_path = settings.db_path
        self._dim = settings.embed_dim
        self._rrf_k = settings.rrf_k
        self._conn: sqlite3.Connection | None = None
        # A single SQLite connection is not safe for concurrent use, even with
        # check_same_thread=False. Eval suites run cases concurrently via
        # asyncio.to_thread, so all connection access is serialised through this
        # lock. Contention is negligible for a single-user local store.
        self._lock = threading.Lock()

    @property
    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("RagStore used before connect()")
        return self._conn

    def connect(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the connection is created in a worker thread
        # via aconnect() but read (e.g. stats) from the event-loop thread. This
        # is safe because every access is serialised — searches/ingest go through
        # asyncio.to_thread and never run concurrently on one store instance.
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(_SCHEMA_SQL)
        # vec0 table dimension is config-driven, so it's created here rather than
        # in the static schema file.
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks "
            f"USING vec0(embedding float[{self._dim}])"
        )
        conn.commit()
        self._conn = conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # --- Ingestion -------------------------------------------------------

    def needs_ingest(self, source_path: str, content_hash: str) -> bool:
        """True if this file is new or its content changed since last ingest."""
        with self._lock:
            row = self._db.execute(
                "SELECT content_hash FROM documents WHERE source_path = ?",
                (source_path,),
            ).fetchone()
        return row is None or row["content_hash"] != content_hash

    def upsert_document(
        self,
        *,
        source_path: str,
        content_hash: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> int:
        """Replace a document and all its chunks atomically. Returns chunk count."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")

        with self._lock:
            return self._upsert_document_locked(source_path, content_hash, chunks, embeddings)

    def _upsert_document_locked(
        self,
        source_path: str,
        content_hash: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> int:
        db = self._db
        # Remove any prior version of this document (cascades to chunks; FTS/vec
        # rows are cleaned explicitly below since they aren't FK-linked).
        old = db.execute(
            "SELECT id FROM documents WHERE source_path = ?", (source_path,)
        ).fetchone()
        if old is not None:
            self._delete_document_rows(old["id"])

        cur = db.execute(
            "INSERT INTO documents (source_path, content_hash, ingested_at) VALUES (?, ?, ?)",
            (source_path, content_hash, _now_iso()),
        )
        document_id = int(cur.lastrowid or 0)

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            ccur = db.execute(
                "INSERT INTO chunks (document_id, source_path, heading_path, "
                "chunk_index, text) VALUES (?, ?, ?, ?, ?)",
                (
                    document_id,
                    chunk.source_path,
                    chunk.heading_path,
                    chunk.chunk_index,
                    chunk.text,
                ),
            )
            chunk_id = int(ccur.lastrowid or 0)
            db.execute(
                "INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)",
                (chunk_id, chunk.text),
            )
            db.execute(
                "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
                (chunk_id, sqlite_vec.serialize_float32(embedding)),
            )

        db.commit()
        return len(chunks)

    def _delete_document_rows(self, document_id: int) -> None:
        db = self._db
        ids = [
            r["id"]
            for r in db.execute(
                "SELECT id FROM chunks WHERE document_id = ?", (document_id,)
            ).fetchall()
        ]
        for chunk_id in ids:
            db.execute("DELETE FROM chunks_fts WHERE rowid = ?", (chunk_id,))
            db.execute("DELETE FROM vec_chunks WHERE rowid = ?", (chunk_id,))
        db.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    def stats(self) -> dict[str, int]:
        with self._lock:
            db = self._db
            docs = db.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
            chunks = db.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        return {"documents": int(docs), "chunks": int(chunks)}

    # --- Search ----------------------------------------------------------

    def _vector_ranking(self, query_embedding: list[float], k: int) -> list[int]:
        rows = self._db.execute(
            "SELECT rowid FROM vec_chunks WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (sqlite_vec.serialize_float32(query_embedding), k),
        ).fetchall()
        return [int(r["rowid"]) for r in rows]

    def _keyword_ranking(self, query: str, k: int) -> list[int]:
        match = _fts_query(query)
        if not match:
            return []
        rows = self._db.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
            (match, k),
        ).fetchall()
        return [int(r["rowid"]) for r in rows]

    def hybrid_search(
        self, query: str, query_embedding: list[float], *, top_k: int
    ) -> list[RetrievedChunk]:
        """Fuse vector KNN and BM25 rankings with RRF, return top_k chunks."""
        with self._lock:
            return self._hybrid_search_locked(query, query_embedding, top_k)

    def _hybrid_search_locked(
        self, query: str, query_embedding: list[float], top_k: int
    ) -> list[RetrievedChunk]:
        # Over-fetch from each arm so the fusion has candidates to work with.
        pool = max(top_k * 4, 20)
        vec_ids = self._vector_ranking(query_embedding, pool)
        kw_ids = self._keyword_ranking(query, pool)

        fused = reciprocal_rank_fusion([vec_ids, kw_ids], k=self._rrf_k)
        top_ids = [cid for cid, _ in fused[:top_k]]
        if not top_ids:
            return []

        scores = dict(fused)
        placeholders = ",".join("?" for _ in top_ids)
        rows = self._db.execute(
            f"SELECT id, source_path, heading_path, chunk_index, text "
            f"FROM chunks WHERE id IN ({placeholders})",
            top_ids,
        ).fetchall()
        by_id = {int(r["id"]): r for r in rows}

        results: list[RetrievedChunk] = []
        for cid in top_ids:
            r = by_id.get(cid)
            if r is None:
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=cid,
                    source_path=r["source_path"],
                    heading_path=r["heading_path"],
                    chunk_index=r["chunk_index"],
                    text=r["text"],
                    score=scores[cid],
                )
            )
        return results

    # --- Async wrappers --------------------------------------------------

    async def aconnect(self) -> None:
        await asyncio.to_thread(self.connect)

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)

    async def ahybrid_search(
        self, query: str, query_embedding: list[float], *, top_k: int
    ) -> list[RetrievedChunk]:
        return await asyncio.to_thread(self.hybrid_search, query, query_embedding, top_k=top_k)


def reciprocal_rank_fusion(rankings: list[list[int]], *, k: int = 60) -> list[tuple[int, float]]:
    """Fuse ranked id lists via Reciprocal Rank Fusion.

    Each list contributes ``1 / (k + rank)`` to an id's score (rank is 0-based).
    Returns (id, score) pairs sorted by descending score. RRF needs no score
    normalisation across arms, which is exactly why it suits fusing a distance
    metric with a BM25 rank.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 OR-query of quoted terms.

    Quoting each token defuses FTS5 operator characters in user input, and OR
    keeps recall high (any term may match) — the BM25 rank still orders them.
    """
    tokens = [t for t in _tokenize(text) if t]
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens)


def _tokenize(text: str) -> list[str]:
    out: list[str] = []
    word: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            word.append(ch)
        elif word:
            out.append("".join(word))
            word = []
    if word:
        out.append("".join(word))
    return out
