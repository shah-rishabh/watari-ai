-- RAG tables. Applied by RagStore after the sqlite-vec extension is loaded
-- (the vec0 virtual table requires it). Lives in the same database file as
-- sessions, keeping "all your data in one local file".

-- One row per ingested source file; content_hash drives incremental re-ingest.
CREATE TABLE IF NOT EXISTS documents (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path   TEXT NOT NULL UNIQUE,
    content_hash  TEXT NOT NULL,
    ingested_at   TEXT NOT NULL
);

-- One row per chunk. chunk rowid is the join key across the vector and FTS
-- tables, so all three share the same integer id space.
CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    source_path   TEXT NOT NULL,
    heading_path  TEXT NOT NULL DEFAULT '',
    chunk_index   INTEGER NOT NULL DEFAULT 0,
    text          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);

-- BM25 keyword index over chunk text. content='chunks' keeps it in sync-by-id
-- with the chunks table (external-content FTS5).
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id'
);
