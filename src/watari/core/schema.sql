-- Watari schema. Applied idempotently by the migration runner in session.py.
-- A single SQLite file holds sessions now; RAG vectors and memory facts land
-- in later phases in this same database.

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, id);

-- Local task/reminder list, exposed via the `tasks` agent tool. The
-- "calendar-ish" capability without CalDAV scope creep.
CREATE TABLE IF NOT EXISTS tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    done        INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);

-- Long-term memory: atomic facts extracted about the user, embedded for
-- similarity retrieval. `active` supports supersede-on-dedup (an updated fact
-- deactivates the old one, keeping an audit trail rather than deleting).
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    fact        TEXT NOT NULL,
    category    TEXT NOT NULL DEFAULT 'other',
    active      INTEGER NOT NULL DEFAULT 1,
    source      TEXT,
    created_at  TEXT NOT NULL,
    superseded_by INTEGER
);

CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(active);
