# ADR-001: sqlite-vec + FTS5 hybrid store for RAG

- **Status:** Accepted
- **Date:** 2026-07-07

## Context

Watari does RAG over a single user's personal documents, entirely on-device.
The store must: run in-process with no server (the "local-only, one file" pitch),
handle a modest corpus (hundreds–thousands of chunks, not millions), support both
semantic and keyword matching, and be simple enough to own end-to-end.

## Decision

Store chunks in the **same SQLite file** as sessions, using three linked tables
that share one integer id space (the `chunks` rowid):

- `chunks` — chunk text + provenance (`source_path`, `heading_path`, `chunk_index`)
- `chunks_fts` — FTS5 external-content table giving **BM25** keyword ranking
- `vec_chunks` — **sqlite-vec** `vec0` virtual table giving cosine **KNN**

Search runs both arms and fuses their rankings with **Reciprocal Rank Fusion
(RRF)**. RRF needs no score normalisation across arms — it combines a distance
metric and a BM25 rank purely by position — which is exactly why it fits here.

Embeddings: **fastembed** with `BAAI/bge-small-en-v1.5` (384-dim, ONNX, CPU, no
PyTorch). Embeddings never touch the GPU, so the 6GB VRAM budget stays free for
the chat model, and CI embeds identically with no GPU.

## Alternatives considered

- **Qdrant** — a Docker server for a single-user local app contradicts the
  one-process, one-file story. Overkill. Rejected.
- **Chroma** — works, but it is the default in every tutorial (zero
  differentiation), a heavier dependency, and its embedded persistence has
  churned across versions. Rejected.
- **Vector-only (no BM25)** — misses exact-term matches (names, code symbols,
  rare tokens) that a personal-docs corpus is full of. The hybrid arm is cheap
  (FTS5 is built into SQLite) and measurably improves retrieval, which the eval
  harness can show as a `vector-only vs hybrid` delta.

## Consequences

- The whole knowledge base is one portable, backup-able `.db` file — reinforcing
  the privacy pitch.
- Hybrid search + RRF is ~100 lines of SQL and Python we own (`rag/store.py`),
  with the fusion math unit-tested in isolation — strong portfolio signal versus
  importing a black-box retriever.
- `vec0`'s dimension is fixed at table creation, so `embed_dim` is config-driven
  and the table is created dynamically to match.
- sqlite-vec is a loadable extension; it requires a Python whose `sqlite3` was
  built with `enable_load_extension` (verified on the target machine and in CI).
- Reranking (a cross-encoder) is intentionally deferred: RRF captures most of the
  precision gain at zero added latency, and a reranker would add a second model
  to the VRAM budget. It is a documented Phase-6 experiment, not a Phase-2 need.
