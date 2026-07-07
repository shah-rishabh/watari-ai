# Watari AI

A **local-first LLM personal assistant** — your data never leaves your machine.
Built as a production-engineering exercise: rigorous evals, security hardening,
observability, and CI are first-class, not afterthoughts.

> **Status:** Phase 4 of 6 — streaming chat, **RAG with hybrid retrieval and
> validated citations**, a **hand-rolled eval harness** with CI gates, and a
> **sandboxed tool-using agent** with a prompt-injection eval suite. Long-term
> memory and observability polish land in the final phases (see [PLAN.md](PLAN.md)).

## Why this exists

Most "local LLM assistant" repos are a thin wrapper over Ollama. This one is
built around the engineering *around* the model: every capability claim is meant
to carry a measured number from an eval harness, and the security posture is
documented rather than assumed.

## Architecture

```
                    ┌──────────────┐     ┌──────────────┐
   CLI (typer+rich) │              │     │  SessionStore│  ┐
   ─────────────────▶  ChatService ├─────▶  (aiosqlite) │  │  one SQLite file
   FastAPI (SSE)    │              │     ├──────────────┤  ├─ ~/.watari/watari.db
   ─────────────────▶              │     │  RagStore    │  │  (sessions + vectors)
                    │              ├─────▶ vec0 + FTS5  │  ┘
                    │              │     └──────────────┘
                    │              │     ┌──────────────┐
                    │              ├─────▶ LLMProvider   │  OpenAI-compatible
                    └──────────────┘     │ (→ Ollama /v1)│  → llama.cpp/vLLM/…
                                         └──────────────┘
```

Both surfaces (CLI and API) consume the **same** `ChatService.stream_reply`
async iterator — the thin-adapter design in one picture. The `LLMProvider`
Protocol is the swappable seam: laptop runs `qwen3.5:4b`, CI runs
`qwen2.5:0.5b`, via configuration only. See
[ADR-000](docs/adr/000-provider-abstraction.md).

## RAG over your documents

Ingest markdown and PDF files into a local store and ask grounded, cited
questions — nothing leaves your machine.

- **Hybrid retrieval:** every query runs *both* a semantic vector search
  (sqlite-vec cosine KNN over `bge-small` embeddings) and a keyword search
  (SQLite FTS5 BM25), fused with **Reciprocal Rank Fusion**. Semantic catches
  paraphrases; BM25 catches exact terms (names, code symbols). See
  [ADR-001](docs/adr/001-vector-store.md).
- **Heading-aware chunking:** markdown is split on its heading structure, so each
  chunk carries a heading trail (`Projects > Watari > Design`) used in citations.
- **Validated citations:** retrieved chunks are numbered `[1]..[k]`; after
  generation, every `[n]` marker is checked against the retrieved set —
  hallucinated citations are logged and stripped, and a `Sources:` footnote lists
  only the chunks actually used. Citation validity is a first-class eval metric
  in Phase 3.
- **One local file:** chunks, vectors, and keyword index live in the same SQLite
  database as your chat sessions.

```bash
# Ingest a folder of notes/PDFs (idempotent — unchanged files are skipped)
uv run watari ingest ~/Documents/notes

# See what's stored
uv run watari stats

# Ask cited questions (RAG auto-enables when the store is non-empty;
# toggle per-turn with /rag inside the REPL)
uv run watari chat
```

## Evaluation — measured, not vibes

Every retrieval and generation claim carries a number from a **hand-rolled eval
harness** (no RAGAS/deepeval — the metric math is ours and unit-tested). Suites
run against a fictional corpus with hand-verified golden answers.

<!-- EVAL_TABLE_START -->
| Suite | Model | Cases | Metric | Value |
| --- | --- | --- | --- | --- |
| retrieval | qwen3.5:4b | 24 | recall@3 | 0.917 |
| retrieval | qwen3.5:4b | 24 | recall@5 | 0.958 |
| retrieval | qwen3.5:4b | 24 | recall@10 | 1.000 |
| retrieval | qwen3.5:4b | 24 | mrr | 0.910 |
| rag-qa | qwen3.5:4b | 15 | faithfulness | 0.981 |
| rag-qa | qwen3.5:4b | 15 | answer_relevance | 0.933 |
| rag-qa | qwen3.5:4b | 15 | citation_validity | 1.000 |
| agent | qwen3.5:4b | 10 | tool_selection | 1.000 |
| agent | qwen3.5:4b | 10 | task_completion | 1.000 |
| agent | qwen3.5:4b | 10 | mean_iterations | 2.100 |
<!-- EVAL_TABLE_END -->

- **Retrieval** — `recall@k` and `MRR` against golden chunk refs (deterministic).
- **Faithfulness** — an LLM-judge decomposes each answer into atomic claims and
  checks each against the retrieved context. The judge is itself **calibrated**
  against human labels (Cohen's kappa), because a judge you can't trust isn't a
  metric — see [docs/evals.md](docs/evals.md).
- **Agent** — `tool_selection` and `task_completion` are checked against **real
  state** (was the file written? was the task row created?), not judged.
- **Regression gates** — [`evals.yml`](.github/workflows/evals.yml) runs the
  smoke suites through a tiny CPU model (`qwen2.5:0.5b`) in CI and fails the build
  if any metric drops below its floor in [`thresholds.json`](evals/thresholds.json).

```bash
uv run watari evals run --suite all     # full suites, local model
uv run watari evals calibrate           # judge-vs-human agreement
```

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com/).

```bash
# 1. Pull a local model (fits a 6GB GPU)
ollama pull qwen3.5:4b

# 2. Install
uv sync --extra dev

# 3. Chat (interactive REPL, streaming Markdown)
uv run watari chat

# …or run the API and open http://127.0.0.1:8000/docs
uv run watari serve
```

**Hardware note:** developed on an RTX 3060 laptop (6GB VRAM). `qwen3.5:4b`
(~4GB at Q4) runs comfortably with headroom for RAG context; `qwen3.5:9b` is a
deeper-reasoning upgrade if you have the VRAM/patience. Any model is
configurable via `WATARI_CHAT_MODEL`.

## Configuration

All settings are environment variables with the `WATARI_` prefix (or a `.env`
file — see [.env.example](.env.example)). Model choice is expressed as named
roles so the same code serves a laptop GPU and CPU-only CI:

| Setting | Default | Purpose |
|---|---|---|
| `WATARI_CHAT_MODEL` | `qwen3.5:4b` | Conversational model |
| `WATARI_LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible endpoint |
| `WATARI_HOST` | `127.0.0.1` | Bind loopback by default (see Security) |
| `WATARI_MAX_CONTEXT_TOKENS` | `8192` | Context assembly budget |

## Agent & tools

Watari can use a small, deliberately-limited tool set to act on the user's
behalf: `read_file` / `list_dir`, `write_file`, a local `tasks` to-do list, and
an opt-in `web_search` (off by default). **There is no shell/exec tool** — a
documented decision, not a gap ([ADR-002](docs/adr/002-no-shell-tool.md)).

- **Permission model** — READ tools auto-approve; WRITE/EXECUTE require
  confirmation (the CLI prompts; `--yolo` bypasses for demos, loudly). Every
  decision and execution is appended to a JSONL **audit log**.
- **Filesystem jail** — every path a tool touches is resolved (following
  symlinks) and confirmed inside the workspace; traversal, symlink escape, and
  null bytes are red-teamed in [`tests/security/`](tests/security/).

```bash
uv run watari agent "Create notes/todo.md with a checklist of my errands."
```

## Security posture

Security is a measured pillar, not a checkbox. The full
[threat model](docs/threat-model.md) is STRIDE-lite and **states residual risk
explicitly** — what's defended, how, and what deliberately isn't.

**Prompt injection** is the flagship: the `injection` eval suite plants
adversarial instructions (carrying canary tokens) in content the model reads,
and measures **attack success rate before and after** the untrusted-content
wrapping mitigation.

| Condition | Attack success rate (12 cases, qwen3.5:4b) |
| --- | --- |
| Unmitigated (raw injected content) | ~0.25 |
| Mitigated (spotlighting / untrusted-content wrapping) | ~0.00 |

The wrapping drives successful injections to near-zero on this suite. (Small-model
runs vary; the number is honest about that — and the threat model notes injection
is *reduced, not eliminated*.)

Other defaults, deliberate not accidental:

- API **binds `127.0.0.1`** with a **closed CORS policy** — a localhost API is
  reachable by malicious web pages via CSRF-style requests.
- Filesystem **jail** + risk-tiered **permissions** + **audit log** for tools.
- Request bodies **pydantic-validated** with size caps; tool I/O byte-capped.
- Structured logging **redacts** secret-looking keys; `gitleaks` in pre-commit.

Deliberately skipped (documented, with reasoning): localhost auth/TLS/rate
limiting, "AI guardrail" content filters, container isolation of tools.

## Development

```bash
uv run pytest          # tests
uv run ruff check .    # lint
uv run ruff format .   # format
uv run pyright         # strict type check
pre-commit install     # enable git hooks
```

CI runs lint, format-check, strict type-check, unit tests, and a Docker build on
every PR ([ci.yml](.github/workflows/ci.yml)); a separate
[evals.yml](.github/workflows/evals.yml) runs the eval smoke suites through a
tiny CPU model and gates on metric floors.

## Roadmap

See [PLAN.md](PLAN.md) for the full six-phase plan. Next up (Phase 5): long-term
memory across sessions (fact extraction, embedded recall, `watari memory`
commands) with its own eval suite, plus observability polish (`/metrics`, span
timing, latency figures into the README).

## License

MIT
