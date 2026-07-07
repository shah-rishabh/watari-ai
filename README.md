# Watari AI

A **local-first LLM personal assistant** — your data never leaves your machine.
Built as a production-engineering exercise: rigorous evals, security hardening,
observability, and CI are first-class, not afterthoughts.

> **Status:** Phase 1 of 6 — streaming chat with a clean, typed, tested core.
> RAG, agent/tool use, long-term memory, and the eval harness land in later
> phases (see [PLAN.md](PLAN.md)).

## Why this exists

Most "local LLM assistant" repos are a thin wrapper over Ollama. This one is
built around the engineering *around* the model: every capability claim is meant
to carry a measured number from an eval harness, and the security posture is
documented rather than assumed.

## Architecture (Phase 1)

```
                    ┌──────────────┐     ┌──────────────┐
   CLI (typer+rich) │              │     │  SessionStore│  SQLite
   ─────────────────▶  ChatService ├─────▶  (aiosqlite) │  ~/.watari/watari.db
   FastAPI (SSE)    │              │     └──────────────┘
   ─────────────────▶              │     ┌──────────────┐
                    │              ├─────▶ LLMProvider   │  OpenAI-compatible
                    └──────────────┘     │ (→ Ollama /v1)│  → llama.cpp/vLLM/…
                                         └──────────────┘
```

Both surfaces (CLI and API) consume the **same** `ChatService.stream_reply`
async iterator — the thin-adapter design in one picture. The `LLMProvider`
Protocol is the swappable seam: laptop runs `qwen3.5:4b`, CI runs
`qwen2.5:0.5b`, via configuration only. See
[ADR-000](docs/adr/000-provider-abstraction.md).

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

## Security posture

Even at Phase 1 the defaults are deliberate, not accidental:

- The API **binds `127.0.0.1`** by default and ships a **closed CORS policy** —
  a localhost API is reachable by malicious web pages via CSRF-style requests,
  so we do not listen on `0.0.0.0` outside a container.
- All request bodies are **pydantic-validated** with size caps.
- The structured-logging pipeline **redacts** secret-looking keys.
- `gitleaks` runs in pre-commit; `.env` is gitignored.

A full threat model arrives with the agent/tool phase (Phase 4), including a
prompt-injection eval suite with before/after attack-success-rate numbers.

## Development

```bash
uv run pytest          # tests
uv run ruff check .    # lint
uv run ruff format .   # format
uv run pyright         # strict type check
pre-commit install     # enable git hooks
```

CI runs lint, format-check, strict type-check, unit tests, and a Docker build on
every PR. See [.github/workflows/ci.yml](.github/workflows/ci.yml).

## Roadmap

See [PLAN.md](PLAN.md) for the full six-phase plan. Next up: RAG over personal
docs with citations, and the hand-rolled eval harness that is the centerpiece of
the project.

## License

MIT
