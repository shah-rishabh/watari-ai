# Watari AI — Local-First LLM Personal Assistant (Portfolio Project Plan)

## Context

Rishabh wants a GitHub portfolio project for **AI/ML Engineer** roles: a local LLM personal assistant that is production-ready — evals, security, observability, CI — not vibes. Repo `watari-ai` is empty (fresh git init). Constraints: RTX 3060 laptop (6GB VRAM), marketed as **local-only** ("your data never leaves your machine"), Python/FastAPI stack, capabilities = RAG over personal docs + tool-use agent + long-term memory. Solo, part-time.

**Alignment verdict: yes, with a framing condition.** "Ollama chat wrapper" repos are commodity; what interviewers credit is the engineering *around* the model. So the eval harness and security work are the product — every capability claim in the README must carry a measured number. The plan below is built around that.

## Key architecture decisions

| Decision | Choice | Why |
|---|---|---|
| Provider layer | `openai` SDK → Ollama's `/v1`, wrapped in an owned `LLMProvider` Protocol | Same client works for laptop (qwen2.5:3b), CI (qwen2.5:0.5b on CPU), any OpenAI-compatible server. Hand-rolling HTTP/SSE plumbing earns no credit; the Protocol seam is where mocking + metrics attach. (ADR-000) |
| Default models | `qwen2.5:3b` chat (fits 6GB), `bge-small-en-v1.5` embeddings via **fastembed** (ONNX, CPU, no PyTorch dep) | Named model roles (chat/judge/extract/embed) in pydantic-settings, all independently overridable; CI sets all to 0.5b |
| Vector store | **sqlite-vec + FTS5 hybrid search with RRF fusion**, single SQLite file at `~/.watari/watari.db` (sessions + vectors + memory) | No server dependency (fits local-only pitch), hybrid search hand-written in ~100 lines of SQL is real portfolio signal, and "your data is one local file" is the privacy story. (ADR-001) |
| Eval harness | **Custom pytest-orchestrated harness, hand-rolled metrics** (no RAGAS/deepeval/promptfoo) | recall@k/MRR/rubric-judge are each <50 lines; writing + unit-testing the metric functions is the strongest signal. Judge prompts calibrated for a local 3B, with a human-labeled calibration set reporting Cohen's kappa. |
| Agent tools | Exactly 4: `read_file`/`list_dir`, `write_file`, `tasks` (SQLite CRUD), `web_search` (opt-in, off by default). **No shell tool** | Small models misfire on big tool sets; no-shell is a documented threat-model decision, not a gap. READ auto-approved; WRITE needs confirmation; JSONL audit log. |
| Sandboxing | Pure-Python path jail (resolve symlinks → `is_relative_to(workspace_root)`), size caps, honest residual-risk docs | Credible and unit-testable; don't claim container isolation that doesn't exist |
| Memory | Post-session LLM fact extraction → embedded facts in SQLite, cosine-dedup (>0.9 supersedes), top-5 retrieval into context, `watari memory list/forget/wipe` | Simple and evaluable; no decay curves/graph memory |
| Observability | structlog (JSON, request/session IDs) from day 1; hand-rolled `@traced` spans with OTel-compatible attribute names; `/metrics` JSON + `watari stats` | Full OTel SDK+collector has no consumer in a single-user local app; document the mapping instead |
| Interface | CLI-first (typer + rich) + FastAPI SSE; demo via scripted **vhs** terminal GIF. Web UI = stretch only | Polished TUI is more distinctive than another React chat page at 20% of the effort |
| CI | `ci.yml`: ruff + pyright(strict) + pytest + docker build. `evals.yml`: Ollama in Actions, cached `qwen2.5:0.5b`, smoke eval suites, PR comment with metrics table, **threshold gates** (absolute floors, not exact-match) | Ollama-in-CI gives exact parity with the dev path through the same client |

## Repo layout (src layout, uv-managed)

```
src/watari/{config.py, cli.py, api/, core/{llm,models,session,prompts/},
            rag/{ingest,chunking,embeddings,store,retrieve,cite},
            agent/{loop,registry,permissions,tools/}, memory/, security/, obs/,
            evals/{runner,metrics/,report,gate}}
evals/{datasets/*.jsonl, corpora/, baselines/}   # data, not code; small, git-versioned
tests/{unit/, integration/, security/}
docs/{architecture.md, threat-model.md, evals.md, adr/}
.github/workflows/{ci.yml, evals.yml}
```

- Golden datasets: hand-verified JSONL (retrieval 40, RAG-QA 30, agent 25, memory 20, injection 25 cases; ~8/suite tagged `smoke` for CI). Synthetic fictional corpus in `evals/corpora/` (no privacy issues, controlled answerability).
- Prompts as versioned `.md` files, not f-strings.
- Chunking: heading-aware markdown split → ~400 tokens, 15% overlap, metadata `{source_path, heading_path, chunk_index}`. PDF via `pymupdf4llm` → markdown. No LangChain.
- Citations: numbered `[n]` chunks; post-processor validates every cited index (hallucinated → logged+stripped); **citation validity rate is an eval metric**.

## Eval metrics (the centerpiece)

| Suite | Metrics | Method |
|---|---|---|
| Retrieval | recall@{3,5,10}, MRR, citation validity | deterministic vs golden chunk IDs |
| Generation | faithfulness, answer relevance | claim-decomposition LLM-judge, binary/3-point rubrics, few-shot anchors; **judge calibrated vs ~30 human labels (kappa reported)** |
| Agent | tool-selection accuracy, arg correctness, task completion, mean iterations | golden tool sequences + deterministic state assertions |
| Memory | extraction P/R vs golden facts, recall-in-context | golden transcripts |
| Security | **injection attack success rate, before/after mitigations** | canary tokens in adversarial RAG docs + malicious tool outputs — deterministic detection |

Runner: `watari evals run --suite X --model Y` → results.json + markdown table → `gate.py` compares vs committed baselines, non-zero exit on regression. README tables auto-regenerated between markers; baselines updated by explicit reviewable commits.

## Security scope (credible vs theater)

Build: injection eval suite + mitigations (untrusted-content wrapping, spotlighting delimiters) with before/after ASR table; path jail + `tests/security/` (traversal, symlinks, null bytes); 2-page STRIDE-lite threat model **explicitly listing residual risks**; pydantic input validation + size caps; gitleaks in pre-commit; `SecretStr` + log redaction for the one optional API key.
Deliberately skip (documented with reasoning in threat model): localhost rate limiting/auth/TLS, "AI guardrail" content filters.
API binds `127.0.0.1` by default (never `0.0.0.0` outside Docker) with a strict CORS posture — localhost APIs are reachable by malicious websites via CSRF-style requests; one explicit threat-model line covers this.

## Milestones (each ends CI-green, demo-able, tagged)

1. **Skeleton & streaming chat** (~2 wks): scaffold, config, LLMProvider, SQLite sessions, FastAPI SSE + typer CLI, structlog, ci.yml, pre-commit, Dockerfile, ADR-000
2. **RAG with citations** (~3 wks): ingestion→hybrid store→citations, eval corpus + retrieval golden set, hand-rolled recall@k/MRR, ADR-001
3. **Eval harness & CI evals** (~2–3 wks, deliberately before agent): runner/report/gate, LLM-judge + calibration, evals.yml, PR comments, README auto-tables, docs/evals.md — *interview-ready after this phase*
4. **Agent & security** (~3–4 wks): registry/loop/3 tools, permissions + audit log, path jail + tests, agent evals, injection suite + ASR table, threat model
5. **Memory & observability polish** (~2 wks): memory pipeline + evals, /metrics, span waterfalls, latency figures (p50 TTFT on RTX 3060) into README
6. **Polish** (~1–2 wks): README overhaul, mermaid diagram, vhs GIF, ADR backfill, compose demo. Stretch (in order): swap hand-rolled spans for real OTel SDK (resume keyword, cheap given semconv-compatible naming) → reranker experiment → static web page

**Top risks:** (1) 3B LLM-judge noise → tight rubrics, calibration set, judge metrics report-only if needed; (2) 3B tool-calling misfires → ≤4 tools, one retry, imperfect scores are fine — measuring them is the product; (3) Ollama-in-CI flakiness → model cache, health-check retries, llama-server fallback (same client).
**Cut order:** reranker → web UI → web_search → OTel exporter → auto-memory (keep explicit /remember) → tasks tool. **Never cut:** eval depth, injection suite, docs.

## Execution scope on approval: Phase 1

1. `pyproject.toml` (uv, PEP 621, py3.12+), src layout, ruff + pyright strict config, pre-commit (ruff, gitleaks, uv lock check)
2. `config.py` — pydantic-settings, `WATARI_` prefix, named model roles
3. `core/models.py` (ChatMessage/ChatDelta/Usage), `core/llm.py` (Protocol + OpenAI-client-against-Ollama impl with streaming)
4. `core/session.py` — aiosqlite, schema.sql + tiny version-table migration runner
5. `obs/logging.py` — structlog with request/session IDs
6. FastAPI app: `/chat` SSE, `/health`; typer CLI: `watari chat` with rich streaming output
7. Unit tests (mocked provider) for context assembly, session persistence, config; `ci.yml`; multi-stage Dockerfile; ADR-000; initial README
8. Verify end-to-end: `ollama pull qwen2.5:3b`, run `watari chat`, confirm streaming + session persistence; `uv run pytest`, ruff, pyright all green

## Verification (ongoing)

- Every phase: `uv run pytest`, `ruff check`, `pyright`, then live check via `watari chat` / CLI commands against local Ollama
- Eval phases: run the suite locally on 3b, confirm CI smoke run on 0.5b passes gates in a PR
- Security phase: `tests/security/` red-team unit cases + injection suite ASR measured before/after mitigations
