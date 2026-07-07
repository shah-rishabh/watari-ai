# ADR-000: LLM provider abstraction via an OpenAI-compatible client

- **Status:** Accepted
- **Date:** 2026-07-07

## Context

Watari must run a local model on a laptop GPU (Ollama, `qwen3.5:4b`) for real
use, and a tiny CPU model (`qwen2.5:0.5b`) inside GitHub Actions for eval CI. It
should also remain portable to any hosted or self-hosted inference server without
a rewrite. We need a seam that:

1. lets us swap the model/endpoint with configuration only, no code branches;
2. keeps vendor SDK types out of the `rag` / `agent` / `memory` layers so those
   stay unit-testable against a fake;
3. gives us a single place to attach token/latency metrics.

## Decision

Use the official **`openai` async SDK pointed at any OpenAI-compatible base URL**,
wrapped behind an owned `LLMProvider` `Protocol` (`src/watari/core/llm.py`).

- Ollama, llama.cpp's `llama-server`, vLLM, and every hosted provider speak the
  OpenAI wire format. Swapping laptop → CI → hosted is a `base_url` change.
- The `Protocol` is the seam. Core/RAG/agent code depends on `LLMProvider` and
  our own `ChatMessage`/`ChatDelta` models — never on `openai` types. Tests use a
  `FakeProvider` that yields scripted deltas.

## Alternatives considered

- **Hand-rolled HTTP + SSE client.** Rejected. Parsing SSE and accumulating
  tool-call deltas is ~500 lines of undifferentiated plumbing. The engineering
  signal this project wants to show lives in the eval harness and security work,
  not in re-implementing an HTTP client a reviewer would not credit.
- **LangChain / LiteLLM.** Rejected for the core seam. They add a large
  dependency surface and their abstractions would obscure exactly the code a
  reviewer looks at. A ~90-line provider is clearer and fully ours.

## Consequences

- Model choice is expressed as *named roles* in `config.py`
  (`chat_model`, `judge_model`, `extract_model`, `embed_model`), each
  independently overridable via `WATARI_*` env vars — this is what makes the
  laptop/CI duality work.
- Metrics and tracing attach at this one seam in later phases.
- If a target server is *not* OpenAI-compatible, we add a second `LLMProvider`
  implementation; nothing else changes.
