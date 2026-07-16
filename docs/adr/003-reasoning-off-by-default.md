# ADR-003: Reasoning ("thinking") off by default

- **Status:** Accepted
- **Date:** 2026-07-07

## Context

The default chat model, Qwen 3.5 4B, is a *reasoning* model: it emits a hidden
chain of thought before its visible answer. On the OpenAI-compatible `/v1`
endpoint, that reasoning streams on a separate `reasoning` field, and in early
testing a single one-sentence answer cost **911 completion tokens** — almost all
of it thinking.

## Decision

Default `reasoning_effort` to `"none"` (`config.py`). The provider disables
thinking for a snappy interactive assistant; `"low"`/`"medium"`/`"high"` remain
available per deployment for tasks that benefit from deliberation. The provider
also captures any `reasoning` deltas on a distinct `ChatDelta.reasoning` field so
they are observable but **never** mixed into or persisted as the answer.

## Rationale

- For an interactive assistant, latency and context budget matter more than the
  marginal quality of chain-of-thought on simple turns. Disabling thinking cut a
  test reply from 911 tokens (~17 s) to **23 tokens (~1.2 s)**.
- The lever is model-specific and non-obvious: the top-level `think` flag and
  `chat_template_kwargs.enable_thinking` did **not** work on Ollama's `/v1`;
  `reasoning_effort` (a standard OpenAI parameter) did. This was found by
  probing raw deltas, not by reading docs — worth recording so it isn't
  re-litigated.

## Consequences

- Clean token accounting: `completion_tokens` reflects the answer, not hidden
  reasoning, which keeps the observability and eval numbers meaningful.
- Servers that don't understand `reasoning_effort` ignore the field (it's passed
  via `extra_body`), so non-reasoning models are unaffected.
- Turning reasoning back on is a one-line config change if a future task wants it.
