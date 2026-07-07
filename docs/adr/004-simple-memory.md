# ADR-004: Keep long-term memory simple and evaluable

- **Status:** Accepted
- **Date:** 2026-07-07

## Context

"Long-term memory" is a scope-creep magnet: decay curves, graph memory,
hierarchical summarisation, reflection loops. Each adds surface area that is hard
to test and easy to get subtly wrong.

## Decision

Memory is a flat list of **atomic facts** with embeddings, and nothing more:

- Extraction: one constrained LLM call turns a transcript into
  `{fact, category, confidence}` JSON.
- Storage: sqlite-vec in the same DB file. A new fact whose cosine similarity to
  an existing active fact exceeds 0.9 **supersedes** it (the old row is
  deactivated, not deleted — an audit trail).
- Recall: top-k active facts above a similarity floor, injected into context.
- Control: `watari memory list / forget / wipe`, and explicit `/remember`.

No decay, no graph, no summarisation hierarchy.

## Rationale

- **Evaluable.** Every piece has a deterministic-ish test: dedup (does a
  near-duplicate supersede?), recall (is the right fact surfaced?), extraction
  (P/R vs golden facts). The memory eval suite reports real numbers precisely
  because the design is simple.
- The fancy alternatives improve recall at the margin but are far harder to
  measure, and an unmeasured memory system contradicts the project's whole
  "measured, not vibes" thesis.
- Supersede-don't-delete keeps an audit trail cheaply, which fits the same
  provenance instinct as the agent audit log.

## Consequences

- Memory won't reason over relationships between facts or forget stale ones on
  its own — the user forgets explicitly. That's an accepted limitation.
- If a future phase wants richer memory, it can build on this store; the flat
  facts are a clean substrate for summarisation or a graph layer later.
