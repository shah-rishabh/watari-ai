# ADR-002: No shell/exec tool

- **Status:** Accepted
- **Date:** 2026-07-07

## Context

A tool-using assistant is far more capable with a "run this shell command" tool.
It is also the single most dangerous thing we could hand a small local model.

## Decision

**Watari ships no shell or arbitrary-code-execution tool.** The tool set is
deliberately small and specific: `read_file` / `list_dir` (READ), `write_file`
(WRITE, jailed), a local `tasks` CRUD, and an opt-in `web_search` that is off by
default.

## Rationale

- A shell tool cannot be credibly sandboxed part-time. The honest options are a
  real container/seccomp jail (out of scope for this phase) or hand-rolled
  command filtering (a well-known losing game). Shipping an unsandboxed shell and
  calling it "for local use" would be exactly the kind of unmeasured hand-waving
  this project exists to avoid.
- Small models (3–4B) misfire on tool calls. A misfired `write_file` writes a
  wrong file inside a jail; a misfired shell command runs with the user's full
  ambient authority. The blast radius is not comparable.
- **Stating this as a decision is stronger signal than shipping the feature.**
  The threat model lists "arbitrary code execution" with mitigation "there is no
  shell tool" — a clean, defensible line an interviewer can probe.

## Consequences

- Some tasks a shell could do are simply not possible. That is the intended
  trade-off, documented rather than hidden.
- If a future phase adds real container isolation, an `exec` tool could be
  reconsidered behind it — but only then, and only EXECUTE-tier with mandatory
  confirmation.
- Every filesystem-touching tool goes through the `Sandbox` jail, so the maximum
  damage of the tools we *do* ship is bounded to the workspace directory.
