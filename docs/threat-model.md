# Threat model

A short, honest STRIDE-lite threat model for Watari. The goal is not to claim
Watari is "secure" — it is to state clearly what is defended, how, and what is
**not** defended. Naming residual risk is the point.

## Assets

- **Personal documents** ingested into the RAG store.
- **The local database** (`~/.watari/watari.db`): sessions, chunks, embeddings,
  tasks, memory.
- **The workspace** (`~/.watari/workspace`): files the agent may read/write.
- **The user's machine** more broadly (the ambient authority the process runs
  with).

## Trust boundaries

1. **User input** — trusted (it's the user's own instruction).
2. **RAG content / read files / tool outputs** — **untrusted**. A document or a
   file the agent reads may contain adversarial instructions (prompt injection).
3. **The model weights / Ollama binary** — trusted as installed. Watari does not
   defend against a malicious local model or a backdoored inference server.
4. **The network** — for the default local build there is essentially none;
   `web_search` (opt-in, off by default) is the only egress path.

## Threats, mitigations, residual risk

| Threat | Mitigation | Residual risk |
| --- | --- | --- |
| **Prompt injection** via adversarial RAG content or file contents | Untrusted content is wrapped with a spotlighting preamble + fences (`security/validation.py`); system prompt instructs the model to treat such content as data. Measured by the injection eval suite (canary tokens, before/after ASR). | **Reduced, not eliminated.** A sufficiently clever injection can still succeed, especially on a small local model. ASR is > 0 in some runs. |
| **Path traversal / sandbox escape** by a file tool | Pure-Python jail (`security/sandbox.py`): resolve symlinks, reject anything outside the workspace; null-byte and absolute-path handling; red-teamed in `tests/security/`. | The jail bounds *where*, not *what*. A permitted write can still place malicious content **inside** the workspace. Not a container/seccomp sandbox. |
| **Destructive / unexpected tool actions** | Risk tiers: READ auto-approved, WRITE/EXECUTE require confirmation; per-session allowlist; every decision + execution appended to a JSONL audit log. | The user can approve a harmful action; `--yolo` disables prompts (loudly logged) and is for demos only. |
| **Arbitrary code execution** | **There is no shell/exec tool.** See [ADR-002](adr/002-no-shell-tool.md). | None from a shell tool — because there isn't one. This is a deliberate capability cut. |
| **Data exfiltration** | Local-first by default; no network egress except opt-in `web_search`, which is off unless explicitly enabled (and says so in config/logs). | If `web_search` is enabled, a successful injection could try to exfiltrate via a crafted query. Documented; off by default. |
| **Secret leakage in logs** | `structlog` redaction filter for secret-looking keys; `gitleaks` in pre-commit; `.env` gitignored. | Best-effort key-name matching; an oddly-named secret could slip through. |
| **CSRF against the local API** | API binds `127.0.0.1` by default and ships a closed CORS policy, so a malicious web page cannot drive it from the browser. | A local process with loopback access is trusted; this is single-user by design. |
| **Resource exhaustion via tool I/O** | Read/write byte caps; tool-result char cap; agent iteration cap. | A permitted write within caps can still consume disk over many calls. |

## Deliberately NOT mitigated (and why)

These are omitted on purpose; shipping them would be security theater for a
single-user, local-first app:

- **Auth / TLS / rate limiting on the localhost API** — there is one user, on
  loopback. Adding auth would be ceremony without a threat it addresses.
- **"AI guardrail" content filters** — brittle, easy to bypass, and not the
  threat model here (the user is not adversarial to themselves).
- **Container / seccomp isolation of tools** — real value, real cost; out of
  scope for the current phase. The jail is honestly described as pure-Python.

## How the claims are checked

- Sandbox escape vectors: `tests/security/test_sandbox.py` (traversal, symlinks,
  null bytes, absolute paths).
- Untrusted-content wrapping: `tests/security/test_validation.py`.
- Prompt-injection resistance: the `injection` eval suite reports **attack
  success rate before and after** the wrapping mitigation. It is report-only
  (not a CI gate) because ASR is "lower is better", which the floor-based gate
  doesn't model; regressions are reviewed via the committed baseline.
