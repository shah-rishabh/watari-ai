# TODO

Backlog of known gaps and follow-ups. Not yet scheduled.

## One interface: chat *is* the agent (`src/watari/cli.py`, `src/watari/core/chat.py`, `src/watari/agent/loop.py`)

**High priority.** `watari chat` and `watari agent` are two front doors to what
should be one product. The split is an artifact of build order, not design — chat
landed early, the agent landed in Phase 4 (`57fe69f`) and got its own command
rather than being folded into the thing that already existed. The result is a
tool-using assistant whose primary surface can't use tools: ask `chat` to read a
file and it will apologise; you have to quit, re-invoke as `watari agent "..."`,
and lose the conversation to do it. Watari *is* the agent. Chat should have the
capability by default, and `agent` should stop being a peer command.

Target end-state: `watari chat` runs the agent loop every turn, tools available
without ceremony, permission prompts inline in the REPL. Tool use becomes a
property of a turn, not a choice of binary.

- [ ] **Unify the two execution paths.** This is the load-bearing item; everything
  else in this section is a consequence. Today they share a provider and nothing
  else. `ChatService.stream_reply` (`chat.py:71`) streams `ChatDelta`s off
  `provider.stream`, persists both turns via `SessionStore`, and never sees a tool.
  `AgentLoop.run` (`loop.py:56`) calls the non-streaming
  `provider.complete_with_tools`, builds its own throwaway `messages` list from a
  system prompt plus one user string, and persists nothing. Neither is a superset:
  chat has the session, RAG, memory, and citations; the agent has the tool loop,
  permissions, and audit. The merge means the agent loop's iteration becomes the
  unit inside `stream_reply` — history and context assembly come from the session
  as they do now, and a turn ends when the model stops asking for tools.
- [ ] **Decide whether tools are always-on or engaged by intent.** Always-on is
  what "watari IS the agent" argues for and is the recommended default — but note
  `PLAN.md:17` (small models misfire on large tool sets), and offering tool specs
  on every turn changes the prompt for turns that are pure conversation. The
  alternative — the model decides per turn, tools offered but rarely called — is
  the same thing in practice and is what the always-on shape already gives you.
  What's *not* on the table is a `/agent` slash command: that's the current split
  relocated into the REPL, and it still makes the user classify their own request
  before typing it.
- [ ] **Streaming is the hard part; scope it before starting.** `AgentLoop` is
  non-streaming by construction — `complete_with_tools` returns a whole turn
  because the loop must see the complete `tool_calls` array before it can act. Chat
  is streaming by construction. Fusing them means either streaming assistant text
  while buffering tool-call deltas until the turn resolves, or accepting that
  tool-using turns show a spinner where conversational turns stream. Interacts
  directly with *Assistant text alongside tool calls* below — that section's
  "should the caller see intermediate text" question stops being speculative here:
  a REPL running a multi-step loop needs to show progress, which is the streaming
  callback that item warns against building on spec. Land these together.
- [ ] **Persist tool calls and results to the session.** `AgentLoop` throws its
  `messages` list away when `run` returns, so tool history dies with the command.
  Under one interface it must round-trip through `SessionStore` — which means the
  session schema carries tool-call and tool-result messages, and `assemble_context`
  handles them. That is exactly the shape the *Context assembly* section's first
  item already flags as its highest priority ("never split a tool-call/tool-result
  pair" — the eviction loop currently breaks it). That item is a prerequisite here,
  not a neighbour: today the bug is unreachable because no tool history ever
  reaches the budgeter. This change makes it reachable on every turn.
- [ ] **Fold `--yolo` and the confirm prompt into the REPL.** `_run_agent`'s
  `confirm` (`cli.py:120-125`) already does a blocking `console.input` — it fits a
  REPL more naturally than the one-shot command it lives in now. `--yolo`
  (`cli.py:153`) is the awkward one: a session-lifetime flag on an interactive
  command is a blanket pre-approval of everything after it, which is the same
  anti-pattern the *Workspace root selection* section rejects for startup
  directory-trust prompts. Prefer a `/yolo` toggle inside the session, and note
  that section's item making `--yolo` refuse a CWD-derived workspace applies here
  unchanged.
- [ ] **Keep a non-interactive entry point.** Unifying the *interactive* surface
  doesn't mean deleting the ability to run one task and exit — scripts, CI, and
  `evals/` need it. The honest split is by interactivity (REPL vs. one-shot), not
  by capability (chat vs. agent), which is what makes this a rename and a
  demotion rather than a deletion. `watari run "<task>"`, or `chat --once`; either
  way it's the same unified path underneath, and the current `agent` command's
  distinguishing feature (tools) stops being a distinguishing feature at all.
- [ ] **Update the docs and the pitch.** `README.md` and `PLAN.md` present chat and
  agent as separate capabilities, and `cli.py`'s module docstring calls `chat` "the
  primary demo surface" while the agent goes unmentioned. The one-interface story
  is a better pitch than the two-command one — a local assistant that can act, not
  a chatbot shipped next to a task runner — but only once the code backs it.

## Context assembly (`src/watari/core/context.py`)

Gaps between the current v1 budgeter and a production-grade one, from review of
`assemble_context`. The skeleton (pinned system prompt, reserved response
headroom, recency-fill + chronological restore, single-message overflow guard)
is correct; these are the layers on top.

- [ ] **Never split a tool-call / tool-result pair.** The eviction loop can drop
  an assistant tool-call message while keeping its result (or vice versa), which
  most providers reject with an API error. Evict such pairs atomically. Highest
  priority — the Phase 4 tool-using agent produces exactly this history shape.
- [ ] **Budget retrieval against history.** `context_block` (RAG chunks) is
  appended to the system message and counted, but is not itself truncated — a
  large retrieval can eat the whole window and starve conversation history. Give
  retrieval and history separate sub-budgets instead of letting retrieval win by
  default.
- [ ] **Compact old turns instead of dropping them.** Currently oldest history is
  silently excluded. Prod summarizes evicted turns into a pinned running summary
  so the original task/instructions survive long conversations. Consider also
  structurally pinning the first user message.
- [ ] **Token-count fidelity.** `_message_tokens` uses a heuristic
  (`count_tokens + 4`) and doesn't account for tool schemas or images, which also
  consume the window. Either use the provider's official token-count endpoint
  (Anthropic `/v1/messages/count_tokens`) or apply a safety margin (e.g. budget ×
  0.9), since the estimate is soft and we currently rely on server-side
  truncation.

## RAG deletion (`src/watari/rag/service.py`, `src/watari/rag/store.py`)

Ingest only ever adds or updates. `ingest_path` walks the files currently on
disk and upserts them; nothing ever removes a document whose source file was
deleted or renamed. `_delete_document_rows` exists but is only called internally
by `upsert_document` when replacing a changed doc — it is not exposed anywhere.

- [ ] **Prune documents whose source file is gone.** Deleting `notes.md` and
  re-running ingest leaves its chunks in the DB forever, where they keep
  surfacing in retrieval as stale results citing a file that no longer exists. A
  rename is worse: it reads as "old file deleted + new file added", so both
  copies end up indexed. Add a reconcile pass that diffs the `source_path` set in
  `documents` against what `discover()` finds on disk and deletes the vanished
  ones. Note this is also step 4 of the standard content-addressed upsert
  pattern.
- [ ] **Expose a per-document delete.** Promote `_delete_document_rows` to a
  public `delete_document(source_path)` on `RagStore`, plus a CLI surface (e.g.
  `watari forget <path>`), so a doc can be removed without re-ingesting the
  whole corpus. Useful on its own and the primitive the prune pass needs.
- [ ] **Guard the three-table cleanup invariant.** `chunks_fts` and `vec_chunks`
  are not FK-linked to `chunks` — they share the id space by convention only, so
  deletes must clean all three explicitly. Today `_delete_document_rows` gets the
  ordering right (fts/vec rows first, then the `documents` delete cascades away
  `chunks`), but that correctness rests on `PRAGMA foreign_keys = ON` being set
  per connection, and nothing tests it. Any connection opened without the pragma
  silently orphans rows. Add an integrity check (assert `chunks`, `chunks_fts`,
  and `vec_chunks` row counts agree) and a test, so drift is caught rather than
  silently corrupting retrieval. Applies to any new delete path too.

## Multi-user / RAG architecture (`rag/store.py`, `docs/adr/001-vector-store.md`)

- [ ] **Watari is currently single-user.** The store is one local SQLite `.db`
  file with no per-user data isolation, auth, or tenant scoping — chunks,
  sessions, and vectors share one id space (see ADR-001). This is an intentional
  local-first design choice, not a bug; multiple people can each run their own
  DB, but there is no shared/concurrent multi-tenant support. If multi-user ever
  becomes a goal, revisit the RAG choices: user isolation would likely require
  tenant-scoped schema changes and possibly a different vector store (the
  server-based Qdrant path ADR-001 deliberately rejected for the single-user
  pitch). Not scheduled — flagged for a future revisit.

## fastembed model cache (`Dockerfile`, `docker-compose.yml`, `.github/workflows/`)

Embedding *inference* is fully local — `FastEmbedEmbedder` runs the ONNX model
in-process on CPU and no text ever leaves the machine. But the `TextEmbedding`
constructor downloads the `BAAI/bge-small-en-v1.5` weights (~130MB) from Hugging
Face on first use, into `~/.cache/fastembed`. Nothing caches that download, so
every cold run pays it. Contrast Ollama, which already gets both an
`actions/cache` step in `evals.yml` and an `ollama-models` volume in compose —
fastembed just never got the same treatment.

- [ ] **Cache the weights in CI.** Both jobs construct a real embedder: `quality`
  via the unit tests (`test_rag_store.py`, `test_memory.py` build a real
  `FastEmbedEmbedder`), and `evals` via `harness.py`. Each therefore pulls 130MB
  from huggingface.co on every run, which makes merge-gating CI depend on HF
  being reachable. Add an `actions/cache` step on `~/.cache/fastembed` keyed by
  model name, mirroring the existing Ollama cache step.
- [ ] **Bake the weights into the Docker image.** The build never instantiates
  `TextEmbedding`, so a fresh container downloads on its first embed. Warm the
  cache in the runtime stage as the `watari` user. Costs ~130MB of image size and
  buys cold-start-ready, fully-offline containers — which is the story the README
  tells.
- [ ] **Or: persist the cache in compose.** Cheaper alternative to baking it in if
  image size matters. The `watari-data` volume covers `/home/watari/.watari`, not
  `/home/watari/.cache`, so the download repeats on every container recreate. A
  `fastembed-cache` volume at `/home/watari/.cache` fixes it. Largely redundant
  with the Dockerfile item — that one is the stronger guarantee since it also
  helps anyone running the image outside compose.

## Citation integrity (`src/watari/rag/cite.py`, `src/watari/evals/metrics/retrieval.py`)

What `cite.py` does is range-checking, not grounding verification. It catches one
failure mode — a marker like `[7]` when only 5 chunks were shown — which is the
least common and least harmful of the citation failures. The one that matters is
misattribution: the model cites `[3]` for a claim that chunk 3 does not support.
Every check in the module passes, and the user gets a confident answer footnoted
to a real document, which is worse than an uncited hallucination because the
citation manufactures trust.

- [ ] **Fix `citation_validity` — it currently rewards the failure it should
  catch.** With no citations at all the metric returns `1.0` ("nothing invalid
  was emitted"), so a model that ignores the retrieved context entirely and
  answers from parametric memory scores a perfect 1.0 and sails through the `0.8`
  gate in `evals/thresholds.json`. Return `0.0` for the empty case, or drop those
  cases from the denominator and track coverage separately. Highest priority —
  it's a cheap change and the metric is actively backwards today.
- [ ] **Add a `citation_coverage` metric.** Validity is a degenerate form of
  precision; nothing measures recall. Track the fraction of answer sentences
  carrying at least one `[n]` marker. Coverage is where "the model ignored the
  context" shows up, and precision + coverage together are a defensible pair.
- [ ] **Score faithfulness offline via entailment.** Per (sentence, cited-chunk)
  pair, check whether the chunk actually entails the sentence — either an NLI
  cross-encoder (DeBERTa-NLI family, tens of ms) or an LLM judge (more accurate,
  much slower). Keep this in the eval harness, *not* the serving path: a second
  full LLM pass per turn is not affordable inline. That tradeoff is the point —
  see the ADR item below.
- [ ] **Close the misattribution loophole in `system.md`.** The line "never invent a
  citation number that was not provided" defines correctness as *membership in the
  provided set*, not *correspondence to the supporting chunk* — so a model that
  cites `[3]` for a claim chunk 1 supports has fully complied. It even nudges toward
  that failure: the prompt asks for grounding *and* citations, so when a claim isn't
  cleanly supported by any single chunk, omitting the marker feels like disobeying,
  inventing one is explicitly forbidden, and picking a plausible in-range number is
  the only unpunished path. Restate the rule per-claim and directional — each `[n]`
  must point to the source that supports the sentence it follows, *even if the number
  is one of the provided ones* — and make "no citation" an explicitly legal output so
  abstention has somewhere to go. Prompt-only and cheap; land it with the abstention
  item below, which touches the same instruction from the enforcement side. Note the
  prompt fix is not *verifiable* until the entailment metric above exists — range
  checks can't tell a right mapping from a wrong one.
- [ ] **Reconsider `strip_invalid_citations`.** It deletes the `[7]` marker but
  keeps the sentence, so "The cap is $2,500 [7]" becomes "The cap is $2,500." — a
  hallucinated claim laundered into an uncited assertion that reads like ordinary
  model output. The stripping happens before `render_sources`, so `chat.py` can
  never surface it to the user either. Either drop the whole sentence or keep the
  marker and render it visibly (`[?]`) so the failure stays legible.
- [ ] **Enforce abstention instead of asking for it.** `_CITE_INSTRUCTION` and
  `system.md` politely ask the model to say when the context doesn't contain the
  answer. A prompt instruction is a suggestion; a retrieval-score threshold is a
  guarantee. Refuse when retrieval scores are weak or entailment fails on most
  sentences.
- [ ] **Evaluate Anthropic's native citations API.** Passing documents and getting
  back cited spans with character offsets — enforced by the serving layer rather
  than trusted from the model — deletes most of this problem class and most of
  `cite.py` with it. Worth checking before building more machinery on top of the
  regex approach. The DIY alternative is constrained generation: have the model
  emit `{claim, chunk_id, quote}` and verify `quote` is a verbatim substring of
  the chunk.
- [ ] **Write the ADR.** The gap between range-checking inline and entailment-
  checking offline is a deliberate cost tradeoff, not an oversight. Record it in
  `docs/adr/` alongside the existing ones so the limitation reads as judgment
  rather than a hole.

## Retrieval context budget (`src/watari/rag/retrieve.py`, `src/watari/rag/cite.py`)

Retrieval is bounded by a *count* — `retrieval_top_k` — and `format_context_block`
concatenates whatever those k chunks happen to be. Nothing bounds the result in
*tokens*. A `top_k=5` where every chunk is a long table is a wildly different
prompt from five one-liners, so the size of the context block is currently a
property of the corpus rather than a thing we control. Production RAG budgets
tokens, not chunks. Complements the "Budget retrieval against history" item under
Context assembly: that one is about how the window is *divided* between retrieval
and history; these are about how retrieval spends whatever slice it gets.

- [ ] **Fit the context block to a token budget.** Add a
  `fit_to_budget(chunks, max_tokens) -> list[RetrievedChunk]` between `retrieve`
  and `format_context_block`, plus a `context_budget_tokens` setting alongside
  `retrieval_top_k`. Highest priority of this group — without it every item below
  has nothing to enforce, and a single pathological document can blow the context
  window and 400 the request in prod.
- [ ] **Drop whole chunks; never truncate mid-chunk.** Chopping a chunk in half
  produces worse answers than omitting it, and a half-shown `[4]` is exactly the
  hallucinated-citation failure `validate_citations` exists to catch — we'd be
  manufacturing it ourselves. Walk the ranked list and stop; if truncating at
  all, cut on a chunk boundary and keep the `heading_path` so the citation still
  resolves.
- [ ] **Make `top_k` a candidate pool, not a cap.** The standard shape is
  over-retrieve (20–50) → rerank → fit to budget, which is strictly better than
  top-5-and-hope, but only pays off with a reranker in front of it. Coupled to
  the reranking work; not worth doing alone.
- [ ] **Add a relevance floor.** A score threshold below which a chunk isn't worth
  its tokens. Weakly-relevant context measurably *hurts* — distractors and the
  lost-in-the-middle effect mean accuracy is non-monotonic in context length. The
  budget is a ceiling; the floor is a separate and equally load-bearing knob, and
  it's the same threshold the abstention item under Citation integrity wants.
- [ ] **Eval answer quality as a function of budget.** The harness can sweep
  `context_budget_tokens` and plot quality against it. This is the artifact that
  justifies whatever number we pick, and it's where the non-monotonicity above
  either shows up in our corpus or doesn't.

## Grounding posture (`src/watari/core/chat.py`, `src/watari/evals/metrics/judge.py`)

`_CITE_INSTRUCTION` picks **strict grounding**: answer from the retrieved context
only, abstain otherwise. That's a deliberate posture, not an accident, and it's
the right default for a corpus of *personal* docs — if retrieval misses "what's my
current role?", a blended model invents a plausible role from context clues while a
strictly-grounded one says the context doesn't cover it. But the posture is
currently *entangled with the metric*: `_FAITHFUL_PROMPT` scores a claim
unsupported whenever "the CONTEXT is silent", and `thresholds.json` gates CI at
`faithfulness: 0.6`. So a true-but-unretrieved fact from parametric knowledge reads
as a hallucination to our own harness. The prompt and the metric are two halves of
one decision and cannot be changed independently.

The three postures, for the record: **strict grounding** (enterprise/regulated RAG
— Glean, Copilot-over-your-docs; optimizes away plausible unsourced assertion),
**blended with provenance** (Perplexity, ChatGPT w/ search — retrieved docs win for
the claims they cover, parametric knowledge fills gaps, `[n]` markers make the
boundary legible), and **hybrid with an escape hatch** (ground by default, allow
stepping outside the context only with an explicit "the docs don't cover this").

- [ ] **Sharpen `_CITE_INSTRUCTION` without changing posture.** "Answer using only
  the retrieved context" literally forbids general knowledge for *reasoning,
  arithmetic, and language*, which is incoherent and pushes small models into
  over-refusal. It also says nothing about partial coverage, which is the common
  case. Separate facts-about-the-user (must be grounded, must be cited) from
  general reasoning/definitions/background (allowed, must not be cited), and name
  the partial-coverage behavior explicitly. Cheap, posture-preserving, keeps
  faithfulness honest — do this one regardless of how the bigger question lands.
- [ ] **Decide: stay strict, or move to blended-with-provenance?** The argument for
  blending is that citations already make provenance explicit. The argument against
  is that *absence* of a citation is a weak signal — users read the answer, not
  citation density, and an uncited sentence reads exactly as authoritative as a
  cited one sitting next to it in the same paragraph. Perplexity gets away with it
  because their UI visually distinguishes uncited spans and open-web Q&A is
  low-stakes; neither is true here.
- [ ] **If blending: split the metric in two first.** `faithfulness` is currently
  doing two jobs, which is why loosening the prompt would silently tank it. Score
  faithfulness only over claims carrying an `[n]` (are cited claims actually
  supported?), and add an attribution/coverage metric for what fraction of
  user-specific claims carry a citation at all. This is the same metric as the
  `citation_coverage` item under Citation integrity — the two should land together.
  Also needs the model to mark which claims it's attributing, and a re-baseline of
  `evals/baselines/*.json`.

## RAG observability (`src/watari/core/chat.py`)

The `chat.reply_complete` log line records `rag=bool(chunks)` — i.e. "did retrieval
actually contribute context", not "was RAG requested for this turn". That's the more
useful of the two facts, but the field name claims the other one, and the two diverge
in exactly the cases worth debugging: `use_rag=True` with no retriever wired up, and
`use_rag=True` where retrieval returned nothing. Both currently log `rag=false` with
no way to tell them apart from a turn where the caller never asked for RAG at all.

- [ ] **Log request and outcome as separate fields.** Replace the single `rag` boolean
  with `rag_requested=use_rag` and `rag_chunks=len(chunks)`. `rag_requested=true,
  rag_chunks=0` then reads as a self-explaining line instead of an absence, and the
  chunk count is strictly more information than the boolean it replaces. Grep for
  consumers of the existing `rag` field before renaming — if anything downstream keys
  off it, keep it alongside the new fields rather than pulling it out from under them.
- [ ] **Log the zero-retrieval case at all.** The `chat.citations` line — the only
  place `chunks_retrieved` is emitted — sits inside `if chunks:`, so a turn where RAG
  was on and retrieval came back empty produces no retrieval telemetry whatsoever. That
  is the single most interesting turn to have a log for, and it's the one turn we're
  silent about. Subsumed by the item above if `rag_chunks` lands on `reply_complete`;
  worth confirming rather than assuming.

## Multi-hop RAG-QA coverage (`evals/datasets/rag_qa_v1.jsonl`, `src/watari/evals/runner.py`, `src/watari/evals/metrics/retrieval.py`)

The eval set never exercises a question that needs facts from more than one chunk.
All 15 rows in `rag_qa_v1.jsonl` (and all 24 in `retrieval_v1.jsonl`) have a
single-element `relevant` list, so multi-hop retrieval and cross-chunk synthesis
are completely untested. The `relevant` field is already a *list* in the schema,
so no format change is needed — the gap is data plus a metric that grades it
strictly. Rows that look multi-part (qa-001 "where does she live *and* grow up")
draw both facts from the *same* chunk, so they're multi-fact, not multi-chunk.

- [ ] **Add genuine multi-hop rows.** Author QA cases whose `reference_answer`
  requires two or more chunks from *different* headings/files (e.g. "what does
  Mara use Rust for, and what rule governs merging that code?" spans
  `Tooling Preferences` + a project section). Tag them `multi-hop` so they can be
  sliced out. This is the load-bearing item — without the data, the metric work
  below has nothing to grade.
- [ ] **Add an all-or-nothing recall variant.** `recall_at_k` is proportional
  (`len(top & relevant) / len(relevant)`), so a 2-chunk row that retrieves only 1
  scores 0.5, averages into the mean, and never trips the `recall@5: 0.75` /
  `recall@10: 0.85` gate in `thresholds.json` on its own. A multi-hop question
  answered from half its evidence should count as a miss. Add a strict
  `full_recall_at_k` (1.0 only if *all* relevant ids are in top-k) and gate the
  `multi-hop` slice on it — the proportional metric hides exactly the failure
  multi-hop rows exist to expose.
- [ ] **`mrr` is structurally blind to multi-hop.** `reciprocal_rank` returns on
  the *first* relevant hit, so it can't distinguish "found 1 of 2" from "found
  2 of 2". Fine as-is for single-chunk rows, but don't report `mrr` as evidence a
  multi-hop question retrieved everything — pair it with the strict recall above.
- [ ] **Raise the retrieval pool above the largest `relevant` set.** `runner.py`
  calls `retrieve(top_k=10)` then slices at 3/5/10. A multi-hop row whose second
  gold chunk ranks past 10 is unrecoverable at any k, and `recall@3`/`@5` can't
  see a gold chunk sitting at rank 8. Ensure `top_k` comfortably exceeds
  `max(len(relevant))` across the dataset, and consider the over-retrieve → rerank
  → fit shape already flagged under *Retrieval context budget*.

## Kappa singularity guard (`src/watari/evals/metrics/calibration.py`)

`cohens_kappa` special-cases the undefined case with `if expected >= 1.0`, but the
real hazard is the *denominator* `1 - expected` being near zero, and that guard is
one-sided. `expected` is a probability, so `> 1.0` is mathematically impossible;
the `>= ` (rather than `== `) is purely there to absorb float rounding that tips a
true-1.0 to `1.0000000002`. But the symmetric drift — rounding a true-1.0 *down* to
`0.9999999998` — slips past the guard and divides by `~2e-16`, producing a garbage
kappa instead of the correct `1.0`. Not reachable on the current calibration set
(human labels are pinned 15/15 balanced, so `expected` can't approach 1 from the
human side), so this is correctness hygiene for `cohens_kappa` as a *reusable*
utility, not a live bug.

- [ ] **Guard the denominator, not `expected`.** Replace `if expected >= 1.0` with a
  check on `1 - expected` against an epsilon (`denom = 1 - expected; if denom <
  1e-9: return 1.0 if observed >= 1.0 else 0.0`). This covers below-1.0 drift the
  current form misses and names the actual singularity. Behavior is identical on
  every input the calibration set can produce; check the unit tests first in case
  any pin the exact `>= 1.0` branch. Low priority — the balanced human labels make
  the missed case unreachable here today.

## Gating the injection suite (`src/watari/evals/gate.py`, `evals/thresholds.json`)

The injection suite is report-only, and the reason is mechanical rather than
principled: `check_gate` hard-codes `m.value < floor`, so every threshold is a
floor and every metric has to be higher-is-better. ASR is lower-is-better, so it
has nowhere to sit in `thresholds.json` and regressions are caught only by a human
reading the committed baseline. Nothing structurally prevents gating it — the gate
just can't currently express "this metric must stay *below* a bound".

- [ ] **Decide how to gate `asr_mitigated`.** Approach not settled; the options
  trade off legibility of `thresholds.json` against how much the gate has to
  learn. Worth doing — injection resistance is the security pillar's headline
  claim and it is currently the one suite CI can't defend.
- [ ] **Gate the mitigated number, not the unmitigated one.** `asr_unmitigated`
  *falling* is not good news — it means the attack corpus stopped landing on the
  raw model, which makes the before/after delta measure nothing and the mitigation
  look effective for free. If it is gated at all it wants the opposite direction
  from `asr_mitigated`, which is a separate decision from the one above.
- [ ] **Grow the attack corpus before tightening any bound.** With n=12 the
  resolution is ~0.083 per case, so any bound is quantised to whole attacks and a
  strict one is a coin flip on a nondeterministic local model. `threat-model.md`
  already concedes "ASR is > 0 in some runs" — the corpus size, not the gate, is
  what caps how strict the bound can honestly be.

## Tool isolation (`src/watari/security/sandbox.py`, `docs/threat-model.md`)

The sandbox is a pure-Python jail: it resolves symlinks and rejects paths outside
the workspace, and it is red-teamed in `tests/security/`. But it bounds *where* a
tool can act, not *what* it can do, and it is enforced in the same process and
with the same ambient authority as the rest of the app — a bug in the path logic,
or any future tool that reaches the filesystem without going through it, is an
escape. The threat model names this honestly under "Deliberately NOT mitigated":
container/seccomp isolation is "real value, real cost; out of scope for the
current phase". Tracking it here so the deferral stays a decision rather than an
omission.

- [ ] **Run tools under OS-level isolation.** Execute file tools in a container or
  under a seccomp/landlock profile so the workspace bound is enforced by the
  kernel rather than by our own path checks. This makes the jail defense-in-depth
  instead of the only line, and it is what makes the "no shell tool" cut
  ([ADR-002](adr/002-no-shell-tool.md)) survivable if a future phase ever wants to
  reintroduce exec. Costs real complexity — a container runtime dependency, or a
  Linux-only syscall filter — which is exactly why it was deferred; revisit when
  either the tool surface grows or the workspace stops being the user's own
  machine.
- [ ] **Update the threat model when it lands (or doesn't).** The residual-risk
  cell for path traversal and the "Deliberately NOT mitigated" bullet both assert
  "not a container/seccomp sandbox". If this ships, both need to change; if it is
  rejected for good, record the reasoning in an ADR so the claim keeps its
  justification.

## Silent absolute-path remapping (`src/watari/security/sandbox.py`, `src/watari/agent/tools/files.py`)

`Sandbox.resolve` strips the leading anchor off an absolute path and joins the
remainder under the workspace, so `/etc/new_file` silently becomes
`<workspace>/etc/new_file` and returns *success*. The security property is fine —
remapping and rejecting are equally contained, and nothing escapes either way.
The problem is the lie told to the caller: the agent asked for `/etc/new_file`,
was told it succeeded, and now believes a false thing about the filesystem. That
belief propagates — the model reports the wrong path to the user, reasons about
it in later turns, and a read-back from `/etc/new_file` misses for reasons
nothing in the trace explains. chroot has roughly these semantics, but there the
caller genuinely *is* inside a new root; here the caller is an LLM reasoning
about a real filesystem it thinks it can see, which is exactly the case where the
substitution misleads. `write_file` makes it worse: `target.parent.mkdir(parents=True)`
means the request also *creates* a spurious `etc/` directory in the workspace.
Secondary: `/etc/foo` and `etc/foo` are distinct requests that collide on one
file, so a legitimate top-level `etc/` is indistinguishable from an absolute-path
request.

- [ ] **Reject absolute paths instead of remapping them.** Replace the strip-and-join
  at `sandbox.py:47-48` with a `SandboxError`. No caller wants the current
  behavior: all four call sites (the three tools in `files.py`, plus
  `agent_eval.py:60`) already treat paths as workspace-relative, every
  `path` Field description says "Workspace-relative path", and all five
  `assert_file` values in `agent_v1.jsonl` are plain relative paths. Cost is one
  extra turn when the model guesses wrong — cheap next to the model confidently
  believing a false thing. Note the comment at `sandbox.py:45-46` presents the
  remap as deliberate, so this reverses a stated decision; record it in an ADR or
  at minimum restate the comment.
- [ ] **Give the tools an error channel for it first.** This is what makes the change
  bigger than a four-line swap. `read_file` / `list_dir` return `"error: ..."`
  strings for a missing file, but `SandboxError` *raises* through `Tool.run` — so
  rejecting absolute paths would route a recoverable mistake (model used the wrong
  path convention) down the same path as a genuine escape attempt. The model can
  read a returned error and retry; it can't read an exception. Decide whether the
  tool layer catches `SandboxError` and returns `"error: paths must be
  workspace-relative"`, or whether escapes and bad-convention paths stay
  deliberately indistinguishable. Land this before the rejection above.
- [ ] **Rewrite `test_absolute_path_is_remapped_into_jail_not_escaped`.** The one
  test that actually pins the behavior (`test_sandbox.py:59-64`). It sits under
  `TestBlockedEscapes` and its name says "not escaped", so its *intent* is the
  containment property — the remap is just the mechanism it was written against.
  Rejecting satisfies the intent and fails the assertion. Reframe it as
  `pytest.raises(SandboxError)`. Leave `test_tilde_is_a_dirname_not_home`
  (`test_sandbox.py:66-70`) alone — `~/secrets` is already relative, so it's
  unaffected by an absolute-only rejection and its no-`expanduser` guarantee is
  worth keeping independently.
- [ ] **Update the threat model's mechanism, not its claim.** `threat-model.md:31`
  credits "null-byte and absolute-path handling" as a mitigation and `:53-54` cites
  the sandbox tests for "absolute paths". Both stay true under rejection — only the
  mechanism changes. Cheap, but don't let the docs describe a remap that no longer
  happens.

## Assistant text alongside tool calls (`src/watari/agent/loop.py`)

The loop treats "has tool calls" and "has text" as mutually exclusive, but the
OpenAI-compatible shape allows both in one turn — a `content` string *and* a
`tool_calls` array. `run` returns only when `turn.tool_calls` is empty
(`loop.py:69`), so a turn carrying both takes the tool branch and the text is
dropped on the floor. `_assistant_tool_message` hardcodes `"content": ""`
(`loop.py:142`), so the text doesn't even survive into the message history. The
`AgentOutcome` dataclass reinforces the assumption: a single terminal `answer`
field, no room for anything said along the way.

- [ ] **Preserve intermediate text in the replayed history.** Pass `turn.content`
  through `_assistant_tool_message` instead of the hardcoded `""`. This is the
  load-bearing item and it's a one-line change: today, if the model narrates its
  plan while calling a tool ("I'll read the config first, then check the
  schema"), that reasoning is erased from its own context on the very next
  iteration — it re-derives the plan each turn, or drifts from it. This is a
  correctness bug, not a missing feature, and it's independent of whether we ever
  surface the text to the caller. Check what the provider does with an empty
  `content` alongside `tool_calls` first — some reject `""` where they'd accept a
  null/omitted field, which would explain why it was written this way.
- [ ] **Decide whether the caller should see it at all.** Separate and bigger
  question. `AgentOutcome` has one `answer`, so intermediate text has nowhere to
  go — options are a `transcript: list[str]` field (cheap, batch-shaped, matches
  how `tool_calls`/`denied` already accumulate) or a streaming callback (right
  answer for a CLI that wants to show progress, but pushes async plumbing into
  every caller). Don't build the streaming path on spec; `chat.py` renders a
  final answer today and nothing is asking for progress yet.
- [ ] **Pin the both-at-once shape in a test.** Nothing in the suite exercises a
  turn with `content` *and* `tool_calls` populated, which is why the drop is
  silent. A fake provider returning both would have caught it. Worth adding
  alongside the fix rather than after — it's the only thing that keeps the
  hardcoded `""` from creeping back.

## `web_search` is a stub (`src/watari/agent/tools/web_search.py`)

The tool is fully wired as a *tool* — `WebSearchArgs`, `Risk.EXECUTE` (so it needs
confirmation), and registration gated on `enable_web_search` in `service.py:30-31`,
off by default per `config.py:101`. What's missing is the search: `web_search()`
returns a fixed "no backend is configured" string and never makes a request.
`httpx` is already a dependency (`pyproject.toml:22`) but is unused here. The
module docstring calls a real backend "a Phase-6 add" — Phase 6 has since landed
(`a962d5c`) without it, so the deferral needs to be re-stated as a decision or
closed out. Watari has **no network egress at all** today, which is a stronger
guarantee than the docs currently claim and worth deciding deliberately.

- [ ] **Decide whether to implement it or delete it.** The two honest end-states are
  a real backend behind the existing opt-in, or removing the module and saying
  Watari is offline-only. The current middle — a registered tool that the model can
  call, burn a turn on, and get a refusal string from — is the worst of both: it
  costs a tool slot in a set deliberately kept small (`PLAN.md:17` — small models
  misfire on big tool sets), and a confirmation prompt for `EXECUTE` risk that
  guards nothing. Note `PLAN.md:70` already lists `web_search` third in the cut
  order, which argues for deletion.
- [ ] **Fix the docs that describe egress as real.** Three places assert a network
  path that does not exist: `threat-model.md:24` ("`web_search` (opt-in, off by
  default) is the only egress path"), `threat-model.md:34` (data-exfiltration
  residual risk — "a successful injection could try to exfiltrate via a crafted
  query"), and `README.md:182`. All three read as describing a live capability; the
  exfil risk in particular is currently hypothetical. This is the load-bearing item
  regardless of how the decision above lands — the docs overclaim either way, and
  overclaiming *risk* is the odd direction to be wrong in. If the tool stays a stub,
  say so where it's described.
- [ ] **If implementing: the threat-model claims become real, not theoretical.** The
  exfil-via-crafted-query row stops being hypothetical the moment a backend lands,
  and it interacts directly with the injection suite — an injected instruction that
  smuggles workspace contents into a search query is exactly the attack the corpus
  should cover, and it currently can't land because there's nowhere for it to go.
  Add attack rows alongside the backend, not after (see *Gating the injection
  suite*). Also decide what the audit log records — query text is the exfil channel,
  so logging it is both the detection mechanism and, if the log is ever shared, a
  second copy of the leak.

## Workspace root selection (`src/watari/config.py`, `src/watari/cli.py`)

`workspace_path` defaults to `~/.watari/workspace` — a directory the user must
populate by hand before the agent is useful for anything they actually care
about. Every comparable tool (Claude Code, Cursor, Aider) defaults to the
invocation directory instead, because that's the only default where the agent's
world matches what the user is already thinking about. The counter-pressure is
that CWD is *ambient*: the jail root stops being a directory that was
deliberately populated and becomes wherever the shell happened to be, which
silently widens the blast radius of the injection threat the security pillar
exists to bound. The items below are the shape that gets the ergonomics without
laundering that widening past the threat model.

- [ ] **Default the agent workspace to CWD, via an explicit `--workspace PATH`.**
  Flag on `watari agent` defaulting to `Path.cwd()`. Explicit beats ambient: the
  flag is what makes a CWD-derived root distinguishable from a chosen one, which
  every item below keys off. Load-bearing — the rest of this section is guardrails
  around this change and shouldn't land without it.
- [ ] **Print the resolved jail root; don't prompt for it.** A startup "can I use
  this folder? [y/N]" is *less* protective than what's already there: `cli.py:120-125`
  confirms every tool call with its actual name and args, whereas a startup gate is
  one blanket yes covering everything after it, and users click through startup
  prompts reflexively. Trading specific informed consent for a vague one only feels
  safer. Print the root as one line of status output — users need to *know*, not to
  *approve*. (Claude Code's directory-trust prompt guards a different threat:
  malicious `.claude/` config auto-loading on open. Watari has no such surface.)
- [ ] **Refuse dangerous roots outright.** `$HOME`, `/`, and probably anything above
  a git root. A hard refusal beats a prompt precisely because it can't be clicked
  through — which is the same reason the item above rejects the startup gate.
- [ ] **Make `--yolo` refuse a CWD-derived workspace.** `--yolo` (`cli.py:153`)
  auto-approves every tool call; combined with an ambient root it means "write
  anywhere under `$PWD`, unattended". That specific pairing is the one that hurts —
  require an explicit `--workspace` before `--yolo` will run. Depends on the flag
  existing to tell the two cases apart.
- [ ] **Consider a first-write confirmation for CWD-derived roots.** The one place a
  real prompt earns its keep: specific, fires when the risk becomes concrete, and
  can't be pre-clicked at startup. Lower priority than the refusals above — those
  are unconditional, this one is a judgment call about friction.
- [ ] **`ensure_workspace` should stop creating the root.** `config.py:130-131` does
  `mkdir(parents=True, exist_ok=True)`, which is fine while the root is a
  Watari-owned directory but wrong once it comes from user input: a typo'd
  `--workspace` (or a UI-supplied path) silently creates a directory tree instead of
  erroring. Once the workspace can come from outside, "must exist" is the right
  posture.
- [ ] **UI (unbuilt) needs an explicit folder picker.** With no CWD to inherit, a
  picker is the only honest option — and it maps onto `workspace_dir` already being
  an override field, so no schema change. This is where the `ensure_workspace` item
  above stops being hygiene and becomes load-bearing.
- [ ] **Update the threat model.** `threat-model.md` currently reasons about a
  workspace that was deliberately populated. A CWD default changes the blast radius
  of a successful injection — from a scratch directory to whatever the user's shell
  was sitting in — and that's a real escalation, not a UX tweak. It should read as a
  decision with guardrails, not go unmentioned. Also touches *Tool isolation* above:
  a user-owned CWD is exactly the "workspace stops being a scratch directory"
  condition that item names as the trigger for revisiting OS-level isolation.

## Tokenizer choice (`src/watari/core/tokens.py`)

- [ ] **Revisit `cl100k_base` for Claude token counts.** `tokens.py` loads
  `cl100k_base`, which is an OpenAI tokenizer — so counts are only an
  approximation for Claude models and won't match exactly. If accurate counts
  matter for prompt/chunk sizing, switch to the Anthropic token-counting endpoint
  (`/v1/messages/count_tokens`). Related to the Token-count fidelity item above.
