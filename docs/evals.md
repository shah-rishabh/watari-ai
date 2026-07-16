# Evaluation methodology

Watari's central claim is *measured, not vibes*: every retrieval and generation
capability carries a number from a hand-rolled eval harness. This document
explains how those numbers are produced and how far to trust them.

## Why a custom harness

The metrics here (recall@k, MRR, an LLM-judge for faithfulness) are each well
under 50 lines. Writing and **unit-testing the metric functions themselves** —
e.g. MRR on synthetic rankings with known answers — demonstrates more than
`pip install ragas` would, and it avoids RAGAS/deepeval's implicit assumption of
a GPT-4-class judge, which we do not have locally. See
[`tests/unit/test_eval_metrics.py`](../tests/unit/test_eval_metrics.py) for the
metric-math tests.

## The corpus and golden data

The eval corpus ([`evals/corpora/`](../evals/corpora/)) is a small **fictional**
document set about an invented person. Fiction is deliberate: answerability is
fully controlled (we know exactly which chunk answers each question) and there
are no privacy concerns in committing it.

Golden datasets are JSONL, one `EvalCase` per line, hand-verified:

- **`retrieval_v1.jsonl`** (24 cases) — each question maps to the chunk(s) that
  should be retrieved, identified by `(source_path, heading_path)` rather than a
  fragile chunk index. A test
  ([`test_golden_heading_paths_resolve_to_real_corpus_chunks`](../tests/unit/test_eval_gate.py))
  guards against drift by asserting every referenced chunk actually exists.
- **`rag_qa_v1.jsonl`** (15 cases) — each carries a reference answer and the
  facts a faithful answer must contain.
- **`judge_calibration_v1.jsonl`** (30 cases) — hand-labeled (answer, context,
  faithful?) pairs for calibrating the judge (below).

A `smoke`-tagged subset runs in CI on a tiny model; the full set runs locally.

## Metrics

| Suite | Metric | How it's computed |
| --- | --- | --- |
| retrieval | recall@{3,5,10} | Fraction of golden chunks in the top-k (deterministic). |
| retrieval | MRR | Mean of 1/(rank of first relevant hit). |
| rag-qa | faithfulness | LLM-judge: decompose the answer into atomic claims, judge each against the retrieved context, score = fraction grounded. |
| rag-qa | answer_relevance | LLM-judge: a 3-point rubric (does the answer address the question?). |
| rag-qa | citation_validity | Fraction of `[n]` markers that fall within the retrieved set (deterministic). |

**Why claim-decomposition for faithfulness.** Asking a 3B model for a single
1–10 faithfulness score is noisy. Decomposing into atomic claims and asking a
simple yes/no per claim turns one hard judgment into several easy ones, which is
far more stable on a small local judge.

## Judge calibration (trusting the judge)

An LLM judge is only useful if we know how much it agrees with a human. We run
the judge's per-claim faithfulness check over the 30-case calibration set and
report **Cohen's kappa** (chance-corrected agreement) and raw accuracy:

```
watari evals calibrate
```

On `qwen3.5:4b` the judge reproduced the human labels exactly
(kappa = 1.00, accuracy = 1.00). **Caveat, stated plainly:** the calibration
cases are deliberately clear-cut (obvious supported/contradicted pairs), so a
perfect score here establishes the judge is not broken — not that it is reliable
on genuinely ambiguous claims. A harder calibration set (paraphrase, partial
support, numeric near-misses) is the natural next iteration, and its kappa is the
number that would actually bound trust in the judge metrics.

## Running

```bash
watari evals run --suite all              # full suites, local model
watari evals run --smoke --gate           # CI subset, fail on regression
watari evals calibrate                    # judge-vs-human kappa
```

Each run writes `results.json` and `results.md` to `eval-results/`.

## Regression gating

[`evals/thresholds.json`](../evals/thresholds.json) declares an **absolute floor**
per metric. The gate fails (exit 1) if any metric drops below its floor. Floors —
not exact-match against a baseline — are deliberate: small local models are
nondeterministic, so an exact-match gate would be permanently flaky. A regression
is a real drop below the floor, not run-to-run noise. Committed baselines
([`evals/baselines/`](../evals/baselines/)) record the trend for humans; they are
updated by explicit, reviewable commits.

## Baseline (qwen3.5:4b, local)

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
<!-- EVAL_TABLE_END -->

Judge calibration: kappa = 1.00, accuracy = 1.00 (n = 30, clear-cut cases).
