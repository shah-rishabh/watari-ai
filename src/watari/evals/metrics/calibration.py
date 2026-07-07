"""Judge calibration: Cohen's kappa between judge labels and human labels.

We hand-label a small set of (answer, context) pairs as faithful/unfaithful, run
the judge over the same set, and report Cohen's kappa — chance-corrected
agreement. Reporting *how well we trust our own judge* is the part almost no
portfolio repo does, and it's why the judge metrics can be believed. See
``docs/evals.md``.
"""

from __future__ import annotations

from collections.abc import Sequence


def cohens_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Cohen's kappa for two binary label sequences.

    Returns 1.0 for perfect agreement, 0.0 for chance-level, negative for
    worse-than-chance. When both raters are perfectly constant and identical,
    agreement is perfect (1.0); if they are constant but expected agreement is
    also 1.0, kappa is undefined and we return 1.0 iff they match.
    """
    if len(a) != len(b):
        raise ValueError("label sequences must be equal length")
    n = len(a)
    if n == 0:
        return 0.0

    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n

    # Expected agreement by chance, from each rater's marginal rate of True.
    pa_true = sum(a) / n
    pb_true = sum(b) / n
    expected = pa_true * pb_true + (1 - pa_true) * (1 - pb_true)

    if expected >= 1.0:
        # Both raters constant; kappa is undefined — treat as perfect iff equal.
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1 - expected)
