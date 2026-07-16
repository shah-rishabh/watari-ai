"""Retrieval metrics — hand-rolled and pure, so they are trivially unit-tested.

All operate on a *ranking*: an ordered list of retrieved ids (best first) and a
set of relevant ids. Keeping these as free functions over plain ids (not coupled
to the store) is what lets us test them against synthetic rankings with known
answers.
"""

from __future__ import annotations

from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items found within the top-k results.

    Undefined when there are no relevant items; we return 1.0 (nothing to miss).
    """
    if not relevant:
        return 1.0
    top = set(retrieved[:k])
    return len(top & relevant) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """1 / (rank of the first relevant hit), or 0.0 if none is retrieved."""
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(
    rankings: Sequence[tuple[Sequence[str], set[str]]],
) -> float:
    """Mean of reciprocal_rank over (retrieved, relevant) pairs."""
    if not rankings:
        return 0.0
    return sum(reciprocal_rank(r, rel) for r, rel in rankings) / len(rankings)


def mean_recall_at_k(rankings: Sequence[tuple[Sequence[str], set[str]]], k: int) -> float:
    if not rankings:
        return 0.0
    return sum(recall_at_k(r, rel, k) for r, rel in rankings) / len(rankings)


def citation_validity(cited: Sequence[int], n_chunks: int) -> float:
    """Fraction of cited [n] markers that fall within the retrieved chunk range.

    No citations at all counts as 1.0 (nothing invalid was emitted).
    """
    if not cited:
        return 1.0
    valid = sum(1 for n in cited if 1 <= n <= n_chunks)
    return valid / len(cited)
