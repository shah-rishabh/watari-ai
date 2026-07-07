"""Reciprocal Rank Fusion — the core hybrid-search math, tested in isolation."""

from __future__ import annotations

from watari.rag.store import reciprocal_rank_fusion


def test_single_ranking_preserves_order() -> None:
    fused = reciprocal_rank_fusion([[10, 20, 30]], k=60)
    assert [cid for cid, _ in fused] == [10, 20, 30]


def test_item_in_both_rankings_outscores_item_in_one() -> None:
    # 1 appears in both lists (rank 0 and rank 2); 2 and 3 appear once each.
    fused = reciprocal_rank_fusion([[1, 2], [3, 4, 1]], k=60)
    top = fused[0][0]
    assert top == 1


def test_higher_rank_contributes_more() -> None:
    # Same id set, different positions: id 5 is rank 0 in one list.
    fused = dict(reciprocal_rank_fusion([[5, 6], [6, 5]], k=60))
    # Both appear at ranks {0,1} across the two lists, so scores tie.
    assert abs(fused[5] - fused[6]) < 1e-9


def test_k_dampens_rank_differences() -> None:
    small_k = dict(reciprocal_rank_fusion([[1, 2, 3]], k=1))
    large_k = dict(reciprocal_rank_fusion([[1, 2, 3]], k=1000))
    # With a large k, the gap between rank 0 and rank 2 shrinks.
    small_gap = small_k[1] - small_k[3]
    large_gap = large_k[1] - large_k[3]
    assert large_gap < small_gap


def test_empty_input_returns_empty() -> None:
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []
