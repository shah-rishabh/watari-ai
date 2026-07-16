"""Metric functions tested against synthetic rankings with known answers.

Testing the *metric math itself* — not just that the harness runs — is the point
of a hand-rolled eval harness. These are the anchor tests for that claim.
"""

from __future__ import annotations

import math

from watari.evals.metrics.calibration import cohens_kappa
from watari.evals.metrics.retrieval import (
    citation_validity,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)


class TestRecallAtK:
    def test_all_relevant_in_top_k(self) -> None:
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0

    def test_partial_recall(self) -> None:
        assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 0.5

    def test_k_truncates(self) -> None:
        # Relevant item sits at rank 4, outside k=3.
        assert recall_at_k(["x", "y", "z", "a"], {"a"}, k=3) == 0.0

    def test_no_relevant_is_one(self) -> None:
        assert recall_at_k(["a", "b"], set(), k=3) == 1.0


class TestReciprocalRank:
    def test_first_hit_at_rank_1(self) -> None:
        assert reciprocal_rank(["a", "b"], {"a"}) == 1.0

    def test_first_hit_at_rank_3(self) -> None:
        assert reciprocal_rank(["x", "y", "a"], {"a"}) == 1 / 3

    def test_no_hit_is_zero(self) -> None:
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_mrr_averages(self) -> None:
        rankings = [
            (["a"], {"a"}),  # rr 1.0
            (["x", "a"], {"a"}),  # rr 0.5
        ]
        assert mean_reciprocal_rank(rankings) == 0.75


class TestCitationValidity:
    def test_all_valid(self) -> None:
        assert citation_validity([1, 2, 3], n_chunks=3) == 1.0

    def test_one_hallucinated(self) -> None:
        assert citation_validity([1, 5], n_chunks=3) == 0.5

    def test_no_citations_is_one(self) -> None:
        assert citation_validity([], n_chunks=3) == 1.0


class TestCohensKappa:
    def test_perfect_agreement(self) -> None:
        a = [True, False, True, False]
        assert cohens_kappa(a, a) == 1.0

    def test_total_disagreement_is_negative(self) -> None:
        a = [True, True, False, False]
        b = [False, False, True, True]
        assert cohens_kappa(a, b) < 0

    def test_chance_level_near_zero(self) -> None:
        # Rater A alternates, rater B is independent-ish; agreement ~ chance.
        a = [True, False, True, False, True, False]
        b = [True, False, False, True, True, False]
        k = cohens_kappa(a, b)
        assert -0.5 < k < 0.6

    def test_both_constant_and_equal(self) -> None:
        assert cohens_kappa([True, True], [True, True]) == 1.0

    def test_length_mismatch_raises(self) -> None:
        try:
            cohens_kappa([True], [True, False])
        except ValueError:
            return
        raise AssertionError("expected ValueError")


def test_kappa_matches_hand_computation() -> None:
    # 8 items: observed agreement 6/8 = 0.75.
    # a has 4 True, b has 4 True -> p_e = .5*.5 + .5*.5 = 0.5.
    # kappa = (0.75 - 0.5) / (1 - 0.5) = 0.5.
    a = [True, True, True, True, False, False, False, False]
    b = [True, True, True, False, True, False, False, False]
    assert math.isclose(cohens_kappa(a, b), 0.5, abs_tol=1e-9)
