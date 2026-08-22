"""Unit tests for the ranking and beyond-accuracy metrics.

These pin the behaviour that the harness depends on, in particular multi-click
impressions (27.9% of MIND) and score ties (every popularity-based scorer
produces them in bulk).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eval import metrics as M


def test_auc_perfect_and_inverted():
    labels = np.array([0, 1, 0, 1])
    assert M.auc(labels, np.array([0.1, 0.9, 0.2, 0.8])) == pytest.approx(1.0)
    assert M.auc(labels, np.array([0.9, 0.1, 0.8, 0.2])) == pytest.approx(0.0)


def test_auc_is_none_when_undefined():
    """All-positive or all-negative impressions have no defined AUC."""
    assert M.auc(np.array([0, 0, 0]), np.array([1.0, 2.0, 3.0])) is None
    assert M.auc(np.array([1, 1]), np.array([1.0, 2.0])) is None


def test_auc_ties_give_half():
    """Tied scores must average to 0.5, not depend on array order."""
    assert M.auc(np.array([0, 1]), np.array([5.0, 5.0])) == pytest.approx(0.5)
    assert M.auc(np.array([1, 0]), np.array([5.0, 5.0])) == pytest.approx(0.5)


def test_mrr_uses_first_relevant():
    labels = np.array([0, 1, 1])
    scores = np.array([0.9, 0.5, 0.4])  # first positive lands at rank 2
    assert M.mrr(labels, scores) == pytest.approx(0.5)


def test_mrr_zero_when_no_positive():
    assert M.mrr(np.array([0, 0]), np.array([0.1, 0.2])) == 0.0


def test_ndcg_handles_multiple_positives():
    """With two positives in the top two slots, nDCG@2 must be exactly 1."""
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    assert M.ndcg(labels, scores, 2) == pytest.approx(1.0)
    # Same labels, positives pushed below the cut -> zero.
    assert M.ndcg(labels, np.array([0.1, 0.2, 0.9, 0.8]), 2) == pytest.approx(0.0)


def test_ndcg_partial_credit_is_between_zero_and_one():
    labels = np.array([1, 0, 1, 0])
    value = M.ndcg(labels, np.array([0.9, 0.8, 0.7, 0.6]), 10)
    assert 0.0 < value < 1.0


def test_diversity_bounds():
    identical = np.array([[1.0, 0.0], [1.0, 0.0]])
    orthogonal = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert M.intra_list_diversity(identical) == pytest.approx(0.0)
    assert M.intra_list_diversity(orthogonal) == pytest.approx(1.0)
    assert M.intra_list_diversity(np.array([[1.0, 0.0]])) is None  # needs 2+


def test_category_entropy():
    assert M.category_entropy(["a", "b", "c", "d"]) == pytest.approx(2.0)
    assert M.category_entropy(["a", "a", "a"]) == pytest.approx(0.0)


def test_novelty_rewards_rare_items():
    popularity = {"rare": 1, "common": 1000}
    total = 1001
    assert M.novelty(["rare"], popularity, total) > M.novelty(["common"], popularity, total)


def test_bootstrap_ci_brackets_the_mean():
    values = np.random.default_rng(0).normal(0.5, 0.1, 2000)
    point, lo, hi = M.bootstrap_ci(values, n_boot=500)
    assert lo < point < hi
    assert lo < 0.5 < hi


def test_bootstrap_ci_ignores_undefined_entries():
    """None values (undefined AUC) must be dropped, not treated as zero."""
    point, _, _ = M.bootstrap_ci([1.0, None, 1.0], n_boot=100)
    assert point == pytest.approx(1.0)
