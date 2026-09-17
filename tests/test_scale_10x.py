"""Q4 phase 5: the synthetic 10x construction and recall arithmetic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.serving.scale_10x import recall_at_k, replicate_embeddings, replicate_index  # noqa: E402


def _unit(n, d, seed=0):
    x = np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_replicated_embeddings_shape_norm_and_original_copy():
    emb = _unit(50, 16)
    big = replicate_embeddings(emb, factor=10, noise_std=0.01)
    assert big.shape == (500, 16) and big.dtype == np.float32
    assert np.array_equal(big[:50], emb), "copy 0 must be the original"
    assert np.allclose(np.linalg.norm(big, axis=1), 1.0, atol=1e-5)


def test_copies_are_close_to_but_not_identical_with_the_original():
    emb = _unit(50, 64)
    big = replicate_embeddings(emb, factor=3, noise_std=0.01)
    sims = np.sum(big[50:100] * emb, axis=1)
    assert (sims > 0.9).all() and (sims < 0.99999).all()


def test_replicate_index_points_at_every_copy():
    idx = replicate_index(np.array([2, 5]), n=10, factor=3)
    assert idx.tolist() == [2, 5, 12, 15, 22, 25]


def test_recall_at_k():
    exact = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    approx = np.array([[1, 2, 9, 9], [5, 6, 7, 8]])
    assert recall_at_k(approx, exact) == pytest.approx((0.5 + 1.0) / 2)
