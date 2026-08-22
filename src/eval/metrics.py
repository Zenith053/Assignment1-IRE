"""Ranking and beyond-accuracy metrics, computed per impression.

Every accuracy metric takes one impression's labels and scores and returns a
single number, so the harness can hold a per-impression array and bootstrap it
directly. Both datasets contain multi-click impressions (27.9% of MIND), so no
metric may assume exactly one positive.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """ROC AUC within one impression, via the rank-sum identity.

    Returns None when every label is identical - AUC is undefined there, and
    silently scoring it 0.5 would bias the mean toward chance.
    """
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    # Average ranks handle score ties correctly, which matters for popularity
    # scores where many candidates share a click count.
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    sorted_scores = scores[order]
    start = 0
    for i in range(1, len(sorted_scores) + 1):
        if i == len(sorted_scores) or sorted_scores[i] != sorted_scores[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    rank_sum = ranks[labels == 1].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def mrr(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mean reciprocal rank over *all* relevant items, per the official scorer.

    Microsoft's evaluate.py sums 1/rank across every positive and divides by the
    number of positives, rather than taking only the first hit. The two agree on
    single-click impressions but diverge on the 27.9% of MIND that is
    multi-click; using first-hit-only overstated MRR by 0.045 (0.349 vs 0.304).
    """
    n_pos = float(labels.sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order]
    reciprocal = ranked / (np.arange(len(ranked)) + 1)
    return float(reciprocal.sum() / n_pos)


def ndcg(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    """nDCG@k with binary gains, accumulating every positive in the cut."""
    order = np.argsort(-scores, kind="mergesort")[:k]
    gains = labels[order]
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float((gains * discounts).sum())
    # Ideal ordering puts min(n_pos, k) positives at the top.
    n_ideal = min(int(labels.sum()), k)
    ideal = float((np.ones(n_ideal) / np.log2(np.arange(2, n_ideal + 2))).sum())
    return dcg / ideal if ideal > EPS else 0.0


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest-scoring candidates, best first."""
    return np.argsort(-scores, kind="mergesort")[:k]


def intra_list_diversity(vectors: np.ndarray) -> float | None:
    """1 - mean pairwise cosine similarity of the recommended items.

    Needs at least two items with embeddings; returns None otherwise so the
    harness can average over the impressions where it is defined.
    """
    if vectors is None or len(vectors) < 2:
        return None
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.maximum(norms, EPS)
    sims = unit @ unit.T
    n = len(vectors)
    # Mean of the strict upper triangle: each unordered pair counted once.
    off_diagonal = (sims.sum() - np.trace(sims)) / (n * (n - 1))
    return float(1.0 - off_diagonal)


def category_entropy(categories: list[str]) -> float:
    """Shannon entropy over the categories of a recommended list, in bits.

    Reported alongside embedding diversity because the two disagree: a list can
    be lexically varied while staying inside one section.
    """
    if not categories:
        return 0.0
    _, counts = np.unique(np.asarray(categories), return_counts=True)
    probs = counts / counts.sum()
    # Counts from np.unique are all positive, so log2 needs no epsilon guard -
    # adding one made a single-category list score -1.4e-12 instead of 0.
    return float(-(probs * np.log2(probs)).sum())


def novelty(article_ids, popularity: dict[str, int], total_clicks: int) -> float:
    """Mean self-information -log2 p(item) over a recommended list.

    Popularity comes from the train split, so a high score means "rarely
    clicked during training", not "rarely clicked in the evaluation window".
    """
    if total_clicks <= 0 or len(article_ids) == 0:
        return 0.0
    values = []
    for article_id in article_ids:
        # +1 smoothing keeps never-clicked articles finite rather than infinite.
        p = (popularity.get(article_id, 0) + 1) / (total_clicks + 1)
        values.append(-np.log2(p))
    return float(np.mean(values))


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 13) -> tuple[float, float, float]:
    """Percentile bootstrap 95% CI over per-impression metric values.

    Resampling impressions (not candidates) is what matches the unit of
    evaluation, and it is what makes the sub-sampled harness runs honest.
    """
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(values.mean())
    if len(values) == 1:
        return point, point, point
    rng = np.random.default_rng(seed)
    # One vectorised draw of all resamples; means along axis 1.
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)
