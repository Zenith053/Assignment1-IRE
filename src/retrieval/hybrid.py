"""Learned combination of BM25 and semantic scores, replacing a fixed alpha blend.

Shared by the eval harness and the submission scripts, so both use one
definition of "hybrid" score. The combiner is fit on the val split's labeled
impressions and applied frozen wherever a hybrid score is needed - the same
discipline as popularity being fit on train only: it must never see the
impressions it is later graded or submitted on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.retrieval.bm25 import BM25Index, build_queries
from src.retrieval.semantic import build_user_history_rows, build_user_vectors, score_topk_similarity

BASE_SCORERS = ["random", "popularity", "bm25", "semantic"]


def minmax(values: np.ndarray) -> np.ndarray:
    """Scale to [0,1] within one impression so two scorers can be blended."""
    lo, hi = values.min(), values.max()
    return np.zeros_like(values) if hi - lo < 1e-12 else (values - lo) / (hi - lo)


def flatten_candidates(impressions: pd.DataFrame, row_of: dict[str, int]):
    """One flat array of every impression's candidates, plus offsets to regroup."""
    lengths = impressions["inview_ids"].map(len).to_numpy()
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    flat_ids = [a for row in impressions["inview_ids"] for a in row]
    flat_doc_rows = np.array([row_of.get(a, -1) for a in flat_ids])
    return offsets, flat_ids, flat_doc_rows


def score_split(index: BM25Index, embeddings: np.ndarray, articles: pd.DataFrame,
                row_of: dict[str, int], popularity: dict, profiles_all: pd.DataFrame,
                impressions: pd.DataFrame, split_name: str, last_n: int, pooling: str,
                topk: int) -> dict:
    """Score every candidate in `impressions` with each base scorer.

    `profiles_all` holds one row per (user, split); a user can appear in more
    than one split with genuinely different click histories (EB-NeRD staggers
    a fresh history snapshot per split), so `split_name` must pin down the
    right row per user rather than leaving it to whichever duplicate happens
    to iterate last.
    """
    needed = set(impressions["user_id"])
    profiles = profiles_all[
        (profiles_all["split"] == split_name) & (profiles_all["user_id"].isin(needed))
    ].reset_index(drop=True)

    user_ids, token_lists = build_queries(profiles, articles, last_n)
    user_row = {u: i for i, u in enumerate(user_ids)}
    query_matrix = index.query_matrix(token_lists)

    offsets, flat_ids, flat_doc_rows = flatten_candidates(impressions, row_of)
    lengths = impressions["inview_ids"].map(len).to_numpy()
    flat_user_rows = np.repeat([user_row.get(u, -1) for u in impressions["user_id"]], lengths)
    valid = (flat_doc_rows >= 0) & (flat_user_rows >= 0)

    flat_scores = {}
    flat_scores["random"] = np.random.default_rng(13).random(len(flat_ids)).astype(np.float32)
    flat_scores["popularity"] = np.array(
        [popularity.get(a, 0) for a in flat_ids], dtype=np.float32
    )

    bm = np.zeros(len(flat_ids), dtype=np.float32)
    bm[valid] = index.score_pairs(query_matrix, flat_user_rows[valid], flat_doc_rows[valid])
    flat_scores["bm25"] = bm

    sem = np.zeros(len(flat_ids), dtype=np.float32)
    if pooling == "topk":
        hist_user_ids, hist_rows_list = build_user_history_rows(profiles, row_of, last_n)
        history_rows_by_user = dict(zip(hist_user_ids, hist_rows_list))
        for i in range(len(impressions)):
            lo, hi = offsets[i], offsets[i + 1]
            hist_rows = history_rows_by_user.get(impressions["user_id"].iat[i])
            if hist_rows is None or len(hist_rows) == 0:
                continue
            doc_rows = flat_doc_rows[lo:hi]
            ok = doc_rows >= 0
            if not ok.any():
                continue
            seg = sem[lo:hi]
            seg[ok] = score_topk_similarity(embeddings, doc_rows[ok], hist_rows, topk)
            sem[lo:hi] = seg
    else:
        _, user_vectors = build_user_vectors(profiles, row_of, embeddings, last_n, False, 5.0)
        sem[valid] = np.einsum(
            "ij,ij->i", user_vectors[flat_user_rows[valid]], embeddings[flat_doc_rows[valid]]
        )
    flat_scores["semantic"] = sem

    ids_by_imp, labels_by_imp = [], []
    per_imp: dict[str, list] = {name: [] for name in BASE_SCORERS}
    clicked_sets = [set(c) for c in impressions["clicked_ids"]]
    for i in range(len(impressions)):
        lo, hi = offsets[i], offsets[i + 1]
        ids = flat_ids[lo:hi]
        ids_by_imp.append(ids)
        labels_by_imp.append(
            np.fromiter((1 if a in clicked_sets[i] else 0 for a in ids),
                        dtype=np.int8, count=len(ids))
        )
        for name in BASE_SCORERS:
            per_imp[name].append(flat_scores[name][lo:hi])

    return {"ids_by_imp": ids_by_imp, "labels_by_imp": labels_by_imp, "per_imp": per_imp}


def fit_hybrid_combiner(per_imp: dict, labels_by_imp: list) -> LogisticRegression:
    """Learn how to combine (bm25, semantic) instead of a fixed alpha blend.

    Both scores are min-max normalised per impression first, exactly as the
    fixed blend did, so the combiner only has to learn the *weighting* - not
    rediscover that raw BM25 and cosine live on different scales.
    """
    X = np.concatenate([
        np.column_stack([minmax(b), minmax(s)])
        for b, s in zip(per_imp["bm25"], per_imp["semantic"])
    ])
    y = np.concatenate(labels_by_imp)
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X, y)
    return model


def hybrid_scores(model: LogisticRegression, per_imp: dict) -> list[np.ndarray]:
    """Apply a fitted combiner to (bm25, semantic) scores, per impression."""
    return [
        model.predict_proba(np.column_stack([minmax(b), minmax(s)]))[:, 1].astype(np.float32)
        for b, s in zip(per_imp["bm25"], per_imp["semantic"])
    ]


def fit_hybrid_from_val(cfg, index: BM25Index, embeddings: np.ndarray, articles: pd.DataFrame,
                        row_of: dict[str, int], popularity: dict, profiles_all: pd.DataFrame,
                        last_n: int = 20, pooling: str = "topk", topk: int = 5,
                        fit_sample: int = 5000):
    """Load the val split, score it, and fit the hybrid combiner on it.

    The one entry point every caller (harness, submission scripts) should use
    to get a fitted combiner, so "how do we fit hybrid" has a single
    definition. Returns (model, n_fit_impressions).
    """
    from src.common.io import read_table

    fit_impressions = read_table(cfg.processed / "val" / "impressions.parquet", "impressions")
    if fit_sample and fit_sample < len(fit_impressions):
        fit_impressions = fit_impressions.sample(fit_sample, random_state=13)
    fit_impressions = fit_impressions.reset_index(drop=True)

    fit_scored = score_split(index, embeddings, articles, row_of, popularity, profiles_all,
                             fit_impressions, "val", last_n, pooling, topk)
    model = fit_hybrid_combiner(fit_scored["per_imp"], fit_scored["labels_by_imp"])
    return model, fit_scored, len(fit_impressions)
