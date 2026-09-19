"""Q2 Option A: LightGBM LambdaRank over the Q1 feature table.

LambdaRank rather than a classifier is the point. A1's design note already
names the hybrid combiner's weakness: `LogisticRegression` minimises log-loss,
which rewards calibrated click probabilities, while every reported metric
(AUC, MRR, nDCG) only cares about the ordering *within* one impression.
LambdaRank optimises nDCG directly, and `group` is what tells it where one
impression ends and the next begins.
"""

from __future__ import annotations

import numpy as np

from src.rerank.features import CATEGORICAL_FEATURES, FEATURE_NAMES


def train(X_train: np.ndarray, y_train: np.ndarray, group_train: np.ndarray,
         X_val: np.ndarray, y_val: np.ndarray, group_val: np.ndarray,
         feature_names: list[str] | None = None,
         num_boost_round: int = 1000, early_stopping_rounds: int = 50, seed: int = 13):
    """Fit on train, early-stop on val nDCG@10 - the same split roles A1 uses
    everywhere else (popularity fit on train, hybrid fit on val, never test).

    `feature_names` defaults to the full Q1 table but must match `X_train`'s
    actual column count - callers that pass a column subset (the single-
    feature sanity check in evaluate_reranker.py) must pass the matching
    subset of names too, or LightGBM raises a shape mismatch rather than
    silently mislabelling columns.
    """
    import lightgbm as lgb

    feature_names = feature_names if feature_names is not None else FEATURE_NAMES
    if len(feature_names) != X_train.shape[1]:
        raise ValueError(
            f"feature_names has {len(feature_names)} entries but X_train has "
            f"{X_train.shape[1]} columns"
        )
    cat_idx = [feature_names.index(c) for c in CATEGORICAL_FEATURES if c in feature_names]
    train_set = lgb.Dataset(X_train, label=y_train, group=group_train,
                            feature_name=feature_names, categorical_feature=cat_idx,
                            free_raw_data=True)
    val_set = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_set,
                          free_raw_data=True)

    params = {
        "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [5, 10],
        "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 30,
        # feature_fraction < 1 and an L2 term both push against a tree leaning
        # on whichever single feature happens to offer the most split points
        # at this sample size - see the note in features.py on why raw
        # popularity/position scores were replaced with in-impression ranks.
        "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l2": 1.0, "lambdarank_truncation_level": 30,
        # Measured, not assumed: a single-feature sanity-check model (semantic
        # only) cannot quite reproduce semantic's own raw AUC (0.6308 vs
        # 0.6455 on a MIND sample) even at the default 255 histogram bins, so
        # some of that gap is unavoidable binning/discretisation loss on an
        # already near-continuous, near-optimal signal. Raising max_bin to
        # 1023 was tried as the direct fix and made both the single-feature
        # and full-feature models *worse* (full-feature AUC 0.6285 -> 0.5966)
        # - finer bins gave the noisier behavioural features more precise
        # ways to overfit, which outweighed any resolution gained on
        # `semantic`. Left at the library default on that evidence.
        # One thread: LightGBM links whichever libomp is already loaded, and faiss and
        # torch (imported by the feature context and the MLP) each bundle an incompatible
        # copy - with more than one thread, training segfaults (reproduced: 4 threads
        # exit -11, 1 thread OK). Fixed seed + one thread also makes runs reproducible.
        "num_threads": 1, "verbosity": -1, "seed": seed,
    }
    booster = lgb.train(
        params, train_set, num_boost_round=num_boost_round, valid_sets=[val_set],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False),
                  lgb.log_evaluation(0)],
    )
    return booster


def predict_per_impression(booster, X: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    """Score every candidate, then split back into the per-impression shape
    `eval.metrics` and both submission scripts already consume."""
    flat = booster.predict(X, num_iteration=booster.best_iteration, num_threads=1)
    return [flat[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


def feature_importance(booster) -> list[tuple[str, float]]:
    """Gain-based importance, normalised to sum to 1 - which features the
    model actually leaned on, for the design note's feature-importance table."""
    gains = booster.feature_importance(importance_type="gain")
    total = gains.sum() or 1.0
    ranked = sorted(zip(FEATURE_NAMES, gains / total), key=lambda kv: -kv[1])
    return ranked
