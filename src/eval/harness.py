#!/usr/bin/env python3
"""Offline evaluation harness: metrics, slices and bootstrap CIs (Q4).

Scores every candidate in each impression's inview list with each retriever,
then reports AUC / MRR / nDCG@5 / nDCG@10 plus intra-list diversity, novelty
and coverage, sliced and with 95% bootstrap confidence intervals.

Scorers evaluated together so they share one sample and one bootstrap draw:
    random      the floor
    popularity  train-split click counts - the real baseline to beat
    bm25        lexical similarity to the user's recent clicks
    semantic    embedding similarity to the user's click history, mean-pooled
                or top-k pooled per --pooling
    hybrid      a logistic regression over (bm25, semantic), fit on the val
                split and applied frozen to whatever split is reported - a
                fixed alpha*bm25 + (1-alpha)*semantic blend measurably lost
                to semantic alone on MIND (0.6209 vs 0.6301 AUC at alpha=0.5)

Usage
-----
    python src/eval/harness.py --config config/mind.yaml --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config  # noqa: E402
from src.common.io import read_table  # noqa: E402
from src.eval import metrics as M  # noqa: E402
from src.retrieval.bm25 import BM25Index, build_queries  # noqa: E402
from src.retrieval.semantic import (  # noqa: E402
    build_user_history_rows, build_user_vectors, encode_articles, l2_normalize,
    load_provided_embeddings, score_topk_similarity,
)

TOP_K_LIST = 10   # list length for the beyond-accuracy metrics
SCORERS = ["random", "popularity", "bm25", "semantic", "hybrid"]
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
                impressions: pd.DataFrame, last_n: int, pooling: str, topk: int) -> dict:
    """Score every candidate in `impressions` with each base scorer.

    Shared by the val-split fit pass and whatever split is being reported, so
    both use identical scoring code - the only thing that differs is which
    impressions are passed in.
    """
    needed = set(impressions["user_id"])
    profiles = profiles_all[profiles_all["user_id"].isin(needed)].reset_index(drop=True)

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
    rediscover that raw BM25 and cosine live on different scales. Fit on the
    val split only and applied frozen to whichever split is reported, the
    same discipline as popularity being fit on train only: the combiner must
    not see the impressions it is graded on.
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


def evaluate(labels_by_imp, scores_by_imp, article_ids_by_imp, article_meta,
             popularity, total_clicks, embeddings, row_of):
    """Per-impression metric values for one scorer."""
    out = {"auc": [], "mrr": [], "ndcg@5": [], "ndcg@10": [],
           "diversity": [], "cat_entropy": [], "novelty": []}
    recommended: set[str] = set()

    for labels, scores, ids in zip(labels_by_imp, scores_by_imp, article_ids_by_imp):
        out["auc"].append(M.auc(labels, scores))
        out["mrr"].append(M.mrr(labels, scores))
        out["ndcg@5"].append(M.ndcg(labels, scores, 5))
        out["ndcg@10"].append(M.ndcg(labels, scores, 10))

        top = M.top_k_indices(scores, TOP_K_LIST)
        top_ids = [ids[i] for i in top]
        recommended.update(top_ids)

        rows = [row_of[a] for a in top_ids if a in row_of]
        vectors = embeddings[rows] if embeddings is not None and rows else None
        out["diversity"].append(M.intra_list_diversity(vectors))
        out["cat_entropy"].append(
            M.category_entropy([article_meta.get(a, "") for a in top_ids])
        )
        out["novelty"].append(M.novelty(top_ids, popularity, total_clicks))

    return out, recommended


def summarise(values, n_boot: int) -> dict:
    """Point estimate with a bootstrap 95% CI, skipping undefined entries."""
    point, lo, hi = M.bootstrap_ci(values, n_boot=n_boot)
    defined = sum(1 for v in values if v is not None)
    return {"value": point, "ci_low": lo, "ci_high": hi,
            "n": defined, "n_undefined": len(values) - defined}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--sample", type=int, default=20000)
    parser.add_argument("--last-n", type=int, default=20)
    parser.add_argument("--pooling", default="topk", choices=["mean", "topk"],
                        help="semantic user representation: mean-pool history into one "
                             "vector, or score each candidate by its k highest similarities "
                             "to individual history clicks. Measured on MIND val: topk beats "
                             "mean pooling, AUC 0.6414 vs 0.6299.")
    parser.add_argument("--topk", type=int, default=5,
                        help="k for --pooling topk; 5 was the peak of a 1..20 sweep on MIND val")
    parser.add_argument("--fit-sample", type=int, default=5000,
                        help="val impressions used to fit the hybrid combiner")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    articles = pd.read_parquet(cfg.features / "articles.parquet")
    articles["tokens"] = articles["tokens"].map(list)
    profiles_all = pd.read_parquet(cfg.features / "user_profiles.parquet")
    profiles_all["clicked_ids"] = profiles_all["clicked_ids"].map(list)

    article_ids = articles["article_id"].tolist()
    row_of = {a: i for i, a in enumerate(article_ids)}
    popularity = dict(zip(articles["article_id"], articles["train_clicks"]))
    total_clicks = int(articles["train_clicks"].sum())
    category_of = dict(zip(articles["article_id"], articles["category"]))
    is_head = dict(zip(articles["article_id"], articles["is_head"]))

    print(f"[{cfg.dataset}] building BM25 index")
    index = BM25Index(article_ids, articles["tokens"].tolist())

    print(f"[{cfg.dataset}] loading embeddings")
    if cfg.can("has_provided_embeddings"):
        raw_vectors = load_provided_embeddings(cfg, article_ids)
    else:
        raw_vectors = encode_articles(cfg, articles, batch_size=128)
    embeddings = l2_normalize(raw_vectors)

    # --- fit the hybrid combiner on val, applied frozen to whatever split is reported ---
    fit_impressions = read_table(cfg.processed / "val" / "impressions.parquet", "impressions")
    if args.fit_sample and args.fit_sample < len(fit_impressions):
        fit_impressions = fit_impressions.sample(args.fit_sample, random_state=13)
    fit_impressions = fit_impressions.reset_index(drop=True)
    print(f"  fitting hybrid combiner on {len(fit_impressions):,} val impressions "
          f"(pooling={args.pooling})")
    fit_scores = score_split(index, embeddings, articles, row_of, popularity, profiles_all,
                             fit_impressions, args.last_n, args.pooling, args.topk)
    combiner = fit_hybrid_combiner(fit_scores["per_imp"], fit_scores["labels_by_imp"])
    coef_bm25, coef_semantic = combiner.coef_[0]
    intercept = float(combiner.intercept_[0])
    print(f"  hybrid = sigmoid({coef_bm25:.3f}*bm25 + {coef_semantic:.3f}*semantic "
          f"+ {intercept:.3f})")

    # --- score the reported split ---
    if args.split == "val":
        impressions = fit_impressions
        report = fit_scores
        print(f"[{cfg.dataset}/val] reporting on the {len(impressions):,} impressions the "
              f"combiner was fit on - hybrid's number here is optimistic; use --split test "
              f"for the honest figure")
    else:
        impressions = read_table(cfg.processed / args.split / "impressions.parquet",
                                 "impressions")
        if args.sample and args.sample < len(impressions):
            impressions = impressions.sample(args.sample, random_state=13)
        impressions = impressions.reset_index(drop=True)
        print(f"[{cfg.dataset}/{args.split}] evaluating {len(impressions):,} impressions")
        report = score_split(index, embeddings, articles, row_of, popularity, profiles_all,
                             impressions, args.last_n, args.pooling, args.topk)

    ids_by_imp = report["ids_by_imp"]
    labels_by_imp = report["labels_by_imp"]
    per_imp: dict[str, list] = dict(report["per_imp"])
    per_imp["hybrid"] = hybrid_scores(combiner, report["per_imp"])

    # --- slices ---
    needed = set(impressions["user_id"])
    profiles = profiles_all[
        (profiles_all["split"] == args.split) & (profiles_all["user_id"].isin(needed))
    ]
    low_history = set(profiles.loc[profiles["is_low_history"], "user_id"])
    slices = {
        "all": np.ones(len(impressions), dtype=bool),
        "cold_users": impressions["user_id"].isin(low_history).to_numpy(),
        "warm_users": ~impressions["user_id"].isin(low_history).to_numpy(),
        # An impression is "head" when any clicked article is a popular one.
        "head_clicks": np.array([
            any(is_head.get(a, False) for a in c) for c in impressions["clicked_ids"]
        ]),
    }
    slices["tail_clicks"] = ~slices["head_clicks"]

    results: dict[str, dict] = {}
    for scorer in SCORERS:
        raw, recommended = evaluate(
            labels_by_imp, per_imp[scorer], ids_by_imp, category_of,
            popularity, total_clicks, embeddings, row_of
        )
        entry = {"slices": {}}
        entry["coverage"] = len(recommended) / max(1, len(article_ids))
        entry["n_distinct_recommended"] = len(recommended)
        for slice_name, mask in slices.items():
            if not mask.any():
                entry["slices"][slice_name] = {"available": False, "n": 0}
                continue
            idx = np.flatnonzero(mask)
            entry["slices"][slice_name] = {
                "available": True,
                "n_impressions": int(len(idx)),
                **{metric: summarise([raw[metric][i] for i in idx], args.n_boot)
                   for metric in raw},
            }
        results[scorer] = entry

    payload = {
        "dataset": cfg.dataset, "split": args.split, "scale": cfg.scale,
        "n_impressions": int(len(impressions)),
        "params": {
            "last_n": args.last_n, "pooling": args.pooling, "topk": args.topk,
            "n_boot": args.n_boot, "top_k_list": TOP_K_LIST,
            "hybrid": {
                "method": "logistic_regression", "fit_split": "val",
                "n_fit_impressions": int(len(fit_impressions)),
                "coef_bm25": float(coef_bm25), "coef_semantic": float(coef_semantic),
                "intercept": intercept,
            },
        },
        "scorers": results,
    }
    out = args.out or REPO_ROOT / "reports" / f"eval_{cfg.dataset}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # --- console table ---
    print(f"\n  {'scorer':<11} {'AUC':>18} {'MRR':>18} {'nDCG@5':>18} "
          f"{'nDCG@10':>18} {'cov':>6}")
    for scorer in SCORERS:
        row = results[scorer]["slices"]["all"]
        cells = []
        for metric in ("auc", "mrr", "ndcg@5", "ndcg@10"):
            m = row[metric]
            cells.append(f"{m['value']:.4f}[{m['ci_low']:.3f},{m['ci_high']:.3f}]")
        print(f"  {scorer:<11} " + " ".join(f"{c:>18}" for c in cells)
              + f" {results[scorer]['coverage']:>6.3f}")

    print(f"\n  slices (nDCG@10):")
    for slice_name in slices:
        cells = []
        for scorer in SCORERS:
            s = results[scorer]["slices"][slice_name]
            cells.append(f"{scorer}={s['ndcg@10']['value']:.4f}" if s.get("available")
                         else f"{scorer}=n/a")
        n = results[SCORERS[0]]["slices"][slice_name].get("n_impressions", 0)
        print(f"    {slice_name:<12} n={n:>6,}  " + "  ".join(cells))
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
