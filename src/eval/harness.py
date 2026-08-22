#!/usr/bin/env python3
"""Offline evaluation harness: metrics, slices and bootstrap CIs (Q4).

Scores every candidate in each impression's inview list with each retriever,
then reports AUC / MRR / nDCG@5 / nDCG@10 plus intra-list diversity, novelty
and coverage, sliced and with 95% bootstrap confidence intervals.

Scorers evaluated together so they share one sample and one bootstrap draw:
    random      the floor
    popularity  train-split click counts - the real baseline to beat
    bm25        lexical similarity to the user's recent clicks
    semantic    embedding similarity to the user's pooled click vector
    hybrid      min-max normalised blend of the two content scorers

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
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config  # noqa: E402
from src.common.io import read_table  # noqa: E402
from src.eval import metrics as M  # noqa: E402
from src.retrieval.bm25 import BM25Index, build_queries  # noqa: E402
from src.retrieval.semantic import (  # noqa: E402
    build_user_vectors, encode_articles, l2_normalize, load_provided_embeddings,
)

TOP_K_LIST = 10   # list length for the beyond-accuracy metrics
SCORERS = ["random", "popularity", "bm25", "semantic", "hybrid"]


def minmax(values: np.ndarray) -> np.ndarray:
    """Scale to [0,1] within one impression so two scorers can be blended."""
    lo, hi = values.min(), values.max()
    return np.zeros_like(values) if hi - lo < 1e-12 else (values - lo) / (hi - lo)


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
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="hybrid weight: alpha*bm25 + (1-alpha)*semantic")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    articles = pd.read_parquet(cfg.features / "articles.parquet")
    articles["tokens"] = articles["tokens"].map(list)
    profiles = pd.read_parquet(cfg.features / "user_profiles.parquet")
    profiles = profiles[profiles["split"] == args.split].copy()
    profiles["clicked_ids"] = profiles["clicked_ids"].map(list)
    impressions = read_table(cfg.processed / args.split / "impressions.parquet",
                             "impressions")

    if args.sample and args.sample < len(impressions):
        impressions = impressions.sample(args.sample, random_state=13)
    impressions = impressions.reset_index(drop=True)
    print(f"[{cfg.dataset}/{args.split}] evaluating {len(impressions):,} impressions")

    article_ids = articles["article_id"].tolist()
    row_of = {a: i for i, a in enumerate(article_ids)}
    popularity = dict(zip(articles["article_id"], articles["train_clicks"]))
    total_clicks = int(articles["train_clicks"].sum())
    category_of = dict(zip(articles["article_id"], articles["category"]))
    is_head = dict(zip(articles["article_id"], articles["is_head"]))

    # --- per-user query representations, shared by bm25 and semantic ---
    needed = set(impressions["user_id"])
    profiles = profiles[profiles["user_id"].isin(needed)]
    print("  building BM25 index")
    index = BM25Index(article_ids, articles["tokens"].tolist())
    user_ids, token_lists = build_queries(profiles, articles, args.last_n)
    user_row = {u: i for i, u in enumerate(user_ids)}
    query_matrix = index.query_matrix(token_lists)

    print("  loading embeddings")
    if cfg.can("has_provided_embeddings"):
        raw_vectors = load_provided_embeddings(cfg, article_ids)
    else:
        raw_vectors = encode_articles(cfg, articles, batch_size=128)
    embeddings = l2_normalize(raw_vectors)
    _, user_vectors = build_user_vectors(
        profiles, row_of, embeddings, args.last_n, False, 5.0
    )

    # --- flatten candidates once; every scorer reuses the same layout ---
    lengths = impressions["inview_ids"].map(len).to_numpy()
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    flat_ids = [a for row in impressions["inview_ids"] for a in row]
    flat_doc_rows = np.array([row_of.get(a, -1) for a in flat_ids])
    flat_user_rows = np.repeat(
        [user_row.get(u, -1) for u in impressions["user_id"]], lengths
    )
    valid = (flat_doc_rows >= 0) & (flat_user_rows >= 0)

    print("  scoring candidates")
    flat_scores = {}
    flat_scores["random"] = np.random.default_rng(13).random(len(flat_ids)).astype(np.float32)
    flat_scores["popularity"] = np.array(
        [popularity.get(a, 0) for a in flat_ids], dtype=np.float32
    )

    bm = np.zeros(len(flat_ids), dtype=np.float32)
    bm[valid] = index.score_pairs(query_matrix, flat_user_rows[valid],
                                  flat_doc_rows[valid])
    flat_scores["bm25"] = bm

    sem = np.zeros(len(flat_ids), dtype=np.float32)
    sem[valid] = np.einsum(
        "ij,ij->i", user_vectors[flat_user_rows[valid]], embeddings[flat_doc_rows[valid]]
    )
    flat_scores["semantic"] = sem

    # --- regroup per impression, blending the hybrid inside each list ---
    per_imp: dict[str, list] = {s: [] for s in SCORERS}
    labels_by_imp, ids_by_imp = [], []
    clicked_sets = [set(c) for c in impressions["clicked_ids"]]

    for i in range(len(impressions)):
        lo, hi = offsets[i], offsets[i + 1]
        ids = flat_ids[lo:hi]
        ids_by_imp.append(ids)
        labels_by_imp.append(
            np.fromiter((1 if a in clicked_sets[i] else 0 for a in ids),
                        dtype=np.int8, count=len(ids))
        )
        b, s = flat_scores["bm25"][lo:hi], flat_scores["semantic"][lo:hi]
        for name in ("random", "popularity", "bm25", "semantic"):
            per_imp[name].append(flat_scores[name][lo:hi])
        # Normalise within the impression: raw BM25 and cosine are not comparable.
        per_imp["hybrid"].append(args.alpha * minmax(b) + (1 - args.alpha) * minmax(s))

    # --- slices ---
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
        "params": {"last_n": args.last_n, "alpha": args.alpha,
                   "n_boot": args.n_boot, "top_k_list": TOP_K_LIST},
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
