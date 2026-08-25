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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config  # noqa: E402
from src.common.io import read_table  # noqa: E402
from src.eval import metrics as M  # noqa: E402
from src.retrieval.bm25 import BM25Index  # noqa: E402
from src.retrieval.hybrid import fit_hybrid_from_val, hybrid_scores, score_split  # noqa: E402
from src.retrieval.semantic import encode_articles, l2_normalize, load_provided_embeddings  # noqa: E402

TOP_K_LIST = 10   # list length for the beyond-accuracy metrics
SCORERS = ["random", "popularity", "bm25", "semantic", "hybrid"]


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


def load_leaky_popularity(cfg, article_ids: list[str]) -> dict[str, float] | None:
    """Article-level popularity that is NOT available at serving time (Q9).

    EB-NeRD's raw `total_pageviews` accumulates over an article's entire
    lifetime - including impressions from the future relative to any given
    split - so ranking with it directly is a serving-time violation. Built
    only to measure and report the inflation it would cause, and only for
    datasets that declare unavailable columns via `serving_time_unavailable`
    (config/ebnerd.yaml); absent that declaration this returns None so the
    harness reports the ablation as unavailable rather than fabricating one.
    """
    if not cfg.serving_time_unavailable or "articles" not in cfg.raw:
        return None
    raw = pd.read_parquet(cfg.raw["articles"], columns=["article_id", "total_pageviews"])
    raw["article_id"] = raw["article_id"].astype(str)
    lookup = dict(zip(raw["article_id"], raw["total_pageviews"].fillna(0)))
    return {a: float(lookup.get(a, 0.0)) for a in article_ids}


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
    print(f"  fitting hybrid combiner on val (pooling={args.pooling})")
    combiner, fit_scores, n_fit = fit_hybrid_from_val(
        cfg, index, embeddings, articles, row_of, popularity, profiles_all,
        args.last_n, args.pooling, args.topk, args.fit_sample
    )
    coef_bm25, coef_semantic = combiner.coef_[0]
    intercept = float(combiner.intercept_[0])
    print(f"  hybrid = sigmoid({coef_bm25:.3f}*bm25 + {coef_semantic:.3f}*semantic "
          f"+ {intercept:.3f}), fit on {n_fit:,} val impressions")

    # --- score the reported split ---
    if args.split == "val":
        impressions = read_table(cfg.processed / "val" / "impressions.parquet", "impressions")
        if args.fit_sample and args.fit_sample < len(impressions):
            impressions = impressions.sample(args.fit_sample, random_state=13)
        impressions = impressions.reset_index(drop=True)
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
                             impressions, args.split, args.last_n, args.pooling, args.topk)

    ids_by_imp = report["ids_by_imp"]
    labels_by_imp = report["labels_by_imp"]
    per_imp: dict[str, list] = dict(report["per_imp"])
    per_imp["hybrid"] = hybrid_scores(combiner, report["per_imp"])

    # --- Q9: serving-time ablation, only where the dataset declares unavailable columns ---
    leaky_popularity = load_leaky_popularity(cfg, article_ids)
    scorers_this_run = list(SCORERS)
    if leaky_popularity is not None:
        per_imp["popularity_leaky"] = [
            np.fromiter((leaky_popularity.get(a, 0.0) for a in ids),
                        dtype=np.float32, count=len(ids))
            for ids in ids_by_imp
        ]
        scorers_this_run.append("popularity_leaky")
        print(f"  Q9 ablation: added popularity_leaky (raw total_pageviews) "
              f"alongside honest train-only popularity")

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
    for scorer in scorers_this_run:
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

    q9_ablation = {"available": False, "reason": "no serving_time_unavailable columns declared"}
    if leaky_popularity is not None:
        honest_auc = results["popularity"]["slices"]["all"]["auc"]["value"]
        leaky_auc = results["popularity_leaky"]["slices"]["all"]["auc"]["value"]
        q9_ablation = {
            "available": True,
            "feature": "total_pageviews (lifetime-aggregated, future relative to any split)",
            "honest_popularity_auc": honest_auc,
            "leaky_popularity_auc": leaky_auc,
            "inflation": leaky_auc - honest_auc,
        }
        print(f"\n  Q9 serving-time ablation: popularity AUC {honest_auc:.4f} (honest, "
              f"train clicks only) vs {leaky_auc:.4f} (leaky, total_pageviews) "
              f"-> inflation {leaky_auc - honest_auc:+.4f}")

    payload = {
        "dataset": cfg.dataset, "split": args.split, "scale": cfg.scale,
        "n_impressions": int(len(impressions)),
        "params": {
            "last_n": args.last_n, "pooling": args.pooling, "topk": args.topk,
            "n_boot": args.n_boot, "top_k_list": TOP_K_LIST,
            "hybrid": {
                "method": "logistic_regression", "fit_split": "val",
                "n_fit_impressions": int(n_fit),
                "coef_bm25": float(coef_bm25), "coef_semantic": float(coef_semantic),
                "intercept": intercept,
            },
        },
        "q9_serving_time_ablation": q9_ablation,
        "scorers": results,
    }
    out = args.out or REPO_ROOT / "reports" / f"eval_{cfg.dataset}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # --- console table ---
    print(f"\n  {'scorer':<15} {'AUC':>18} {'MRR':>18} {'nDCG@5':>18} "
          f"{'nDCG@10':>18} {'cov':>6}")
    for scorer in scorers_this_run:
        row = results[scorer]["slices"]["all"]
        cells = []
        for metric in ("auc", "mrr", "ndcg@5", "ndcg@10"):
            m = row[metric]
            cells.append(f"{m['value']:.4f}[{m['ci_low']:.3f},{m['ci_high']:.3f}]")
        print(f"  {scorer:<15} " + " ".join(f"{c:>18}" for c in cells)
              + f" {results[scorer]['coverage']:>6.3f}")

    print(f"\n  slices (nDCG@10):")
    for slice_name in slices:
        cells = []
        for scorer in scorers_this_run:
            s = results[scorer]["slices"][slice_name]
            cells.append(f"{scorer}={s['ndcg@10']['value']:.4f}" if s.get("available")
                         else f"{scorer}=n/a")
        n = results[scorers_this_run[0]]["slices"][slice_name].get("n_impressions", 0)
        print(f"    {slice_name:<12} n={n:>6,}  " + "  ".join(cells))
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
