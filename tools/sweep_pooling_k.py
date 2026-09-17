#!/usr/bin/env python3
"""Sweep k for top-k similarity pooling in the semantic scorer (design note §4.1).

Scores each candidate by the mean of its k highest cosine similarities to the
user's individual history clicks, for a range of k, and reports val AUC per k
plus paired bootstrap CIs on the differences. Two endpoints anchor the sweep:
k=1 is max pooling, and k>=|history| is *exactly* mean pooling (the mean of a
candidate's similarities to every click equals its similarity to the mean
history vector, up to a per-user constant that cannot reorder one impression),
so the run also emits a `mean_pool_reference` the largest k must reproduce.

Usage
-----
    python tools/sweep_pooling_k.py --config config/mind.yaml \
        --out reports/sweep_pooling_k_mind_val.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config  # noqa: E402
from src.common.io import read_table  # noqa: E402
from src.eval import metrics as M  # noqa: E402
from src.retrieval.semantic import (  # noqa: E402
    build_user_history_rows, encode_articles, l2_normalize, load_provided_embeddings,
)

KS = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 10 ** 5]  # last entry >= any history length


def paired_bootstrap(a: list, b: list, n_boot: int = 1000, seed: int = 13) -> dict:
    """Bootstrap CI on the per-impression AUC difference between two k values.

    Paired on the impression, so the shared per-impression difficulty cancels
    and the CI answers "does k=a beat k=b", not "do their means differ".
    """
    diff = np.array([x - y for x, y in zip(a, b) if x is not None and y is not None])
    rng = np.random.default_rng(seed)
    draws = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    return {"delta": float(diff.mean()),
            "ci_low": float(np.percentile(draws, 2.5)),
            "ci_high": float(np.percentile(draws, 97.5))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"],
                        help="val by default: k is a hyperparameter, so it is chosen "
                             "off the split the hybrid combiner is also fit on")
    parser.add_argument("--sample", type=int, default=20000)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    articles = pd.read_parquet(cfg.features / "articles.parquet")
    profiles_all = pd.read_parquet(cfg.features / "user_profiles.parquet")
    profiles_all["clicked_ids"] = profiles_all["clicked_ids"].map(list)

    article_ids = articles["article_id"].tolist()
    row_of = {a: i for i, a in enumerate(article_ids)}
    raw_vectors = (load_provided_embeddings(cfg, article_ids)
                   if cfg.can("has_provided_embeddings")
                   else encode_articles(cfg, articles, batch_size=128))
    embeddings = l2_normalize(raw_vectors)

    impressions = read_table(cfg.processed / args.split / "impressions.parquet", "impressions")
    if args.sample and args.sample < len(impressions):
        impressions = impressions.sample(args.sample, random_state=13)
    impressions = impressions.reset_index(drop=True)

    needed = set(impressions["user_id"])
    profiles = profiles_all[
        (profiles_all["split"] == args.split) & (profiles_all["user_id"].isin(needed))
    ].reset_index(drop=True)
    user_ids, rows_list = build_user_history_rows(profiles, row_of)
    history_of = dict(zip(user_ids, rows_list))

    # Sort each candidate's similarity row once; every k is then a suffix mean.
    labels_by_imp, sims_by_imp, history_len = [], [], []
    started = time.time()
    for i in range(len(impressions)):
        ids = impressions["inview_ids"].iat[i]
        hist_rows = history_of.get(impressions["user_id"].iat[i])
        if hist_rows is None or len(hist_rows) == 0:
            continue
        doc_rows = np.array([row_of.get(a, -1) for a in ids])
        ok = doc_rows >= 0
        if not ok.any():
            continue
        clicked = set(impressions["clicked_ids"].iat[i])
        sims = np.zeros((len(ids), len(hist_rows)), dtype=np.float32)
        sims[ok] = np.sort(embeddings[doc_rows[ok]] @ embeddings[hist_rows].T, axis=1)
        labels_by_imp.append(np.array([1 if a in clicked else 0 for a in ids]))
        sims_by_imp.append(sims)
        history_len.append(len(hist_rows))
    print(f"  scored {len(labels_by_imp):,} impressions in {time.time() - started:.1f}s")

    per_k: dict[int, list] = {}
    sweep = {}
    for k in KS:
        values = [M.auc(labels_by_imp[i], sims[:, -min(k, sims.shape[1]):].mean(axis=1))
                  for i, sims in enumerate(sims_by_imp)]
        per_k[k] = values
        defined = [v for v in values if v is not None]
        sweep[str(k)] = {"auc": float(np.mean(defined)), "n": len(defined)}

    # Independent mean-pool baseline: cosine to the (unnormalised) mean history vector.
    mean_pool = [M.auc(labels_by_imp[i], sims.mean(axis=1)) for i, sims in enumerate(sims_by_imp)]
    best_k = int(max(sweep, key=lambda k: sweep[k]["auc"]))

    lengths = np.array(history_len)
    quartiles = np.percentile(lengths, [25, 50, 75])
    bands = {"q1_shortest": lengths <= quartiles[0],
             "q2": (lengths > quartiles[0]) & (lengths <= quartiles[1]),
             "q3": (lengths > quartiles[1]) & (lengths <= quartiles[2]),
             "q4_longest": lengths > quartiles[2]}
    by_band = {}
    for name, mask in bands.items():
        rows = np.where(mask)[0]
        entry = {"n": int(mask.sum()), "median_history": float(np.median(lengths[mask]))}
        for k in (1, 5, 20, 10 ** 5):
            defined = [per_k[k][i] for i in rows if per_k[k][i] is not None]
            entry[f"k={k}"] = float(np.mean(defined)) if defined else None
        by_band[name] = entry

    result = {
        "dataset": cfg.dataset, "split": args.split, "n_impressions": len(labels_by_imp),
        "history_len": {"median": float(np.median(lengths)), "mean": float(lengths.mean()),
                        "p25": float(quartiles[0]), "p75": float(quartiles[2]),
                        "frac_used_at_k5_median": float(np.median(np.minimum(5, lengths) / lengths))},
        "sweep": sweep,
        "mean_pool_reference": float(np.mean([v for v in mean_pool if v is not None])),
        "best_k": best_k,
        "paired": {
            "k5_vs_meanpool": paired_bootstrap(per_k[5], per_k[10 ** 5], args.n_boot),
            "k5_vs_k1": paired_bootstrap(per_k[5], per_k[1], args.n_boot),
            "k5_vs_k20": paired_bootstrap(per_k[5], per_k[20], args.n_boot),
            f"best_k{best_k}_vs_k5": paired_bootstrap(per_k[best_k], per_k[5], args.n_boot),
        },
        "by_history_band": by_band,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"  wrote {args.out} (best k={best_k}, AUC {sweep[str(best_k)]['auc']:.4f}, "
          f"mean-pool {result['mean_pool_reference']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
