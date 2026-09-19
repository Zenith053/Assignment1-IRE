#!/usr/bin/env python3
"""Q5: extended evaluation of the full two-stage retrieve-then-rank pipeline.

`src/eval/harness.py` (A1) reports AUC/MRR/nDCG/diversity/novelty/coverage
with slices and bootstrap CIs, but only for single-stage scorers ranking the
impression's own inview list. Q5 asks specifically for those same metrics
over the *two-stage* system Q2 built: retrieve top-K (~100-200) from the
whole catalogue with `UnionRetriever`, then re-rank with the trained GBDT.
That is `CandidateSet.universe == "retrieved"` (Universe B in
`evaluate_reranker.py`) - this module reuses every scorer/feature/metric
primitive already built for Q1/Q2, it just wires them through the extended
metric set and slicing that only `harness.py` had until now.

Two scorers are compared per slice, matching Q2's `before`/`after` naming so
results stay comparable with `reports/rerank_<ds>.json`:
    stage1_order   semantic similarity, standing in for the retriever's own
                   ranking (the retriever itself only returns a deduplicated
                   *set*, not a score - see `UnionRetriever.retrieve`)
    gbdt           the same Universe-A-trained booster, applied to the
                   retrieved candidates (the distribution-shift comparison)

Slices: `cold_users`/`warm_users` uses the *absolute* cold-start definition
(`is_cold`, <5 train clicks) the assignment names; it is expected to be N/A
on EB-NeRD, whose shortest history is already >=5 clicks, so
`low_history_users`/`high_history_users` (the dataset-relative quartile) is
reported alongside as the always-non-empty stand-in - this is the slice
`harness.py` mislabels as `cold_users`, corrected here rather than repeated.
`head_clicks`/`tail_clicks` follows `harness.py`'s own definition exactly.

Usage
-----
    python src/eval/evaluate_twostage.py --config config/mind.yaml
    python src/eval/evaluate_twostage.py --config config/ebnerd.yaml --sample 3500
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

from src.eval import metrics as M  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.rerank import gbdt  # noqa: E402
from src.rerank.candidates import from_inview, from_retrieval  # noqa: E402
from src.rerank.evaluate_reranker import (  # noqa: E402
    _snapshot_for, build_universe_a_frame, load_split,
)
from src.rerank.features import build_context, profiles_for, score_base_scorers  # noqa: E402
from src.rerank.features import build_frame  # noqa: E402
from src.rerank.retriever import UnionRetriever  # noqa: E402

TOP_K_LIST = 10  # list length for diversity/novelty/coverage, matching harness.py


def extended_metrics(labels_by_imp, scores_by_imp, ids_by_imp, category_of,
                     popularity, total_clicks, embeddings, row_of) -> dict:
    """Per-impression values for every Q5 metric, one scorer at a time.

    Click-based metrics (auc/mrr/ndcg) are undefined (`None`) whenever a
    candidate list has no positive - the norm for `ids_by_imp` here, since
    most retrieved lists never contain the true click (see `recall@K`
    reported alongside). Diversity/novelty/coverage need no positive label at
    all, so they stay defined even for those impressions - this is what lets
    Q5 report a "full" metric set instead of mostly None.
    """
    out = {"auc": [], "mrr": [], "ndcg@5": [], "ndcg@10": [],
          "diversity": [], "cat_entropy": [], "novelty": []}
    recommended: set[str] = set()

    for labels, scores, ids in zip(labels_by_imp, scores_by_imp, ids_by_imp):
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
            M.category_entropy([category_of.get(a, "") for a in top_ids])
        )
        out["novelty"].append(M.novelty(top_ids, popularity, total_clicks))

    return out, recommended


def summarise(values, n_boot: int) -> dict:
    point, lo, hi = M.bootstrap_ci(values, n_boot=n_boot)
    defined = sum(1 for v in values if v is not None)
    return {"value": point, "ci_low": lo, "ci_high": hi,
           "n": defined, "n_undefined": len(values) - defined}


def build_slices(impressions: pd.DataFrame, profiles: pd.DataFrame, is_head: dict) -> dict:
    """cold-start vs warm (absolute, then quartile fallback) and head vs tail clicks."""
    is_cold_of = dict(zip(profiles["user_id"], profiles["is_cold"]))
    is_low_of = dict(zip(profiles["user_id"], profiles["is_low_history"]))
    cold_mask = np.array([bool(is_cold_of.get(u, False)) for u in impressions["user_id"]])
    low_mask = np.array([bool(is_low_of.get(u, False)) for u in impressions["user_id"]])
    head_mask = np.array([
        any(is_head.get(a, False) for a in c) for c in impressions["clicked_ids"]
    ])
    return {
        "all": np.ones(len(impressions), dtype=bool),
        "cold_users": cold_mask,               # true cold-start (<5 train clicks); may be N/A
        "warm_users": ~cold_mask,
        "low_history_users": low_mask,         # dataset-relative quartile; always non-empty
        "high_history_users": ~low_mask,
        "head_clicks": head_mask,
        "tail_clicks": ~head_mask,
    }


def evaluate_scorer(name, cand, scores_by_imp, category_of, popularity, total_clicks,
                    embeddings, row_of, slices, n_boot) -> dict:
    """Per-slice metrics, reported two ways for mrr/ndcg - unlike `auc`, neither
    ever returns `None` for a zero-click impression (they return 0.0), so a
    plain mean over Universe B silently bakes the recall miss into the
    "ranking quality" number. `*_given_retrieved` restricts to impressions
    that actually retrieved the click - the real "how good is the ranking
    when it had a chance" question - and `recall_at_k * that` is the correct
    end-to-end figure, not `recall_at_k * the plain (already-diluted) mean`.
    """
    raw, recommended = extended_metrics(
        cand.labels_by_imp, scores_by_imp,
        [cand.ids_by_imp(i) for i in range(cand.n_impressions)],
        category_of, popularity, total_clicks, embeddings, row_of,
    )
    entry = {"slices": {}, "coverage": len(recommended) / max(1, len(row_of)),
             "n_distinct_recommended": len(recommended)}
    for slice_name, mask in slices.items():
        if not mask.any():
            entry["slices"][slice_name] = {"available": False, "n": 0}
            continue
        idx = np.flatnonzero(mask)
        retrieved_idx = [i for i in idx if cand.labels_by_imp[i].sum() > 0]
        recall_at_k = len(retrieved_idx) / len(idx)
        slice_entry = {
            "available": True,
            "n_impressions": int(len(idx)),
            "recall_at_k": recall_at_k,
            **{metric: summarise([raw[metric][i] for i in idx], n_boot) for metric in raw},
        }
        for metric in ("mrr", "ndcg@5", "ndcg@10"):
            cond = summarise([raw[metric][i] for i in retrieved_idx], n_boot) if retrieved_idx \
                else {"value": None, "ci_low": None, "ci_high": None, "n": 0, "n_undefined": 0}
            slice_entry[f"{metric}_given_retrieved"] = cond
        ndcg10_cond = slice_entry["ndcg@10_given_retrieved"]["value"]
        slice_entry["end_to_end_ndcg10"] = (
            recall_at_k * ndcg10_cond if ndcg10_cond is not None else 0.0
        )
        entry["slices"][slice_name] = slice_entry
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=3500,
                       help="impressions per split; matches evaluate_reranker.py's EB-NeRD run")
    parser.add_argument("--retrieve-k", type=int, default=200)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    print(f"[{cfg.dataset}] Q5: extended two-stage evaluation")
    ctx = build_context(cfg)

    imp = {s: load_split(cfg, s, args.sample) for s in ("train", "val", "test")}
    print(f"  sampled impressions: " + ", ".join(f"{s}={len(imp[s]):,}" for s in imp))

    # Train the same Universe-A GBDT Q2 uses, then apply it out-of-distribution
    # to retrieved candidates - the actual object under test for Q5.
    print("  training GBDT on Universe A (inview) train/val")
    cand_tr, _, X_tr, y_tr, g_tr, names = build_universe_a_frame(ctx, cfg, imp["train"], "train")
    cand_va, _, X_va, y_va, g_va, _ = build_universe_a_frame(ctx, cfg, imp["val"], "val")
    booster = gbdt.train(X_tr, y_tr, g_tr, X_va, y_va, g_va)
    print(f"    best iteration {booster.best_iteration}")

    print(f"  stage 1: retrieving top-{args.retrieve_k} per test user")
    retriever = UnionRetriever(cfg, ctx.bm25, ctx.embeddings, ctx.articles, imp["test"], ctx.row_of)
    profiles_test = profiles_for(ctx, from_inview(imp["test"], ctx.row_of), "test")
    retrieved = retriever.retrieve(profiles_test, ctx.articles, k=args.retrieve_k)

    cand_b = from_retrieval(imp["test"], retrieved, ctx.row_of)
    base_b = score_base_scorers(ctx, cand_b, profiles_test, topk=5)
    snap_test = _snapshot_for(cfg, "test")
    X_b, y_b, group_b, _ = build_frame(ctx, cand_b, profiles_test, snap_test, base_b)

    def per_imp(flat):
        return [flat[cand_b.offsets[i]:cand_b.offsets[i + 1]] for i in range(cand_b.n_impressions)]

    scorers = {
        "stage1_order": per_imp(base_b["semantic"]),  # stand-in: retrieve() itself returns no score
        "gbdt": gbdt.predict_per_impression(booster, X_b, cand_b.offsets),
    }

    is_head = dict(zip(ctx.articles["article_id"], ctx.articles["is_head"]))
    slices = build_slices(imp["test"], profiles_test, is_head)
    popularity = ctx.popularity
    total_clicks = int(ctx.articles["train_clicks"].sum())
    category_of = ctx.category_of

    results = {}
    for name, scores in scorers.items():
        results[name] = evaluate_scorer(
            name, cand_b, scores, category_of, popularity, total_clicks,
            ctx.embeddings, ctx.row_of, slices, args.n_boot,
        )

    overall_recall = results["gbdt"]["slices"]["all"]["recall_at_k"]

    payload = {
        "dataset": cfg.dataset, "split": "test", "scale": cfg.scale,
        "pipeline": "two-stage (UnionRetriever -> GBDT)", "retrieve_k": args.retrieve_k,
        "n_impressions": cand_b.n_impressions,
        "overall_recall_at_k": overall_recall,
        "scorers": results,
        "note": ("gbdt is the Universe-A-trained booster applied to retrieved candidates, "
                "not retrained on them - the same convention reports/rerank_<ds>.json uses, "
                "so the distribution-shift cost stays a visible, comparable number. "
                "click-based metrics (auc/mrr/ndcg) are undefined for any impression whose "
                "retrieved list never contained the true click; n_undefined in each slice "
                "reports exactly how many that is, never averaged in as zero."),
    }
    out = args.out or REPO_ROOT / "reports" / f"eval_twostage_{cfg.dataset}_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")

    print(f"\n  overall recall@{args.retrieve_k}: {overall_recall:.4f}")
    print(f"\n  {'scorer':<14} {'AUC':>18} {'MRR':>18} {'nDCG@5':>18} {'nDCG@10':>18} "
         f"{'div':>7} {'novelty':>8} {'cov':>6}")
    for name, entry in results.items():
        row = entry["slices"]["all"]
        cells = []
        for metric in ("auc", "mrr", "ndcg@5", "ndcg@10"):
            m = row[metric]
            cells.append(f"{m['value']:.4f}[{m['ci_low']:.3f},{m['ci_high']:.3f}]")
        print(f"  {name:<14} " + " ".join(f"{c:>18}" for c in cells)
             + f" {row['diversity']['value']:>7.3f} {row['novelty']['value']:>8.3f} "
             f"{entry['coverage']:>6.3f}")

    print(f"\n  slices (gbdt): recall@{args.retrieve_k}, nDCG@10 (unconditional, includes "
         f"zero-click impressions) vs nDCG@10|retrieved (conditional), and the correctly "
         f"decomposed end-to-end = recall x conditional")
    for slice_name in slices:
        s = results["gbdt"]["slices"][slice_name]
        if not s.get("available"):
            print(f"    {slice_name:<18} N/A (no impressions in this slice)")
            continue
        cond = s["ndcg@10_given_retrieved"]["value"]
        cond_str = f"{cond:.4f}" if cond is not None else "n/a"
        print(f"    {slice_name:<18} n={s['n_impressions']:>6,}  recall={s['recall_at_k']:.4f}  "
             f"ndcg@10={s['ndcg@10']['value']:.4f}  ndcg@10|retrieved={cond_str}  "
             f"end_to_end={s['end_to_end_ndcg10']:.4f}")
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
