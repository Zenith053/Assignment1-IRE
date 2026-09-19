#!/usr/bin/env python3
"""Q2: train both re-rankers, score before/after, report Q2.1's retrieval ceiling.

Two things are reported, on purpose kept apart rather than blended into one
number:

  Universe A (`--candidates inview`) - re-rank the impression's own inview
  list, exactly what the Codabench leaderboards and A1's harness score. This is
  where "before vs after re-ranking" (Q2.4) is a fair, paired comparison: same
  impressions, same labels, only the score changes.

  Universe B (`--candidates retrieved`) - Q2.1 literally asks to retrieve top-K
  with A1's candidate generator, then re-rank. A1 already measured circulating-
  pool recall@50 at 0.079 (MIND) / 0.037 (EB-NeRD), so most clicks are absent
  from a retrieved list before a re-ranker ever sees it. Reporting only a
  Universe-B AUC would hide that: `metrics.auc` returns None for the ~90% of
  impressions where the click was never retrieved, and averaging around that
  silently mixes "the ranker did nothing wrong" with "the ranker had nothing to
  find". So Universe B reports recall@K first, and nDCG@10 conditional on the
  click being retrieved second - multiplying the two gives the honest
  end-to-end number.

Usage
-----
    python src/rerank/evaluate_reranker.py --config config/mind.yaml
    python src/rerank/evaluate_reranker.py --config config/ebnerd.yaml --sample 8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config  # noqa: E402
from src.common.io import read_table  # noqa: E402
from src.eval import metrics as M  # noqa: E402
from src.rerank import gbdt, mlp  # noqa: E402
from src.rerank.candidates import CandidateSet, from_inview, from_retrieval  # noqa: E402
from src.rerank.features import build_context, build_frame, profiles_for, score_base_scorers  # noqa: E402
from src.rerank.retriever import UnionRetriever  # noqa: E402

TOP_K_RETRIEVE = 200


def load_split(cfg, split: str, sample: int, seed: int = 13) -> pd.DataFrame:
    impressions = read_table(cfg.processed / split / "impressions.parquet", "impressions")
    if sample and sample < len(impressions):
        impressions = impressions.sample(sample, random_state=seed)
    return impressions.reset_index(drop=True)


def per_impression_metrics(labels_by_imp, scores_by_imp) -> dict[str, list]:
    return {
        "auc": [M.auc(l, s) for l, s in zip(labels_by_imp, scores_by_imp)],
        "mrr": [M.mrr(l, s) for l, s in zip(labels_by_imp, scores_by_imp)],
        "ndcg@5": [M.ndcg(l, s, 5) for l, s in zip(labels_by_imp, scores_by_imp)],
        "ndcg@10": [M.ndcg(l, s, 10) for l, s in zip(labels_by_imp, scores_by_imp)],
    }


def summarise(values: list, n_boot: int) -> dict:
    point, lo, hi = M.bootstrap_ci(values, n_boot=n_boot)
    defined = sum(1 for v in values if v is not None)
    return {"value": point, "ci_low": lo, "ci_high": hi, "n": defined,
           "n_undefined": len(values) - defined}


def build_universe_a_frame(ctx, cfg, impressions: pd.DataFrame, split_name: str):
    """Universe A candidate set + feature frame for one split."""
    cand = from_inview(impressions, ctx.row_of)
    profiles = profiles_for(ctx, cand, split_name)
    base = score_base_scorers(ctx, cand, profiles, topk=5)
    # The snapshot dwell-time features must read - independent of whether this
    # dataset even has dwell time; `SessionContext` short-circuits to NaN when
    # `not available`, so this is correct (if unused) on MIND too.
    history_snapshot = _snapshot_for(cfg, split_name)
    X, y, group, names = build_frame(ctx, cand, profiles, history_snapshot, base)
    return cand, base, X, y, group, names


def _snapshot_for(cfg, split_name: str) -> str:
    meta = json.loads((cfg.processed / "split_meta.json").read_text(encoding="utf-8"))
    return meta["splits"][split_name]["history_snapshot"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=8000,
                       help="impressions per split; A1's own default elsewhere is 20000")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--retrieve-k", type=int, default=TOP_K_RETRIEVE)
    parser.add_argument("--retrieve-sample", type=int, default=None,
                       help="Universe B on this many of the sampled test impressions (default: all); "
                            "its feature frame is retrieve_k rows per impression")
    parser.add_argument("--diagnose-single-feature", action="store_true",
                       help="sanity-check GBDT plumbing with a semantic-only / bm25-only model")
    parser.add_argument("--skip-retrieval", action="store_true",
                       help="skip Universe B (Q2.1's retrieve-then-rank); Universe A only")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    print(f"[{cfg.dataset}] building feature context (BM25 index, embeddings, "
         f"rolling popularity, session context)")
    t0 = time.time()
    ctx = build_context(cfg)
    print(f"  context ready in {time.time() - t0:.1f}s "
         f"(session features {'available' if ctx.session.available else 'N/A - has_session_id=false'})")

    imp = {s: load_split(cfg, s, args.sample) for s in ("train", "val", "test")}
    print(f"  sampled impressions: " + ", ".join(f"{s}={len(imp[s]):,}" for s in imp))

    # ---------------- Universe A: re-rank the inview list ----------------
    print("\n[Universe A] building feature frames (inview candidates)")
    frames = {}
    for split_name in ("train", "val", "test"):
        cand, base, X, y, group, names = build_universe_a_frame(ctx, cfg, imp[split_name], split_name)
        frames[split_name] = {"cand": cand, "base": base, "X": X, "y": y, "group": group, "names": names}
        print(f"  {split_name:<6} {X.shape[0]:>7,} candidates over {len(group):>6,} impressions "
             f"({int(y.sum())} clicks)")

    print("\n[Universe A] training GBDT (LightGBM LambdaRank)")
    booster = gbdt.train(
        frames["train"]["X"], frames["train"]["y"], frames["train"]["group"],
        frames["val"]["X"], frames["val"]["y"], frames["val"]["group"],
    )
    print(f"  best iteration {booster.best_iteration}")

    print("\n[Universe A] training MLP (listwise softmax ranker)")
    model, scaler, best_val_auc = mlp.train(
        frames["train"]["X"], frames["train"]["y"], frames["train"]["group"],
        frames["val"]["X"], frames["val"]["y"], frames["val"]["group"],
    )
    print(f"  best val AUC during training: {best_val_auc:.4f}")

    test = frames["test"]
    test_cand: CandidateSet = test["cand"]
    labels_test = test_cand.labels_by_imp

    # `base` arrays are flat; slice per impression the same way `CandidateSet` does.
    def _per_imp(flat: np.ndarray) -> list[np.ndarray]:
        return [flat[test_cand.offsets[i]:test_cand.offsets[i + 1]] for i in range(test_cand.n_impressions)]

    scorers_test = {
        "semantic": _per_imp(test["base"]["semantic"]),        # A1's best single scorer on MIND
        "bm25": _per_imp(test["base"]["bm25"]),
        "popularity": _per_imp(test["base"]["popularity"]),
        "gbdt": gbdt.predict_per_impression(booster, test["X"], test_cand.offsets),
        "mlp": mlp.predict_per_impression(model, scaler, test["X"], test_cand.offsets),
    }

    print("\n[Universe A] test-split metrics (before vs after re-ranking)")
    per_scorer_metrics = {}
    for name, scores in scorers_test.items():
        raw = per_impression_metrics(labels_test, scores)
        per_scorer_metrics[name] = {m: summarise(v, args.n_boot) for m, v in raw.items()}
        print(f"  {name:<12} AUC {per_scorer_metrics[name]['auc']['value']:.4f}  "
             f"MRR {per_scorer_metrics[name]['mrr']['value']:.4f}  "
             f"nDCG@5 {per_scorer_metrics[name]['ndcg@5']['value']:.4f}  "
             f"nDCG@10 {per_scorer_metrics[name]['ndcg@10']['value']:.4f}")

    if args.diagnose_single_feature:
        # Sanity check on the training/eval plumbing, not the feature set: a
        # tree restricted to exactly one monotonic feature should reproduce
        # that feature's own AUC almost exactly. If a single-feature GBDT
        # cannot match the raw score's AUC, the bug is in how X/y/group are
        # built or scored - not in interactions between many features - and
        # that would need fixing before any amount of regularisation helps.
        print("\n[diagnostic] single-feature GBDT sanity check")
        names = frames["train"]["names"]
        for feat_name in ("semantic", "bm25"):
            idx = names.index(feat_name)
            b1 = gbdt.train(
                frames["train"]["X"][:, [idx]], frames["train"]["y"], frames["train"]["group"],
                frames["val"]["X"][:, [idx]], frames["val"]["y"], frames["val"]["group"],
                feature_names=[feat_name], num_boost_round=200,
            )
            single_scores = gbdt.predict_per_impression(b1, test["X"][:, [idx]], test_cand.offsets)
            raw = per_impression_metrics(labels_test, single_scores)
            point, _, _ = M.bootstrap_ci(raw["auc"], n_boot=100)
            raw_auc = per_scorer_metrics[feat_name]["auc"]["value"]
            print(f"  GBDT({feat_name} only) test AUC = {point:.4f}  vs raw {feat_name} = {raw_auc:.4f}  "
                 f"(gap {point - raw_auc:+.4f})")

    # Paired bootstrap: does each re-ranker beat the best single-feature "before"?
    best_before = max(("semantic", "bm25", "popularity"),
                      key=lambda n: per_scorer_metrics[n]["auc"]["value"])
    print(f"\n[Universe A] paired bootstrap CI, after vs before='{best_before}' "
         f"(95% CI excluding zero = statistically significant gain)")
    raw_before = per_impression_metrics(labels_test, scorers_test[best_before])
    paired = {}
    for after_name in ("gbdt", "mlp"):
        raw_after = per_impression_metrics(labels_test, scorers_test[after_name])
        paired[after_name] = {
            metric: M.paired_bootstrap_ci(raw_after[metric], raw_before[metric], n_boot=10_000)
            for metric in raw_after
        }
        for metric, res in paired[after_name].items():
            flag = "SIGNIFICANT" if res["excludes_zero"] else "not significant"
            print(f"  {after_name:<5} {metric:<8} delta={res['mean_diff']:+.4f} "
                 f"[{res['ci_low']:+.4f}, {res['ci_high']:+.4f}]  n={res['n_paired']}  {flag}")

    importances = gbdt.feature_importance(booster)
    print("\n[Universe A] top-10 GBDT feature importances (gain, normalised)")
    for name, imp_val in importances[:10]:
        print(f"  {name:<28} {imp_val:.3f}")

    result = {
        "dataset": cfg.dataset, "candidates": "inview",
        "n_impressions": {s: len(imp[s]) for s in imp},
        "best_before": best_before,
        "metrics": per_scorer_metrics,
        "paired_vs_before": paired,
        "gbdt_feature_importance": [{"feature": n, "gain": float(g)} for n, g in importances],
        "mlp_best_val_auc": best_val_auc,
    }

    # ---------------- Universe B: retrieve top-K, then re-rank ----------------
    if not args.skip_retrieval:
        print(f"\n[Universe B] stage 1: retrieving top-{args.retrieve_k} per user")
        retriever = UnionRetriever(cfg, ctx.bm25, ctx.embeddings, ctx.articles, imp["test"], ctx.row_of)
        imp_b = imp["test"]
        if args.retrieve_sample and args.retrieve_sample < len(imp_b):
            imp_b = imp_b.sample(args.retrieve_sample, random_state=13).reset_index(drop=True)
        profiles_test = profiles_for(ctx, from_inview(imp_b, ctx.row_of), "test")
        retrieved = retriever.retrieve(profiles_test, ctx.articles, k=args.retrieve_k)

        cand_b = from_retrieval(imp_b, retrieved, ctx.row_of)
        n_with_click = sum(1 for l in cand_b.labels_by_imp if l.sum() > 0)
        recall_at_k = n_with_click / max(1, cand_b.n_impressions)
        print(f"  recall@{args.retrieve_k}: {recall_at_k:.4f} "
             f"({n_with_click:,}/{cand_b.n_impressions:,} impressions retrieved a click)")

        base_b = score_base_scorers(ctx, cand_b, profiles_test, topk=5)
        history_snapshot_test = _snapshot_for(cfg, "test")
        X_b, y_b, group_b, _ = build_frame(ctx, cand_b, profiles_test, history_snapshot_test, base_b)

        gbdt_b_scores = gbdt.predict_per_impression(booster, X_b, cand_b.offsets)
        stage1_scores = _per_imp_generic(base_b["semantic"], cand_b.offsets)

        # Conditional on the click being retrieved. M.ndcg / M.mrr return 0.0 (not None)
        # for an impression with no positive, so filtering on `is not None` - as this
        # code previously did - kept every impression: the "conditional" figure was the
        # unconditional mean and end_to_end applied recall twice. Select the impressions
        # with a retrieved click explicitly instead (fix found in the Q5 pass,
        # src/eval/evaluate_twostage.py).
        retrieved_idx = [i for i, l in enumerate(cand_b.labels_by_imp) if l.sum() > 0]
        labels_r = [cand_b.labels_by_imp[i] for i in retrieved_idx]
        raw_b_before = per_impression_metrics(labels_r, [stage1_scores[i] for i in retrieved_idx])
        raw_b_after = per_impression_metrics(labels_r, [gbdt_b_scores[i] for i in retrieved_idx])
        cond = {name: {m: summarise(v, args.n_boot) for m, v in raw.items()}
                for name, raw in (("stage1_order", raw_b_before), ("gbdt_inview_applied_to_retrieved", raw_b_after))}
        paired_b = {m: M.paired_bootstrap_ci(raw_b_after[m], raw_b_before[m], n_boot=10_000) for m in raw_b_after}
        end_to_end_before = recall_at_k * cond["stage1_order"]["ndcg@10"]["value"]
        end_to_end_after = recall_at_k * cond["gbdt_inview_applied_to_retrieved"]["ndcg@10"]["value"]

        print(f"  conditional on a retrieved click ({len(retrieved_idx):,} impressions):")
        for name, c in cond.items():
            print(f"    {name:<34} AUC {c['auc']['value']:.4f}  MRR {c['mrr']['value']:.4f}  "
                  f"nDCG@5 {c['ndcg@5']['value']:.4f}  nDCG@10 {c['ndcg@10']['value']:.4f}")
        for m, res in paired_b.items():
            print(f"    gbdt - stage1 {m:<8} {res['mean_diff']:+.4f} [{res['ci_low']:+.4f}, {res['ci_high']:+.4f}]"
                  f"{'  SIGNIFICANT' if res['excludes_zero'] else ''}")
        print(f"  end-to-end nDCG@10 = recall@K x conditional: "
             f"before={end_to_end_before:.4f}  after={end_to_end_after:.4f}")

        result["universe_b"] = {
            "retrieve_k": args.retrieve_k,
            "n_impressions": cand_b.n_impressions,
            "recall_at_k": recall_at_k,
            "n_impressions_click_retrieved": len(retrieved_idx),
            "conditional_on_retrieval": cond,
            "paired_gbdt_vs_stage1_conditional": paired_b,
            "ndcg10_conditional_on_retrieval": {
                "stage1_order": cond["stage1_order"]["ndcg@10"]["value"],
                "gbdt_inview_applied_to_retrieved": cond["gbdt_inview_applied_to_retrieved"]["ndcg@10"]["value"],
            },
            "end_to_end_ndcg10": {"before": end_to_end_before, "after": end_to_end_after},
            "note": ("gbdt here is gbdt_inview scoring retrieved candidates, not a model "
                    "trained on retrieved candidates - reported this way on purpose, so the "
                    "distribution-shift cost (train-time negatives are editorially-selected "
                    "inview non-clicks; here they are mostly off-topic retrieved articles) "
                    "is a visible number rather than an assumption."),
        }

    out = args.out or REPO_ROOT / "reports" / f"rerank_{cfg.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"\n  -> {out}")
    return 0


def _per_imp_generic(flat: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    return [flat[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


if __name__ == "__main__":
    sys.exit(main())
