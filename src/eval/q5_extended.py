#!/usr/bin/env python3
"""A2 Q5: extended evaluation of the full two-stage pipeline.

Every metric the A1 harness defines (AUC, MRR, nDCG@5, nDCG@10, intra-list
diversity, category entropy, novelty, coverage), on two slices (cold vs warm
users, head vs tail clicks), each with a bootstrap 95% CI. Per-impression
metrics reuse `src/eval/harness.evaluate`, so definitions are A1's exactly.

Two evaluation universes, reported separately:

  A  re-rank the impression's own inview list (what Codabench scores), full test
     split, every model whose test scores are saved: A1 popularity / BM25 /
     semantic, Q3 NRMS and the Q3 improved model. Q3 models are evaluated per
     seed (13, 14, 15) and each impression's metric is averaged over seeds
     before bootstrapping, so training noise is averaged out as in Q3.

  B  the served two-stage pipeline (Q4 `ServingPipeline`): stage 1 retrieves up
     to 200 candidates from the circulating pool, stage 2 re-ranks them. Metrics
     over the retrieved list: recall@200 (a click was retrieved), accuracy
     conditional on that, end-to-end nDCG@10 (0 when nothing clicked was
     retrieved), and beyond-accuracy of the top-10 actually shown. "Stage-1 only"
     keeps the retriever's own order. Retrieved articles the user was never shown
     count as non-clicks, which understates accuracy in this universe.

Slices: cold = the test split's lowest-history quartile (`is_low_history`, as A1)
plus users with no history at all (A1's harness counted those as warm); head =
the impression has a click on a top-20%-by-train-clicks article (A1's rule).

CIs: percentile bootstrap over impressions (`paired_bootstrap_ci_many` against
zero = CI of the mean, bounded memory). Coverage CI resamples impressions and
recounts distinct recommended articles.

Writes reports/a2/q5_extended_<dataset>.json.

Usage
-----
    python src/eval/q5_extended.py --config config/mind.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.serving import load_serving_config  # noqa: E402,F401  (pins BLAS threads for the serving pipeline)

import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402

from src.baseline.news_data import load_news_tokens  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.common.io import read_table  # noqa: E402
from src.eval import metrics as M  # noqa: E402
from src.eval.harness import TOP_K_LIST, evaluate  # noqa: E402

PER_IMP = ("auc", "mrr", "ndcg@5", "ndcg@10", "diversity", "cat_entropy", "novelty")
SEEDS = (13, 14, 15)
IMPROVED = {"mind": "nrms_pop_sum", "ebnerd": "nrms_popfresh"}   # selected on validation in Q3


def ci_many(values: dict[str, np.ndarray], n_boot: int) -> dict:
    """CI of the mean of each array (NaN = undefined), all on one set of resamples."""
    res = M.paired_bootstrap_ci_many(values, n_boot=n_boot)
    return {k: {"value": v["mean_diff"], "ci_low": v["ci_low"], "ci_high": v["ci_high"], "n": v["n_paired"]}
            for k, v in res.items()}


def coverage_ci(top_lists: list[list[str]], mask: np.ndarray, catalogue: int, n_boot: int = 200,
                seed: int = 13) -> dict:
    """Distinct recommended articles / catalogue, with a bootstrap CI over impressions."""
    idx = np.flatnonzero(mask)
    ids = {}
    rows, cols = [], []
    for r, i in enumerate(idx):
        for a in top_lists[i]:
            cols.append(ids.setdefault(a, len(ids)))
            rows.append(r)
    A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(idx), max(len(ids), 1)))
    point = len(ids) / catalogue
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, len(idx), len(idx)), minlength=len(idx)).astype(np.float64)
        boots.append(float((A.T @ counts > 0).sum()) / catalogue)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"value": point, "ci_low": float(lo), "ci_high": float(hi), "distinct": len(ids), "catalogue": catalogue}


def per_imp_arrays(raw: dict) -> dict[str, np.ndarray]:
    return {m: np.array([np.nan if v is None else v for v in raw[m]], dtype=np.float64) for m in PER_IMP}


def summarise_model(metric_arrays: dict[str, np.ndarray], top_lists_by_seed: list[list[list[str]]],
                    slices: dict[str, np.ndarray], catalogue: int, n_boot: int) -> dict:
    out = {}
    for name, mask in slices.items():
        vals = {m: np.where(mask, a, np.nan) for m, a in metric_arrays.items()}
        entry = ci_many(vals, n_boot)
        covs = [coverage_ci(tl, mask, catalogue) for tl in top_lists_by_seed]
        entry["coverage"] = {k: float(np.mean([c[k] for c in covs])) for k in ("value", "ci_low", "ci_high")}
        entry["coverage"]["distinct"] = float(np.mean([c["distinct"] for c in covs]))
        entry["n_impressions"] = int(mask.sum())
        out[name] = entry
    return out


# --------------------------------------------------------------------------- #
# universe A
# --------------------------------------------------------------------------- #

def universe_a(ds: str, ctx, meta: dict, slices: dict, n_boot: int) -> dict:
    runs_dir = ctx.cfg.features / "q3_runs"
    news_ids = load_news_tokens(ctx.cfg).article_ids
    a1 = np.load(runs_dir / "a1_scorers_test_scores.npz")
    offsets, labels, cand_rows = a1["offsets"], a1["labels"], a1["cand_rows"]
    ids_by_imp = [[news_ids[r] for r in cand_rows[offsets[i]:offsets[i + 1]]] for i in range(len(offsets) - 1)]
    labels_by_imp = [labels[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]

    def split_scores(flat):
        return [flat[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]

    def run(flat):
        raw, _ = evaluate(labels_by_imp, split_scores(flat), ids_by_imp, meta["category_of"],
                          meta["popularity"], meta["total_clicks"], meta["embeddings"], meta["row_of"])
        tops = [[ids[j] for j in M.top_k_indices(s, TOP_K_LIST)] for ids, s in zip(ids_by_imp, split_scores(flat))]
        return per_imp_arrays(raw), tops

    models = {f"a1_{s}": [a1[s]] for s in ("popularity", "bm25", "semantic")}
    for variant in ("nrms", IMPROVED[ds]):
        seeds = []
        for sd in SEEDS:
            z = np.load(runs_dir / f"{variant}_seed{sd}_test_scores.npz")
            if not (np.array_equal(z["offsets"], offsets) and np.array_equal(z["cand_rows"], cand_rows)):
                raise ValueError(f"{variant} seed {sd} is not aligned with the A1 scores")
            seeds.append(z["scores"])
        models[f"q3_{variant}"] = seeds

    out = {}
    for name, score_list in models.items():
        t0 = time.time()
        arrays, tops = [], []
        for sc in score_list:
            a, t = run(sc)
            arrays.append(a)
            tops.append(t)
        mean_arrays = {m: np.mean([a[m] for a in arrays], axis=0) for m in PER_IMP}
        out[name] = {"seeds": len(score_list), **summarise_model(mean_arrays, tops, slices, meta["catalogue"], n_boot)}
        print(f"  [{ds}] A/{name}: AUC {out[name]['all']['auc']['value']:.4f}  "
              f"div {out[name]['all']['diversity']['value']:.3f}  nov {out[name]['all']['novelty']['value']:.2f}  "
              f"cov {out[name]['all']['coverage']['value']:.4f}  [{time.time() - t0:.0f}s]", flush=True)
    return out


# --------------------------------------------------------------------------- #
# universe B
# --------------------------------------------------------------------------- #

def universe_b(ds: str, pipe, meta: dict, n_requests: int, n_boot: int, seed: int = 13) -> dict:
    imps = pipe.impressions
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(imps), size=min(n_requests, len(imps)), replace=False))
    k = pipe.conf["pipeline"]["stage1"]["k_total"]
    results = {"stage1_order": [], "two_stage": []}
    clicked_sets, users = [], []
    t0 = time.time()
    for i in pick:
        row = imps.iloc[i]
        resp = pipe.handle(row["user_id"], row["timestamp"])
        clicked = set(row["clicked_ids"])
        lab = np.fromiter((1 if a in clicked else 0 for a in resp.candidate_ids), dtype=np.int8,
                          count=len(resp.candidate_ids))
        order_scores = -np.arange(len(resp.candidate_ids), dtype=np.float32)   # retriever's own order
        results["stage1_order"].append((resp.candidate_ids, lab, order_scores))
        results["two_stage"].append((resp.candidate_ids, lab, resp.scores.astype(np.float32)))
        clicked_sets.append(clicked)
        users.append(row["user_id"])
    print(f"  [{ds}] B: {len(pick):,} requests served in {time.time() - t0:.0f}s", flush=True)

    sub = imps.iloc[pick].reset_index(drop=True)
    slices = build_slices(sub, meta)
    out = {"n_requests": int(len(pick)), "k_total": k}
    for name, rows in results.items():
        ids_by_imp = [r[0] for r in rows]
        labels_by_imp = [r[1] for r in rows]
        scores_by_imp = [r[2] for r in rows]
        raw, _ = evaluate(labels_by_imp, scores_by_imp, ids_by_imp, meta["category_of"], meta["popularity"],
                          meta["total_clicks"], meta["embeddings"], meta["row_of"])
        arrays = per_imp_arrays(raw)
        retrieved = np.array([l.sum() > 0 for l in labels_by_imp], dtype=np.float64)
        # conditional accuracy only where a click was retrieved; end-to-end nDCG@10 counts misses as 0
        for m in ("auc", "mrr", "ndcg@5", "ndcg@10"):
            arrays[m] = np.where(retrieved > 0, arrays[m], np.nan)
        arrays["recall@200"] = retrieved
        arrays["ndcg@10_end_to_end"] = np.where(retrieved > 0, arrays["ndcg@10"], 0.0)
        tops = [[ids[j] for j in M.top_k_indices(s, TOP_K_LIST)] for ids, s in zip(ids_by_imp, scores_by_imp)]
        entry = {}
        for sl, mask in slices.items():
            vals = {m: np.where(mask, a, np.nan) for m, a in arrays.items()}
            e = ci_many(vals, n_boot)
            e["coverage"] = coverage_ci(tops, mask, meta["catalogue"])
            e["n_impressions"] = int(mask.sum())
            entry[sl] = e
        out[name] = entry
        a = entry["all"]
        print(f"  [{ds}] B/{name}: recall@200 {a['recall@200']['value']:.3f}  AUC|retrieved {a['auc']['value']:.4f}  "
              f"nDCG@10 e2e {a['ndcg@10_end_to_end']['value']:.4f}  cov {a['coverage']['value']:.4f}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# shared
# --------------------------------------------------------------------------- #

def build_slices(impressions, meta) -> dict[str, np.ndarray]:
    users = impressions["user_id"].astype(str)
    cold = users.map(lambda u: u in meta["low_history"] or u not in meta["has_profile"]).to_numpy()
    head = np.array([any(meta["is_head"].get(a, False) for a in c) for c in impressions["clicked_ids"]])
    return {"all": np.ones(len(impressions), dtype=bool), "cold_users": cold, "warm_users": ~cold,
            "head_clicks": head, "tail_clicks": ~head}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--b-requests", type=int, default=20000)
    args = parser.parse_args(argv)

    from src.serving.pipeline import ServingPipeline

    ds = load_config(args.config).dataset
    t0 = time.time()
    pipe = ServingPipeline(ds, mode="serving").load()           # also builds the A1/Q2 feature context
    ctx = pipe.ctx
    arts = ctx.articles
    profiles = ctx.profiles_all[ctx.profiles_all["split"] == "test"]
    meta = {
        "category_of": dict(zip(arts["article_id"], arts["category"])),
        "popularity": dict(zip(arts["article_id"], arts["train_clicks"])),
        "total_clicks": int(arts["train_clicks"].sum()),
        "is_head": dict(zip(arts["article_id"], arts["is_head"])),
        "embeddings": ctx.embeddings, "row_of": ctx.row_of, "catalogue": len(arts),
        "low_history": set(profiles.loc[profiles["is_low_history"], "user_id"].astype(str)),
        "has_profile": set(profiles["user_id"].astype(str)),
    }
    test = read_table(ctx.cfg.processed / "test" / "impressions.parquet", "impressions")
    slices = build_slices(test, meta)
    print(f"[{ds}] loaded in {time.time() - t0:.0f}s; test {len(test):,} impressions; "
          f"cold {slices['cold_users'].mean():.1%}, head {slices['head_clicks'].mean():.1%}", flush=True)

    result = {
        "dataset": ds, "scale": ctx.cfg.scale, "n_boot": args.n_boot, "top_k_list": TOP_K_LIST,
        "definitions": {
            "cold_users": "test-split lowest-history quartile (is_low_history) or no history",
            "head_clicks": "impression has a click on a top-20%-by-train-clicks article",
            "diversity": "1 - mean pairwise cosine of the top-10's A1 embeddings",
            "cat_entropy": "Shannon entropy (bits) of the top-10's categories",
            "novelty": "mean -log2 p(article), p from train clicks (+1 smoothing)",
            "coverage": "distinct articles in any top-10 / catalogue",
        },
        "slice_share": {k: float(v.mean()) for k, v in slices.items()},
        "improved_variant": IMPROVED[ds],
        "universe_b_model": pipe.conf["models"][ds]["checkpoint"],
    }
    result["universe_a"] = universe_a(ds, ctx, meta, slices, args.n_boot)
    result["universe_b"] = universe_b(ds, pipe, meta, args.b_requests, args.n_boot)
    out = REPO_ROOT / "reports" / "a2" / f"q5_extended_{ds}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"-> {out} [{(time.time() - t0) / 60:.1f} min]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
