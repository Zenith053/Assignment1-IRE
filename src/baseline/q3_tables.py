#!/usr/bin/env python3
"""Q3 step 4: turn the saved test scores into the report's tables.

Reads every `reports/q3/<ds>_<variant>_seed<N>.json` and its per-candidate
test scores (`data/feature_store/<ds>/q3_runs/<variant>_seed<N>_test_scores.npz`),
and produces, per dataset:

  1. variants  - test metrics per seed, mean and std over seeds, mean val AUC
  2. selection - which improved variant is "the" improvement, chosen on mean
                 *validation* AUC among variants with a full set of seeds.
                 Test numbers never enter the choice.
  3. main      - improved vs NRMS: paired bootstrap 95% CI per seed (same seed
                 against same seed) and seed-averaged (each impression's metric
                 averaged over seeds first, so training noise is averaged out
                 and the CI reflects only which impressions were sampled)
  4. ablation  - each single-seed arm vs NRMS and vs the full model (seed 13),
                 and learned gate vs plain sum
  5. reference - NRMS and the improved model vs A1's scorers (semantic, BM25,
                 frozen train popularity), rescored here on the identical
                 test impressions so the comparison is paired too

Every comparison in a dataset shares one set of 10,000 bootstrap resamples
(`metrics.paired_bootstrap_ci_many`), so the CIs are mutually comparable.

Writes reports/q3_summary.json and reports/q3_summary.md.

Usage
-----
    python src/baseline/q3_tables.py
    python src/baseline/q3_tables.py --datasets mind --n-boot 2000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config  # noqa: E402
from src.eval import metrics as M  # noqa: E402

METRICS = ("auc", "mrr", "ndcg@5", "ndcg@10")
BASELINE = "nrms"
# Pre-registered improvement candidates per dataset (gate first = the planned model).
IMPROVED_CANDIDATES = {"mind": ["nrms_pop", "nrms_pop_sum"],
                       "ebnerd": ["nrms_popfresh", "nrms_popfresh_sum"]}
FULL_MODEL_FOR_ABLATION = {"mind": "nrms_pop", "ebnerd": "nrms_popfresh"}
A1_SCORERS = ("semantic", "bm25", "popularity")
ABLATION_SEED = 13
RUN_FILE = re.compile(r"^(?P<ds>mind|ebnerd)_(?P<variant>.+)_seed(?P<seed>\d+)\.json$")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def discover_runs(dataset: str) -> dict[str, dict[int, dict]]:
    runs: dict[str, dict[int, dict]] = defaultdict(dict)
    for path in sorted((REPO_ROOT / "reports" / "q3").glob(f"{dataset}_*.json")):
        m = RUN_FILE.match(path.name)
        if m:
            runs[m["variant"]][int(m["seed"])] = json.loads(path.read_text(encoding="utf-8"))
    return dict(runs)


def per_impression(offsets: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> dict[str, np.ndarray]:
    """Official per-impression metrics; undefined AUC (all-click impression) is NaN."""
    n = len(offsets) - 1
    out = {m: np.empty(n) for m in METRICS}
    for i in range(n):
        lo, hi = offsets[i], offsets[i + 1]
        lab, s = labels[lo:hi], scores[lo:hi]
        a = M.auc(lab, s)
        out["auc"][i] = np.nan if a is None else a
        out["mrr"][i] = M.mrr(lab, s)
        out["ndcg@5"][i] = M.ndcg(lab, s, 5)
        out["ndcg@10"][i] = M.ndcg(lab, s, 10)
    return out


def load_scores(cfg, variant: str, seed: int) -> dict[str, np.ndarray]:
    z = np.load(cfg.features / "q3_runs" / f"{variant}_seed{seed}_test_scores.npz")
    return {k: z[k] for k in z.files}


def check_aligned(ref: dict, other: dict, name: str) -> None:
    for k in ("offsets", "cand_rows", "labels"):
        if not np.array_equal(ref[k], other[k]):
            raise ValueError(f"{name}: '{k}' differs from the reference run - not a paired comparison")


def a1_scores(cfg, ref: dict) -> dict[str, np.ndarray]:
    """A1's scorers on the identical test candidates, cached next to the NRMS scores."""
    cache = cfg.features / "q3_runs" / "a1_scorers_test_scores.npz"
    if cache.exists():
        z = np.load(cache)
        if all(np.array_equal(z[k], ref[k]) for k in ("offsets", "cand_rows", "labels")):
            return {s: z[s] for s in A1_SCORERS}

    from src.baseline.news_data import load_news_tokens
    from src.common.io import read_table
    from src.rerank.candidates import from_inview
    from src.rerank.features import build_context, profiles_for, score_base_scorers

    print(f"  [{cfg.dataset}] rescoring A1 scorers on the test split (cached afterwards)")
    ctx = build_context(cfg)
    imps = read_table(cfg.processed / "test" / "impressions.parquet", "impressions")
    if not np.array_equal(imps["impression_id"].to_numpy(), ref["impression_id"]):
        raise ValueError("test impression order differs from the NRMS runs")
    cand = from_inview(imps, ctx.row_of)
    base = score_base_scorers(ctx, cand, profiles_for(ctx, cand, "test"), topk=5)

    # Map A1's candidate ids onto NRMS article rows to prove alignment.
    row_of = load_news_tokens(cfg).row_of
    rows = np.fromiter((row_of.get(a, -1) for a in cand.flat_ids), dtype=np.int64, count=len(cand.flat_ids))
    labels = np.concatenate(cand.labels_by_imp).astype(np.int8)
    if not (np.array_equal(cand.offsets, ref["offsets"]) and np.array_equal(rows, ref["cand_rows"])
            and np.array_equal(labels, ref["labels"])):
        raise ValueError("A1 candidate lists do not line up with the NRMS test arrays")

    out = {s: base[s].astype(np.float32) for s in A1_SCORERS}
    np.savez(cache, offsets=ref["offsets"], cand_rows=ref["cand_rows"], labels=ref["labels"], **out)
    return out


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #

def nanmean(x: np.ndarray) -> float:
    return float(np.nanmean(x))


def seed_average(per_seed: dict[int, dict[str, np.ndarray]], seeds) -> dict[str, np.ndarray]:
    return {m: np.mean([per_seed[s][m] for s in seeds], axis=0) for m in METRICS}


def analyse(dataset: str, n_boot: int) -> dict:
    cfg = load_config(REPO_ROOT / "config" / f"{dataset}.yaml")
    runs = discover_runs(dataset)
    if BASELINE not in runs:
        raise SystemExit(f"{dataset}: no baseline runs in reports/q3/")

    # --- per-impression metrics for every run --------------------------------
    ref, metrics = None, defaultdict(dict)
    for variant, by_seed in runs.items():
        for seed in by_seed:
            sc = load_scores(cfg, variant, seed)
            if ref is None:
                ref = sc
            check_aligned(ref, sc, f"{variant} seed {seed}")
            metrics[variant][seed] = per_impression(sc["offsets"], sc["labels"], sc["scores"])
    n_imp = len(ref["offsets"]) - 1
    print(f"  [{dataset}] {sum(len(v) for v in runs.values())} runs over {n_imp:,} identical test impressions")

    a1 = {s: per_impression(ref["offsets"], ref["labels"], v) for s, v in a1_scores(cfg, ref).items()}

    # --- 1. variants table -----------------------------------------------------
    variants = {}
    for variant, by_seed in sorted(runs.items()):
        seeds = sorted(by_seed)
        per_seed = {m: {s: nanmean(metrics[variant][s][m]) for s in seeds} for m in METRICS}
        variants[variant] = {
            "seeds": seeds,
            "test": {m: {"per_seed": per_seed[m], "mean": float(np.mean(list(per_seed[m].values()))),
                         "std": float(np.std(list(per_seed[m].values()), ddof=1)) if len(seeds) > 1 else None}
                     for m in METRICS},
            "val_auc_mean": float(np.mean([by_seed[s]["best_val_auc"] for s in seeds])),
            "val_auc_per_seed": {s: by_seed[s]["best_val_auc"] for s in seeds},
            "best_epoch_per_seed": {s: by_seed[s]["best_epoch"] for s in seeds},
            "variant": by_seed[seeds[0]].get("variant"),
        }

    # --- 2. selection on validation only ---------------------------------------
    base_seeds = sorted(runs[BASELINE])
    eligible = [v for v in IMPROVED_CANDIDATES[dataset]
                if v in runs and set(base_seeds) <= set(runs[v])]
    if not eligible:
        raise SystemExit(f"{dataset}: no improved variant has all baseline seeds {base_seeds}")
    improved = max(eligible, key=lambda v: variants[v]["val_auc_mean"])
    selection = {
        "rule": "highest mean validation AUC among pre-registered candidates with every baseline seed",
        "candidates": {v: variants[v]["val_auc_mean"] if v in variants else None
                       for v in IMPROVED_CANDIDATES[dataset]},
        "eligible": eligible, "selected": improved,
    }

    # --- 3-5. every paired comparison, one shared bootstrap -----------------------
    comparisons: dict[str, dict] = {}   # name -> {"a", "b", "kind", "diffs": {metric: array}}

    def add(name, kind, a_label, b_label, a_metrics, b_metrics):
        comparisons[name] = {"kind": kind, "a": a_label, "b": b_label,
                             "a_value": {m: nanmean(a_metrics[m]) for m in METRICS},
                             "b_value": {m: nanmean(b_metrics[m]) for m in METRICS},
                             "diffs": {m: a_metrics[m] - b_metrics[m] for m in METRICS}}

    for s in base_seeds:
        add(f"main/seed{s}", "main", f"{improved} seed {s}", f"{BASELINE} seed {s}",
            metrics[improved][s], metrics[BASELINE][s])
    avg_improved = seed_average(metrics[improved], base_seeds)
    avg_base = seed_average(metrics[BASELINE], base_seeds)
    add("main/seed_avg", "main", f"{improved} (mean of seeds {base_seeds})",
        f"{BASELINE} (mean of seeds {base_seeds})", avg_improved, avg_base)

    full = FULL_MODEL_FOR_ABLATION[dataset]
    s0 = ABLATION_SEED
    for variant in sorted(runs):
        if variant == BASELINE or s0 not in runs[variant] or s0 not in runs[BASELINE]:
            continue
        add(f"ablation/{variant}-vs-{BASELINE}", "ablation", f"{variant} seed {s0}",
            f"{BASELINE} seed {s0}", metrics[variant][s0], metrics[BASELINE][s0])
        if variant != full and full in runs and s0 in runs[full]:
            add(f"ablation/{variant}-vs-{full}", "ablation", f"{variant} seed {s0}",
                f"{full} seed {s0}", metrics[variant][s0], metrics[full][s0])
    gate, plain = IMPROVED_CANDIDATES[dataset]
    if gate in runs and plain in runs:
        shared = sorted(set(runs[gate]) & set(runs[plain]))
        # With one shared seed this would only repeat the "<plain>-vs-<full>" row above.
        if len(shared) > 1:
            add("ablation/sum-vs-gate", "ablation", f"{plain} (mean of seeds {shared})",
                f"{gate} (mean of seeds {shared})",
                seed_average(metrics[plain], shared), seed_average(metrics[gate], shared))

    for scorer in A1_SCORERS:
        add(f"reference/{BASELINE}-vs-a1_{scorer}", "reference", f"{BASELINE} (seed mean)",
            f"A1 {scorer}", avg_base, a1[scorer])
        add(f"reference/{improved}-vs-a1_{scorer}", "reference", f"{improved} (seed mean)",
            f"A1 {scorer}", avg_improved, a1[scorer])

    flat = {f"{name}|{m}": c["diffs"][m] for name, c in comparisons.items() for m in METRICS}
    print(f"  [{dataset}] paired bootstrap: {len(flat)} differences x {n_boot:,} resamples")
    ci = M.paired_bootstrap_ci_many(flat, n_boot=n_boot)
    for name, c in comparisons.items():
        c["ci"] = {m: ci[f"{name}|{m}"] for m in METRICS}
        del c["diffs"]

    return {
        "dataset": dataset, "scale": cfg.scale, "n_test_impressions": n_imp, "n_boot": n_boot,
        "variants": variants, "selection": selection, "improved": improved,
        "a1_reference": {s: {m: nanmean(a1[s][m]) for m in METRICS} for s in A1_SCORERS},
        "comparisons": comparisons,
    }


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #

def fmt_ci(c: dict) -> str:
    mark = "✓" if c["excludes_zero"] else "✗"
    return f"{c['mean_diff']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] {mark}"


def markdown(results: list[dict]) -> str:
    L = ["# Q3 results: NRMS baseline, popularity/freshness-aware NRMS, ablation", "",
         "Generated by `src/baseline/q3_tables.py` from `reports/q3/*.json` and the saved test scores. "
         "Δ = row model − comparison model, per-impression paired bootstrap 95% CI "
         "(✓ = CI excludes zero). All comparisons use the full test split.", ""]
    for r in results:
        ds, imp = r["dataset"], r["improved"]
        L += [f"## {ds.upper()}{' (' + r['scale'] + ')' if r['scale'] else ''} — "
              f"{r['n_test_impressions']:,} test impressions", ""]

        L += ["### Test metrics by variant", "",
              "| variant | seeds | AUC | MRR | nDCG@5 | nDCG@10 | val AUC | best epoch |",
              "|---|---|---|---|---|---|---|---|"]
        for v, info in r["variants"].items():
            cells = []
            for m in METRICS:
                t = info["test"][m]
                cells.append(f"{t['mean']:.4f}" + (f" ± {t['std']:.4f}" if t["std"] is not None else ""))
            epochs = ", ".join(str(e) for e in info["best_epoch_per_seed"].values())
            L.append(f"| `{v}` | {', '.join(map(str, info['seeds']))} | " + " | ".join(cells)
                     + f" | {info['val_auc_mean']:.4f} | {epochs} |")
        for s, vals in r["a1_reference"].items():
            L.append(f"| A1 {s} | – | " + " | ".join(f"{vals[m]:.4f}" for m in METRICS) + " | – | – |")
        L += ["", "Mean ± std over seeds (std omitted for single-seed arms).", ""]

        sel = r["selection"]
        L += ["### Which variant is the improvement (validation only)", "",
              "| candidate | mean val AUC |", "|---|---|"]
        for v, val in sel["candidates"].items():
            L.append(f"| `{v}` | {'n/a' if val is None else f'{val:.4f}'}"
                     f"{' (not all seeds)' if v not in sel['eligible'] else ''}"
                     f"{' ← selected' if v == sel['selected'] else ''} |")
        L.append("")

        for kind, title in (("main", f"Improved (`{imp}`) vs NRMS"),
                            ("ablation", "Ablation (seed 13 unless stated)"),
                            ("reference", "Against A1 scorers")):
            L += [f"### {title}", "", "| comparison | AUC | MRR | nDCG@5 | nDCG@10 |", "|---|---|---|---|---|"]
            for name, c in r["comparisons"].items():
                if c["kind"] == kind:
                    L.append(f"| {c['a']} − {c['b']} | " + " | ".join(fmt_ci(c["ci"][m]) for m in METRICS) + " |")
            L.append("")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=["mind", "ebnerd"])
    parser.add_argument("--n-boot", type=int, default=10_000)
    args = parser.parse_args(argv)

    results = [analyse(ds, args.n_boot) for ds in args.datasets]
    out_json = REPO_ROOT / "reports" / "q3_summary.json"
    out_md = REPO_ROOT / "reports" / "q3_summary.md"
    out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(markdown(results), encoding="utf-8")
    print(f"-> {out_json}\n-> {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
