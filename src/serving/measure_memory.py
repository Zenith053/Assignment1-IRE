#!/usr/bin/env python3
"""Q4 phase 2 (Q4.1): memory footprint of the served pipeline.

Loads `ServingPipeline` (serving mode) and measures every structure a request
touches, grouped the way the request flows:

  stage 1 index   what candidate generation reads (FAISS, BM25, popularity, profiles)
  stage 1 cache   what serving mode precomputes once so requests stop rebuilding it
  stage 2         what re-ranking reads (NRMS weights, article vectors, signals)
  ANN option      FAISS HNSW over the full catalogue, built here, the index a
                  larger catalogue would need (flat vs HNSW is the Q4 "ANN index")
  not served      loaded by the shared Q2 feature context but never read by a request

Measurement is exact for arrays/tensors/sparse matrices/FAISS indexes (their
byte buffers) and an object-graph walk for Python dicts/lists/strings. Python
objects shared between structures (e.g. article-id strings) are counted in each,
so the per-component sum is an upper bound; process RSS before/after loading is
the ground truth it is cross-checked against.

Writes reports/q4_memory_<dataset>.json.

Usage
-----
    python src/serving/measure_memory.py --config config/mind.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.serving import load_serving_config  # noqa: E402  (sets BLAS thread env first)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy import sparse  # noqa: E402

from src.common.config import load_config  # noqa: E402

MB = 1_000_000   # decimal megabytes, the unit file sizes and the design note use


# --------------------------------------------------------------------------- #
# byte counting
# --------------------------------------------------------------------------- #

def deep_bytes(obj) -> int:
    """Bytes held by an object graph, each object counted once within this call."""
    import faiss

    seen, stack, total = set(), [obj], 0
    while stack:
        o = stack.pop()
        if id(o) in seen or o is None:
            continue
        seen.add(id(o))
        if isinstance(o, np.ndarray):
            total += o.nbytes
            if o.dtype == object:
                stack.extend(o.ravel())
        elif isinstance(o, torch.Tensor):
            total += o.numel() * o.element_size()
        elif sparse.issparse(o):
            total += sum(getattr(o, a).nbytes for a in ("data", "indices", "indptr") if hasattr(o, a))
        elif isinstance(o, faiss.Index):
            total += faiss.serialize_index(o).nbytes
        elif isinstance(o, pd.DataFrame):
            for col in o.columns:
                s = o[col]
                if s.dtype == object:
                    total += s.to_numpy().nbytes
                    stack.extend(s.to_numpy())
                else:
                    total += int(s.memory_usage(deep=True, index=False))
        elif isinstance(o, dict):
            total += sys.getsizeof(o)
            stack.extend(o.keys())
            stack.extend(o.values())
        elif isinstance(o, (list, tuple, set, frozenset)):
            total += sys.getsizeof(o)
            stack.extend(o)
        elif hasattr(o, "__dict__") and not isinstance(o, type) and type(o).__module__ != "builtins":
            # A custom object (e.g. RollingPopularity): count it and follow its attributes.
            total += sys.getsizeof(o)
            stack.extend(vars(o).values())
        else:
            total += sys.getsizeof(o)
    return total


def rss_bytes() -> int:
    """Current resident set size of this process (macOS/Linux `ps`, in KB)."""
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True)
    return int(out.stdout.strip()) * 1024


def peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024   # macOS reports bytes, Linux KB


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #

def build_hnsw(vectors: np.ndarray, m: int, ef_search: int):
    import faiss
    t0 = time.perf_counter()
    index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = ef_search
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    return index, time.perf_counter() - t0


def measure(dataset: str) -> dict:
    from src.serving.pipeline import ServingPipeline

    conf = load_serving_config()
    rss_start = rss_bytes()
    # Import everything load() will import, so library code is not counted as data.
    import faiss  # noqa: F401
    import src.rerank.features  # noqa: F401
    import src.rerank.retriever  # noqa: F401
    import src.retrieval.semantic  # noqa: F401
    gc.collect()
    rss_before = rss_bytes()
    t0 = time.perf_counter()
    pipe = ServingPipeline(dataset, mode="serving").load()
    load_s = time.perf_counter() - t0
    gc.collect()                  # drop loader temporaries before reading RSS
    rss_after = rss_bytes()

    r, ctx, bm = pipe.retriever, pipe.ctx, pipe.ctx.bm25
    n_articles, n_pool = len(ctx.articles), len(r.pool_idx)
    served_articles = ctx.articles.drop(columns=[c for c in ctx.articles.columns
                                                 if c not in ("article_id", "tokens", "train_clicks")])

    components = {
        # --- stage 1: candidate generation --------------------------------------
        "stage1/faiss_flat_index (pool)": (r.faiss_index,
            f"IndexFlatIP over {n_pool:,} circulating articles x {ctx.embeddings.shape[1]} dims"),
        "stage1/a1_article_embeddings": (ctx.embeddings,
            f"{n_articles:,} x {ctx.embeddings.shape[1]} float32, read to build each user's query vector"),
        "stage1/bm25_weights (doc x term)": (bm.weights, "sparse BM25 weight matrix"),
        "stage1/bm25_weights_t (term x doc)": (bm.weights_t, "the same weights transposed, kept for fast scoring"),
        "stage1/bm25_vocabulary": (bm.vocabulary, "term -> column dict"),
        "stage1/bm25_other": ({k: v for k, v in vars(bm).items()
                               if k not in ("weights", "weights_t", "vocabulary")}, "idf, doc ids/lengths"),
        "stage1/popularity_ranker": (vars(r.popularity), "train-click ranking"),
        "stage1/pool_index": ((r.pool_idx, r._pop_allowed), "circulating-pool rows + id set"),
        "stage1/article_tokens+ids": (served_articles, "tokens BM25 queries are built from"),
        "stage1/user_profile_store": (pipe.clicks_of,
            f"{len(pipe.clicks_of):,} users' full click histories"),
        # --- stage 1: serving-mode caches ----------------------------------------
        "cache/bm25_pooled_weights_t": (pipe.bm25_pooled.weights_t, "pool-sliced BM25 matrix, built once"),
        "cache/tokens_by_id": (pipe.tokens_by_id, "article -> tokens dict, built once"),
        "cache/popularity_top_k": (pipe.pop_ids, "static top-20, built once"),
        # --- stage 2: re-ranking --------------------------------------------------
        "stage2/nrms_parameters": (list(pipe.model.parameters()),
            f"{sum(p.numel() for p in pipe.model.parameters()):,} float32 parameters"),
        "stage2/nrms_title_token_table": (pipe.model.title_tokens, "int64 token ids, only needed to encode new articles"),
        "stage2/article_vectors": (pipe.news_vecs,
            f"{tuple(pipe.news_vecs.shape)} float32, precomputed at start-up"),
        "stage2/nrms_row_index": (pipe.nrms_row_of, "article id -> row dict"),
        "stage2/click_timeline": (pipe.timeline.keys if pipe.timeline else None, "one int64 per click"),
        "stage2/publish_times": (getattr(pipe, "pub_hours", None), "float64 per article (freshness)"),
    }

    rows = {}
    for name, (obj, what) in components.items():
        if obj is None:
            continue
        rows[name] = {"mb": round(deep_bytes(obj) / MB, 2), "what": what}

    # --- ANN option: HNSW over the full catalogue ----------------------------------
    import faiss
    s5 = conf["scale_10x"]
    hnsw, build_s = build_hnsw(ctx.embeddings, s5["hnsw_m"], 64)
    flat_full = faiss.IndexFlatIP(ctx.embeddings.shape[1])
    flat_full.add(np.ascontiguousarray(ctx.embeddings))
    ann = {
        "catalogue_articles": n_articles, "dims": int(ctx.embeddings.shape[1]),
        "flat_full_catalogue_mb": round(deep_bytes(flat_full) / MB, 2),
        "hnsw_full_catalogue_mb": round(deep_bytes(hnsw) / MB, 2),
        "hnsw_m": s5["hnsw_m"], "hnsw_build_seconds_1_thread": round(build_s, 1),
    }
    ann["hnsw_overhead_mb"] = round(ann["hnsw_full_catalogue_mb"] - ann["flat_full_catalogue_mb"], 2)
    ann["hnsw_overhead_bytes_per_article"] = round(ann["hnsw_overhead_mb"] * MB / n_articles, 1)
    del hnsw, flat_full

    # --- loaded by the shared feature context, but never read by a request ---------
    unserved = {
        "rolling_popularity (Q2 features)": deep_bytes(ctx.rolling),
        "session_context (Q2 features)": deep_bytes(ctx.session),
        "other_splits_profiles": deep_bytes(ctx.profiles_all[ctx.profiles_all["split"] != pipe.split]),
        "unused_article_columns": deep_bytes(ctx.articles) - deep_bytes(served_articles),
        "test_impressions (pool + replay)": deep_bytes(pipe.impressions),
    }
    unserved = {k: round(v / MB, 2) for k, v in unserved.items()}
    accounted = sum(v["mb"] for v in rows.values()) + sum(unserved.values())

    def total(prefix):
        return round(sum(v["mb"] for k, v in rows.items() if k.startswith(prefix)), 2)

    summary = {
        "stage1_index_mb": total("stage1/"), "stage1_serving_cache_mb": total("cache/"),
        "stage2_mb": total("stage2/"),
    }
    summary["served_total_mb"] = round(sum(summary.values()), 2)

    disk = {
        "nrms_checkpoint_mb": round((REPO_ROOT / conf["models"][dataset]["checkpoint"]).stat().st_size / MB, 2),
        "feature_store_dir_mb": round(sum(f.stat().st_size for f in
                                          (REPO_ROOT / "data" / "feature_store" / dataset).glob("*")
                                          if f.is_file()) / MB, 2),
    }
    cfg = load_config(REPO_ROOT / "config" / f"{dataset}.yaml")
    return {
        "dataset": dataset, "scale": cfg.scale, "mode": "serving",
        "components": rows, "summary": summary, "ann_index_option": ann,
        "loaded_but_not_served_mb": unserved,
        "process": {"rss_python_torch_numpy_mb": round(rss_start / MB, 1),
                    "rss_after_all_imports_mb": round(rss_before / MB, 1),
                    "rss_before_load_mb": round(rss_before / MB, 1),
                    "rss_after_load_mb": round(rss_after / MB, 1),
                    "rss_growth_mb": round((rss_after - rss_before) / MB, 1),
                    "growth_accounted_by_components_mb": round(accounted, 1),
                    "growth_unaccounted_mb": round((rss_after - rss_before) / MB - accounted, 1),
                    "peak_rss_mb": round(peak_rss_bytes() / MB, 1),
                    "load_seconds": round(load_s, 1), "load_breakdown_seconds": pipe.load_seconds},
        "on_disk": disk,
        "method": "arrays/tensors/sparse/FAISS by buffer size; Python objects by object-graph walk "
                  "(shared strings counted per component, so component sum is an upper bound)",
    }


# --------------------------------------------------------------------------- #
# loader residue: why process RSS grows far more than the served structures
# --------------------------------------------------------------------------- #

LOADERS = {
    # name -> (datasets it applies to, what it parses)
    "provided_word2vec": (("ebnerd",), "load_provided_embeddings: document_vector.parquet, lists of floats"),
    "session_context": (("ebnerd",), "SessionContext: raw behaviors/history parquet for Q2 session features"),
    "rolling_popularity": (("mind", "ebnerd"), "RollingPopularity.from_impressions: all three splits"),
}


def isolate_loader(dataset: str, name: str) -> dict:
    """Run ONE loader in this (fresh) process: live result size vs RSS it leaves behind."""
    import faiss  # noqa: F401  - imports first, so only data is measured
    import src.rerank.features  # noqa: F401
    from src.common.io import read_table
    cfg = load_config(REPO_ROOT / "config" / f"{dataset}.yaml")
    gc.collect()
    r0 = rss_bytes()
    if name == "provided_word2vec":
        from src.retrieval.semantic import load_provided_embeddings
        ids = pd.read_parquet(cfg.features / "articles.parquet")["article_id"].tolist()
        obj = load_provided_embeddings(cfg, ids)
    elif name == "session_context":
        from src.rerank.timeline import SessionContext
        obj = SessionContext(cfg)
    elif name == "rolling_popularity":
        from src.rerank.timeline import RollingPopularity
        obj = RollingPopularity.from_impressions({
            s: read_table(cfg.processed / s / "impressions.parquet", "impressions")
            for s in ("train", "val", "test")})
    else:
        raise ValueError(name)
    gc.collect()
    return {"loader": name, "parses": LOADERS[name][1], "live_result_mb": round(deep_bytes(obj) / MB, 1),
            "rss_left_behind_mb": round((rss_bytes() - r0) / MB, 1),
            "peak_rss_mb": round(peak_rss_bytes() / MB, 1)}


def isolate_all_loaders(dataset: str) -> list[dict]:
    """Each loader in its own subprocess, so one loader's residue cannot hide another's."""
    out = []
    for name, (datasets, _) in LOADERS.items():
        if dataset not in datasets:
            continue
        proc = subprocess.run([sys.executable, __file__, "--config", f"config/{dataset}.yaml",
                               "--isolate", name], capture_output=True, text=True, cwd=REPO_ROOT)
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
        out.append(json.loads(line[-1]) if line else {"loader": name, "error": proc.stderr[-300:]})
    return out


def print_report(res: dict) -> None:
    print(f"\n[{res['dataset']}] memory by component (MB)")
    group = None
    for name, row in res["components"].items():
        g = name.split("/")[0]
        if g != group:
            group = g
            print(f"  -- {g}")
        print(f"    {name.split('/', 1)[1]:<34} {row['mb']:>9.2f}   {row['what']}")
    s = res["summary"]
    print(f"  stage 1 index {s['stage1_index_mb']:.1f} + serving cache {s['stage1_serving_cache_mb']:.1f}"
          f" + stage 2 {s['stage2_mb']:.1f} = served total {s['served_total_mb']:.1f} MB")
    a = res["ann_index_option"]
    print(f"  ANN option over {a['catalogue_articles']:,} articles: flat {a['flat_full_catalogue_mb']:.1f} MB,"
          f" HNSW(M={a['hnsw_m']}) {a['hnsw_full_catalogue_mb']:.1f} MB"
          f" (+{a['hnsw_overhead_bytes_per_article']:.0f} B/article, built in {a['hnsw_build_seconds_1_thread']}s)")
    print(f"  loaded but not served: {res['loaded_but_not_served_mb']}")
    p = res["process"]
    for iso in res.get("loader_residue", []):
        if "error" in iso:
            print(f"  isolated loader {iso['loader']}: FAILED")
            continue
        print(f"  isolated loader {iso['loader']:<20} live result {iso['live_result_mb']:>6.1f} MB, "
              f"RSS left behind {iso['rss_left_behind_mb']:>6.1f} MB  ({iso['parses']})")
    print(f"  process RSS: python+torch+numpy {p['rss_python_torch_numpy_mb']:.0f} MB -> after all imports "
          f"{p['rss_after_all_imports_mb']:.0f} MB -> after load {p['rss_after_load_mb']:.0f} MB "
          f"(+{p['rss_growth_mb']:.0f}: {p['growth_accounted_by_components_mb']:.0f} measured components, "
          f"{p['growth_unaccounted_mb']:.0f} unaccounted); peak {p['peak_rss_mb']:.0f} MB; load {p['load_seconds']}s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--isolate", choices=list(LOADERS), help=argparse.SUPPRESS)
    parser.add_argument("--skip-loader-isolation", action="store_true")
    args = parser.parse_args(argv)

    dataset = load_config(args.config).dataset
    if args.isolate:                                  # child process of isolate_all_loaders
        print(json.dumps(isolate_loader(dataset, args.isolate)))
        return 0
    res = measure(dataset)
    if not args.skip_loader_isolation:
        res["loader_residue"] = isolate_all_loaders(dataset)
        res["loader_residue_note"] = (
            "Process RSS after load includes memory the research loaders (Q1/Q2 feature context) "
            "leave reserved after parsing raw files into Python objects; it is not needed to "
            "serve requests. A production server would load prebuilt serving artifacts instead.")
    print_report(res)
    out = REPO_ROOT / "reports" / f"q4_memory_{dataset}.json"
    out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
