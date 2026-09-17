#!/usr/bin/env python3
"""Q4 phase 5 (Q4.4): what breaks first at 10x? Measured on a synthetic 10x world.

Loads the Phase 1 serving pipeline, then builds the article-dependent structures
at 1x and at `factor`x and replays the same real test users against both:

  articles x10   A1 article embeddings (each article copied, copies perturbed by
                 Gaussian noise and re-normalised; copy 0 is the original),
                 circulating pool, FAISS flat over the pool (what is served),
                 FAISS flat + HNSW over the whole catalogue (the ANN option:
                 build time, memory, latency and recall@k vs exact per efSearch),
                 BM25 weight matrix (rows copied) in serving and as_is modes,
                 NRMS article-vector table
  clicks x10     click timeline (every click event repeated)
  users x10      profile store, extrapolated linearly (a dict lookup does not slow down)
  traffic x10    Phase 4 queueing simulation on the 10x per-request service times

Per-request latency at 10x = the measured article-dependent stages (BM25, FAISS,
NRMS scoring, click counts) + the Phase 3 means of the stages that do not depend
on catalogue size (profile lookup, popularity, merge, features other than click
counts, sort), paired query by query.

Caveats (reported): copies are near-duplicates, so HNSW recall is measured on a
clustered synthetic set; a real 10x catalogue would also grow the BM25 vocabulary.
Index builds use every core (offline work); all timed searches use 1 thread.

FAISS runs in a child process that never imports PyTorch. This environment ships
three OpenMP runtimes (faiss, torch, sklearn); with torch loaded, multi-threaded
FAISS aborts ("OMP: Error #15 ... libomp.dylib already initialized"), and the
KMP_DUPLICATE_LIB_OK override is documented as possibly giving wrong results.

Writes reports/q4_scale_<dataset>.json.

Usage
-----
    python src/serving/scale_10x.py --config config/mind.yaml
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.serving import load_serving_config  # noqa: E402  (sets BLAS thread env first)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402

from src.common.config import load_config  # noqa: E402

MB = 1_000_000


# --------------------------------------------------------------------------- #
# synthetic scaling (pure functions, unit-tested)
# --------------------------------------------------------------------------- #

def replicate_embeddings(emb: np.ndarray, factor: int, noise_std: float, seed: int = 13) -> np.ndarray:
    """`factor` copies of every row; copy 0 exact, others perturbed then L2-normalised."""
    rng = np.random.default_rng(seed)
    n, d = emb.shape
    out = np.empty((n * factor, d), dtype=np.float32)
    out[:n] = emb
    for k in range(1, factor):
        block = emb + rng.normal(0.0, noise_std, size=emb.shape).astype(np.float32)
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        out[k * n:(k + 1) * n] = block / np.maximum(norms, 1e-12)
    return out


def replicate_index(rows: np.ndarray, n: int, factor: int) -> np.ndarray:
    """Row ids of the same items in every copy: rows, rows + n, rows + 2n, ..."""
    return np.concatenate([np.asarray(rows, dtype=np.int64) + k * n for k in range(factor)])


def recall_at_k(approx: np.ndarray, exact: np.ndarray) -> float:
    k = exact.shape[1]
    return float(np.mean([len(set(a) & set(e)) / k for a, e in zip(approx, exact)]))


def pct(values, ps=(50, 95, 99)) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {"mean": round(float(v.mean()), 4), **{f"p{p}": round(float(np.percentile(v, p)), 4) for p in ps}}


# --------------------------------------------------------------------------- #
# measurement helpers
# --------------------------------------------------------------------------- #

def time_each(fn, items, warmup: int) -> list[float]:
    for it in items[:warmup]:
        fn(it)
    out = []
    for it in items:
        t0 = time.perf_counter()
        fn(it)
        out.append(1e3 * (time.perf_counter() - t0))
    return out


def faiss_threads(n: int) -> None:
    import faiss
    faiss.omp_set_num_threads(n)


def csr_bytes(m) -> int:
    return m.data.nbytes + m.indices.nbytes + m.indptr.nbytes


def faiss_child(handoff: Path, scale: int) -> dict:
    """FAISS flat-over-pool, flat-over-catalogue and HNSW at one scale. Never imports torch."""
    import faiss

    assert "torch" not in sys.modules, "FAISS child must not load PyTorch's OpenMP runtime"
    sc = load_serving_config()["scale_10x"]
    n_q, warm = sc["n_queries"], sc["warmup_queries"]
    build_threads = os.cpu_count() if sc["build_threads"] == "all" else int(sc["build_threads"])
    z = np.load(handoff)
    base, user_vecs = z["embeddings"], np.ascontiguousarray(z["user_vecs"])
    n_articles = len(base)
    emb = base if scale == 1 else replicate_embeddings(base, scale, sc["noise_std"])
    pool = z["pool"] if scale == 1 else replicate_index(z["pool"], n_articles, scale)
    queries = list(range(n_q))
    index_mb = lambda idx: round(faiss.serialize_index(idx).nbytes / MB, 1)  # noqa: E731
    s: dict = {}

    faiss_threads(build_threads)
    flat_pool = faiss.IndexFlatIP(emb.shape[1])
    flat_pool.add(np.ascontiguousarray(emb[pool]))
    faiss_threads(1)
    t = time_each(lambda q: flat_pool.search(user_vecs[q:q + 1], sc["recall_k"]), queries, warm)
    s["faiss_flat_pool"] = {"latency_ms": pct(t), "mb": index_mb(flat_pool)}
    s["faiss_pool_times_ms"] = t
    del flat_pool

    faiss_threads(build_threads)
    flat_all = faiss.IndexFlatIP(emb.shape[1])
    flat_all.add(np.ascontiguousarray(emb))
    _, exact = flat_all.search(user_vecs, sc["recall_k"])            # ground truth, batched
    faiss_threads(1)
    t = time_each(lambda q: flat_all.search(user_vecs[q:q + 1], sc["recall_k"]), queries[:200], 5)
    s["faiss_flat_catalogue"] = {"latency_ms": pct(t), "mb": index_mb(flat_all)}
    del flat_all

    faiss_threads(build_threads)
    t0 = time.perf_counter()
    hnsw = faiss.IndexHNSWFlat(emb.shape[1], sc["hnsw_m"], faiss.METRIC_INNER_PRODUCT)
    hnsw.add(np.ascontiguousarray(emb))
    build_s = time.perf_counter() - t0
    faiss_threads(1)
    ef_rows = []
    for ef in sc["hnsw_ef_search"]:
        hnsw.hnsw.efSearch = ef
        _, approx = hnsw.search(user_vecs, sc["recall_k"])
        t = time_each(lambda q: hnsw.search(user_vecs[q:q + 1], sc["recall_k"]), queries, warm)
        ef_rows.append({"ef_search": ef, "recall_at_k": round(recall_at_k(approx, exact), 4),
                        "latency_ms": pct(t)})
    s["hnsw_catalogue"] = {"m": sc["hnsw_m"], "build_seconds": round(build_s, 1),
                           "build_threads": build_threads, "mb": index_mb(hnsw), "ef_search": ef_rows}
    return s


def run(dataset: str) -> dict:
    from src.retrieval.semantic import build_user_vectors
    from src.serving.measure_memory import deep_bytes
    from src.serving.pipeline import ServingPipeline

    import tempfile

    import torch

    conf = load_serving_config()
    sc = conf["scale_10x"]
    factor, n_q, warm = sc["factor"], sc["n_queries"], sc["warmup_queries"]
    build_threads = os.cpu_count() if sc["build_threads"] == "all" else int(sc["build_threads"])

    pipe = ServingPipeline(dataset, mode="serving").load()
    r, ctx, bm = pipe.retriever, pipe.ctx, pipe.ctx.bm25
    n_articles = len(ctx.articles)
    lat3 = json.loads((REPO_ROOT / "reports" / f"q4_latency_{dataset}.json").read_text())
    mem2 = json.loads((REPO_ROOT / "reports" / f"q4_memory_{dataset}.json").read_text())

    # --- the replayed requests: real test users -----------------------------------
    rng = np.random.default_rng(conf["latency"]["seed"])
    imps = pipe.impressions.iloc[rng.permutation(len(pipe.impressions))[:n_q]]
    users, stamps = imps["user_id"].tolist(), imps["timestamp"].to_numpy()
    profile = pd.DataFrame({"user_id": users, "clicked_ids": [pipe.clicks_of.get(u, []) for u in users]})
    _, user_vecs = build_user_vectors(profile, r.row_of, r.embeddings, False, 5.0)
    token_lists = [[tok for a in pipe.clicks_of.get(u, []) for tok in pipe.tokens_by_id.get(a, ())] for u in users]
    queries = list(range(n_q))
    handoff = Path(tempfile.mkdtemp(prefix="q4_scale_")) / "faiss_inputs.npz"
    np.savez(handoff, embeddings=r.embeddings, pool=r.pool_idx, user_vecs=user_vecs)
    out = {"dataset": dataset, "factor": factor, "n_queries": n_q, "articles_1x": n_articles,
           "pool_1x": int(len(r.pool_idx)), "build_threads": build_threads, "search_threads": 1,
           "noise_std": sc["noise_std"], "scales": {}}

    timeline_1x_mean = 0.0   # the part of Phase 3's "features" stage re-measured here as click counts
    for scale in (1, factor):
        print(f"[{dataset}] scale {scale}x: building structures", flush=True)
        s: dict = {"articles": n_articles * scale}
        pool = r.pool_idx if scale == 1 else replicate_index(r.pool_idx, n_articles, scale)
        s["pool_articles"] = int(len(pool))

        # ---- FAISS (child process without PyTorch) ---------------------------------------
        proc = subprocess.run([sys.executable, __file__, "--config", f"config/{dataset}.yaml",
                               "--faiss-child", str(handoff), "--scale", str(scale)],
                              capture_output=True, text=True, cwd=REPO_ROOT)
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
        if proc.returncode != 0 or not lines:
            raise RuntimeError(f"FAISS child failed (exit {proc.returncode}): {proc.stderr[-800:]}")
        child = json.loads(lines[-1])
        faiss_times = child.pop("faiss_pool_times_ms")
        s.update(child)

        # ---- BM25 --------------------------------------------------------------------
        weights = bm.weights if scale == 1 else sparse.vstack([bm.weights] * scale, format="csr")
        pooled_t = weights[pool].T.tocsr()
        served = copy.copy(bm)
        served.weights, served.weights_t = weights, pooled_t
        k_bm25 = r.k_bm25

        def bm25_serving(q, served=served, pool=pool):
            local = served.retrieve(served.query_matrix([token_lists[q]]), k=k_bm25)[0]
            return pool[local]

        bm25_times = time_each(bm25_serving, queries, warm)
        s["bm25_serving"] = {"latency_ms": pct(bm25_times),
                             "mb_weights": round(csr_bytes(weights) / MB, 1),   # BM25Index also keeps a transposed copy of the same size
                             "mb_pooled_cache": round(csr_bytes(pooled_t) / MB, 1)}

        as_is = copy.copy(bm)
        as_is.weights = weights

        def bm25_as_is(q, as_is=as_is, pool=pool):
            return as_is.retrieve(as_is.query_matrix([token_lists[q]]), k=k_bm25, pool=pool)[0]

        n_as_is = sc["as_is_queries"]
        bm25_as_is_times = time_each(bm25_as_is, queries[:n_as_is], 3)
        s["bm25_as_is"] = {"latency_ms": pct(bm25_as_is_times), "n_queries": n_as_is,
                           "note": "matrix re-cut + scoring only; token list pre-built"}

        # ---- as_is per-request work that scans every article --------------------------------
        # build_queries rebuilds an article -> tokens dict over the whole catalogue per call, and
        # PopularityRanker.top_k(allowed=...) scans every ranked article in Python.
        from src.retrieval.bm25 import build_queries
        from src.retrieval.popularity import PopularityRanker
        if scale == 1:
            arts = ctx.articles
        else:
            arts = pd.concat([ctx.articles.assign(article_id=ctx.articles["article_id"] + (f"#{k}" if k else ""))
                              for k in range(scale)], ignore_index=True)
        ranker = PopularityRanker(arts)
        allowed = set(arts["article_id"].to_numpy()[pool])
        one_user = [pd.DataFrame({"user_id": [users[q]], "clicked_ids": [pipe.clicks_of.get(users[q], [])]})
                    for q in queries[:n_as_is]]
        build_q_times = time_each(lambda q: build_queries(one_user[q], arts), queries[:n_as_is], 3)
        pop_scan_times = time_each(lambda q: ranker.top_k(r.k_pop, allowed=allowed), queries[:n_as_is], 3)
        s["as_is_build_queries"] = {"latency_ms": pct(build_q_times), "n_queries": n_as_is}
        s["as_is_popularity_scan"] = {"latency_ms": pct(pop_scan_times), "n_queries": n_as_is}
        del arts, ranker, allowed
        del weights, pooled_t, served, as_is
        gc.collect()

        # ---- stage 2: NRMS scoring against a scale-x article-vector table ---------------
        base = pipe.news_vecs
        table = base if scale == 1 else base.repeat(scale, 1)
        n_rows = table.shape[0]
        k_total = r.k_bm25 + r.k_semantic + r.k_pop
        cand_rows = torch.from_numpy(rng.integers(1, n_rows, size=(n_q, min(200, k_total))))
        hist_rows = torch.from_numpy(rng.integers(1, n_rows, size=(n_q, 20)))
        feats = torch.from_numpy(rng.random((n_q, cand_rows.shape[1], pipe.model.n_signals)).astype(np.float32)) \
            if pipe.model.n_signals else None

        def nrms_score(q, table=table):
            h = hist_rows[q]
            user = pipe.model.user_encoder(table[h][None], (h != 0)[None])
            content = (table[cand_rows[q]] @ user[0])[None]
            return pipe.model.combine(content, user, feats[q:q + 1] if feats is not None else None)

        nrms_times = time_each(nrms_score, queries, warm)
        s["nrms_scoring"] = {"latency_ms": pct(nrms_times),
                             "mb_article_vectors": round(table.numel() * table.element_size() / MB, 1)}
        del table

        # ---- click timeline ----------------------------------------------------------------
        timeline_times = [0.0] * n_q
        if pipe.timeline is not None:
            tl = copy.copy(pipe.timeline)
            tl.keys = pipe.timeline.keys if scale == 1 else np.repeat(pipe.timeline.keys, scale)
            rows200 = rng.integers(1, len(pipe.news.article_ids), size=(n_q, 200))

            def counts(q, tl=tl):
                tq = np.full(200, np.datetime64(stamps[q], "us"))
                for w in (1.0, 24.0, 168.0):
                    tl.counts_before(rows200[q], tq, w)

            timeline_times = time_each(counts, queries, warm)
            if scale == 1:
                timeline_1x_mean = float(np.mean(timeline_times))
            s["click_timeline"] = {"latency_ms": pct(timeline_times), "mb": round(tl.keys.nbytes / MB, 2),
                                   "clicks": int(len(tl.keys))}

        # ---- paired end-to-end estimate -----------------------------------------------------
        const = lat3["modes"]["serving"]["stages_ms"]
        const_as_is = lat3["modes"]["as_is"]["stages_ms"]
        # Stages that do not depend on catalogue size, from Phase 3; "features" minus its
        # click-count part, which is measured above at this scale (same subtraction at both scales).
        constant_ms = sum(const[k]["mean"] for k in ("profile", "popularity", "merge", "sort")) + \
            max(const["features"]["mean"] - timeline_1x_mean, 0.0)
        s["constant_stages_mean_ms"] = round(constant_ms, 4)
        a = slice(0, n_as_is)
        as_is_const = sum(const_as_is[k]["mean"] for k in ("profile", "merge", "sort")) + \
            max(const_as_is["features"]["mean"] - timeline_1x_mean, 0.0)
        s["_raw_total_as_is"] = np.asarray(build_q_times) + np.asarray(bm25_as_is_times) + \
            np.asarray(pop_scan_times) + np.asarray(faiss_times[a]) + np.asarray(nrms_times[a]) + \
            np.asarray(timeline_times[a]) + as_is_const
        s["_raw_total"] = np.asarray(bm25_times) + np.asarray(faiss_times) + np.asarray(nrms_times) + \
            np.asarray(timeline_times) + constant_ms
        out["scales"][str(scale)] = s
        gc.collect()

    # ---- calibrate to Phase 3's measured end-to-end latency ------------------------------------
    # The component timings above pre-build each user's query vector and token list, work
    # that Phase 3's end-to-end requests include and that does not depend on catalogue size.
    # Add the measured 1x gap as a constant at both scales, so 1x reproduces Phase 3's mean.
    phase3_mean = lat3["modes"]["serving"]["total_ms"]["mean"]
    gap = max(phase3_mean - float(out["scales"]["1"]["_raw_total"].mean()), 0.0)
    out["calibration"] = {"phase3_serving_mean_ms": phase3_mean,
                          "component_sum_1x_mean_ms": round(float(out["scales"]["1"]["_raw_total"].mean()), 4),
                          "catalogue_independent_overhead_ms": round(gap, 4),
                          "note": "per-request query-vector/token-list building and response assembly, "
                                  "measured end-to-end in Phase 3, added at both scales"}
    phase3_as_is = lat3["modes"]["as_is"]["total_ms"]["mean"]
    gap_as_is = max(phase3_as_is - float(out["scales"]["1"]["_raw_total_as_is"].mean()), 0.0)
    out["calibration"]["as_is"] = {"phase3_as_is_mean_ms": phase3_as_is,
                                   "component_sum_1x_mean_ms": round(float(out["scales"]["1"]["_raw_total_as_is"].mean()), 4),
                                   "catalogue_independent_overhead_ms": round(gap_as_is, 4)}
    sla = conf["cost"]["sla_p99_ms"]
    for key, sdict in out["scales"].items():
        total = sdict.pop("_raw_total") + gap
        sdict["serving_total_estimate_ms"] = pct(total)
        sdict["serving_total_per_request_ms"] = [round(x, 4) for x in total]
        total_as_is = sdict.pop("_raw_total_as_is") + gap_as_is
        sdict["as_is_total_estimate_ms"] = pct(total_as_is)
        sdict["as_is_meets_sla_single_request"] = bool(np.percentile(total_as_is, 99) < sla)

    # ---- memory at 10x: measured structures + linear extrapolation of the rest ----------------
    one, ten = out["scales"]["1"], out["scales"][str(factor)]
    comp = mem2["components"]
    measured_10x = {
        "a1_article_embeddings": comp["stage1/a1_article_embeddings"]["mb"] * factor,
        "faiss_flat_index (pool)": ten["faiss_flat_pool"]["mb"],
        "bm25_weights x2": 2 * ten["bm25_serving"]["mb_weights"],
        "bm25_pooled_cache": ten["bm25_serving"]["mb_pooled_cache"],
        "article_vectors": ten["nrms_scoring"]["mb_article_vectors"],
        "click_timeline (10x clicks)": ten.get("click_timeline", {}).get("mb", 0.0),
    }
    extrapolated_10x = {
        "user_profile_store (10x users)": comp["stage1/user_profile_store"]["mb"] * factor,
        "article_tokens + tokens_by_id + row index + popularity + pool (10x articles)":
            factor * sum(comp[k]["mb"] for k in comp if k in (
                "stage1/article_tokens+ids", "cache/tokens_by_id", "stage2/nrms_row_index",
                "stage1/popularity_ranker", "stage1/pool_index", "stage1/bm25_vocabulary", "stage1/bm25_other",
                "stage2/nrms_title_token_table", "stage2/publish_times")),
        "nrms_parameters (unchanged)": comp["stage2/nrms_parameters"]["mb"],
    }
    served_10x = sum(measured_10x.values()) + sum(extrapolated_10x.values())
    runtime = mem2["process"]["rss_after_all_imports_mb"]
    ram_per_vcpu = conf["cost"]["ram_gb_per_vcpu"]
    out["memory_10x"] = {
        "served_1x_mb": mem2["summary"]["served_total_mb"],
        "measured_mb": {k: round(v, 1) for k, v in measured_10x.items()},
        "extrapolated_linear_mb": {k: round(v, 1) for k, v in extrapolated_10x.items()},
        "served_10x_mb": round(served_10x, 1),
        "worker_ram_1x_gb": round((mem2["summary"]["served_total_mb"] + runtime) / 1000, 2),
        "worker_ram_10x_gb": round((served_10x + runtime) / 1000, 2),
        "ram_gb_per_vcpu_assumed": ram_per_vcpu,
        "fits_one_vcpu_share_at_10x": (served_10x + runtime) / 1000 <= ram_per_vcpu,
        "hnsw_catalogue_10x_mb": ten["hnsw_catalogue"]["mb"],
    }

    # ---- traffic x10 on top of catalogue x10: Phase 4 queue simulation ------------------------
    from src.serving.cost_model import cores_needed, simulate_queue
    c = conf["cost"]
    sim = c["simulation"]
    traffic = []
    for label, sc_key, qps in (("today: 1x catalogue, 1,000 QPS", "1", 1000),
                               ("10x traffic only", "1", 10000),
                               ("10x catalogue only", str(factor), 1000),
                               ("10x catalogue and 10x traffic", str(factor), 10000)):
        svc = np.asarray(out["scales"][sc_key]["serving_total_per_request_ms"])
        capacity = 1000.0 / svc.mean()
        cores = cores_needed(qps, capacity, c["target_utilisation"])
        q = simulate_queue(svc, cores, qps, sim["n_arrivals"], sim["warmup_fraction"], sim["seed"])
        worker_gb = out["memory_10x"]["worker_ram_10x_gb" if sc_key != "1" else "worker_ram_1x_gb"]
        ram_gb = cores * worker_gb
        vcpus_for_ram = math.ceil(ram_gb / ram_per_vcpu)
        billed = max(cores, vcpus_for_ram)
        traffic.append({"scenario": label, "qps": qps, "service_mean_ms": round(float(svc.mean()), 3),
                        "cores_for_cpu": cores, "worker_ram_gb": worker_gb, "ram_gb": round(ram_gb, 1),
                        "vcpus_needed_for_ram": vcpus_for_ram, "billed_vcpus": billed,
                        "binding_constraint": "RAM" if vcpus_for_ram > cores else "CPU",
                        "sim_p99_ms": round(q["p99_ms"], 3), "meets_sla": q["p99_ms"] < c["sla_p99_ms"],
                        "usd_per_hour": round(billed * c["usd_per_core_hour"], 2)})
    out["scenarios"] = traffic
    out["caveats"] = [
        "Synthetic 10x catalogue = perturbed copies of real articles: HNSW recall is measured on a clustered set.",
        "BM25 rows are copied, so vocabulary and term statistics do not grow as a real 10x catalogue's would.",
        "Users x10 is a linear memory extrapolation; profile lookup is O(1).",
        "Index builds used every core; every timed search used 1 thread.",
    ]
    return out


def print_report(res: dict) -> None:
    f = res["factor"]
    one, ten = res["scales"]["1"], res["scales"][str(f)]
    cal = res["calibration"]
    print(f"\ncalibration: component sum 1x mean {cal['component_sum_1x_mean_ms']} ms + "
          f"{cal['catalogue_independent_overhead_ms']} ms overhead = Phase 3 mean {cal['phase3_serving_mean_ms']} ms")
    print(f"[{res['dataset']}] 1x = {one['articles']:,} articles (pool {one['pool_articles']:,});"
          f" {f}x = {ten['articles']:,} (pool {ten['pool_articles']:,}); {res['n_queries']} real users")

    def row(name, a, b, unit="ms", key="p99"):
        print(f"  {name:<40} 1x {a:>9.3f}  {f}x {b:>9.3f} {unit}   ({b / a if a else float('nan'):.1f}x)")

    for name, k in (("BM25 serving p99", "bm25_serving"), ("BM25 as_is p99", "bm25_as_is"),
                    ("FAISS flat over pool p99", "faiss_flat_pool"),
                    ("FAISS flat over catalogue p99", "faiss_flat_catalogue"),
                    ("NRMS scoring p99", "nrms_scoring"), ("click counts p99", "click_timeline")):
        if k in one:
            row(name, one[k]["latency_ms"]["p99"], ten[k]["latency_ms"]["p99"])
    row("as_is build_queries p99", one["as_is_build_queries"]["latency_ms"]["p99"], ten["as_is_build_queries"]["latency_ms"]["p99"])
    row("as_is popularity scan p99", one["as_is_popularity_scan"]["latency_ms"]["p99"], ten["as_is_popularity_scan"]["latency_ms"]["p99"])
    row("as_is total (estimate) p99", one["as_is_total_estimate_ms"]["p99"], ten["as_is_total_estimate_ms"]["p99"])
    print(f"  as_is p99 under SLA: 1x {one['as_is_meets_sla_single_request']}, {f}x {ten['as_is_meets_sla_single_request']}")
    row("serving total (estimate) p50", one["serving_total_estimate_ms"]["p50"], ten["serving_total_estimate_ms"]["p50"])
    row("serving total (estimate) p99", one["serving_total_estimate_ms"]["p99"], ten["serving_total_estimate_ms"]["p99"])
    for sc_name, s in (("1x", one), (f"{f}x", ten)):
        h = s["hnsw_catalogue"]
        print(f"  HNSW {sc_name}: build {h['build_seconds']}s ({h['build_threads']} threads), {h['mb']} MB; " +
              "; ".join(f"ef {e['ef_search']}: recall {e['recall_at_k']:.3f}, p99 {e['latency_ms']['p99']:.3f} ms"
                        for e in h["ef_search"]))
    m = res["memory_10x"]
    print(f"  served memory {m['served_1x_mb']:.0f} MB -> {m['served_10x_mb']:.0f} MB; worker RAM "
          f"{m['worker_ram_1x_gb']} -> {m['worker_ram_10x_gb']} GB (fits {m['ram_gb_per_vcpu_assumed']} GB/vCPU: "
          f"{m['fits_one_vcpu_share_at_10x']})")
    for t in res["scenarios"]:
        print(f"  {t['scenario']:<32} {t['qps']:>6} QPS  mean {t['service_mean_ms']:6.2f} ms  cores {t['cores_for_cpu']:>4}"
              f"  RAM {t['ram_gb']:>6.1f} GB -> billed vCPUs {t['billed_vcpus']:>4} ({t['binding_constraint']})"
              f"  p99 {t['sim_p99_ms']:6.2f} ms  ${t['usd_per_hour']}/h")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--faiss-child", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--scale", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.faiss_child:
        print(json.dumps(faiss_child(args.faiss_child, args.scale)))
        return 0
    dataset = load_config(args.config).dataset
    res = run(dataset)
    print_report(res)
    out = REPO_ROOT / "reports" / f"q4_scale_{dataset}.json"
    out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
