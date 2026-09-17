# Assignment 2, Q4 — Implementation Report

Scope: **Q4 (serving & scale analysis)**, built on the Q3 branch (`parth/a2-work`).
All four Q4 requirements are covered with measurements:

| Q4 requirement | Where | Headline |
|---|---|---|
| 1. Index memory | Phase 2, `src/serving/measure_memory.py` | serving needs 446 MB (MIND) / 169 MB (EB-NeRD); HNSW adds 272 B/article over exact FAISS |
| 2. p99 latency, retrieval + re-ranking, one request | Phase 3, `src/serving/benchmark_latency.py` | p99 **2.24 ms** (MIND) / **3.81 ms** (EB-NeRD), 2,000 requests, 1 thread |
| 3. Cost per 1,000 queries at p99 < 100 ms | Phase 4, `src/serving/cost_model.py` | 1,000 QPS = 3 / 4 cores, ≈ $0.00003 / $0.00004 per 1k queries (assumed $0.04/core-hour) |
| 4. What breaks at 10× | Phase 5, `src/serving/scale_10x.py` (measured) | per-request whole-catalogue scans (308 ms p99), then RAM per worker (4.2 GB), then ANN recall |

- **All tables:** [`reports/q4_summary.md`](q4_summary.md)
- **Design-note section:** [`reports/design_note_q4.md`](design_note_q4.md)

Both are generated from `reports/q4_*.json` by `src/serving/q4_report.py`, so their numbers match the measurements.

---

## What was built

| Phase | What | Output |
|---|---|---|
| 0 Setup | `config/serving.yaml` holds every Q4 setting; `src/serving/__init__.py` pins PyTorch, FAISS/OpenMP and BLAS to 1 thread (defaults were FAISS 10, PyTorch 4); `check_setup.py` verifies checkpoints load and records the machine | `q4_environment.json` |
| 1 Pipeline | `ServingPipeline`: `load()` once (indexes, model, every article's NRMS vector), `handle(user, t)` per request, timed in 8 stages. Two stage-1 modes: `as_is` (existing batch functions per request) and `serving` (request-independent parts built once). | `q4_correctness_*.json` |
| 2 Memory | Exact bytes per served structure; flat vs HNSW; process RSS; research-loader residue isolated per loader in fresh processes | `q4_memory_*.json` |
| 3 Latency | 2,000 real test requests, 50 warm-up, both modes on the same requests (alternating order), per-stage and by history length, repeated twice | `q4_latency_*.json`, `q4_latency_repeatability.json` |
| 4 Cost | Cores = QPS ÷ (capacity × 60%); p99 under load from a queueing simulation over measured service times (validated against M/M/1 theory); RAM per worker; price and cloud-slowdown sensitivity | `q4_cost.json` |
| 5 10× scale | Synthetic 10× catalogue (perturbed article copies), 10× clicks; 1,000 real users replayed at 1× and 10×; HNSW recall per efSearch; RAM and cost scenarios | `q4_scale_*.json` |
| 6 Write-up | Generated summary and design-note section, `make q4` | `q4_summary.md`, `design_note_q4.md` |

## Correctness

The served path is the evaluated model: re-scoring 200 test impressions per dataset through `handle()`
matches the scores saved by `train_nrms.py` (max difference 1.2e-6 MIND, 4.2e-6 EB-NeRD; identical top-1).
Stage-1 candidates in both modes equal `UnionRetriever.retrieve` exactly.

## Findings worth knowing (affect shared code)

1. **`build_queries` is the per-request bottleneck in the A1/Q2 retrieval code.** It rebuilds an
   article → tokens dict over the whole catalogue on every call: ~20 ms mean on MIND, 271 ms p99 at 10×.
   Re-cutting the BM25 matrix costs only ~0.35 ms. Fine for batch evaluation; must be cached for serving.
   (An earlier draft of this analysis blamed the matrix re-cut; measuring each part separately corrected it.)
2. **Three OpenMP runtimes in the environment** (inside faiss, torch and scikit-learn). With PyTorch
   imported, multi-threaded FAISS aborts with `OMP: Error #15`. Single-threaded FAISS (all of Phases 1–4)
   was unaffected and verified; Phase 5 runs FAISS in a child process that never imports PyTorch.
   Avoid `KMP_DUPLICATE_LIB_OK=TRUE`: its own warning says results may be wrong.
3. **The research loaders hold far more RAM than serving needs.** E.g. EB-NeRD's word2vec loader keeps a
   25 MB result but leaves 931 MB reserved (parsing lists of floats into Python objects). The process
   peaks at 3.1 GB (MIND) / 3.8 GB (EB-NeRD) during load.
4. **Tail latency follows history length.** BM25 queries use the user's whole click history; EB-NeRD
   users with 200+ clicks have p99 3.96 ms vs 1.06 ms for 1–10 clicks. Truncating history would cut the
   tail but A1 measured it costs AUC.

## Caveats (also in the generated docs)

- **Hardware:** laptop CPU (Apple M4, 1 thread). Per-core numbers are indicative, not cloud-vCPU exact.
- **Not timed:** network, request parsing, and remote profile/click-counter stores.
- **Assumptions:** price $0.04/core-hour, 60% utilisation, 4 GB RAM per vCPU. All are parameters in `config/serving.yaml`.
- **10× limits:** the catalogue is synthetic (near-duplicate copies make HNSW recall pessimistic; BM25 vocabulary does not grow), and users ×10 is a linear memory extrapolation.

## Files

| File | Status |
|---|---|
| `config/serving.yaml` | new |
| `src/serving/__init__.py`, `check_setup.py`, `pipeline.py`, `measure_memory.py`, `benchmark_latency.py`, `cost_model.py`, `scale_10x.py`, `q4_report.py` | new |
| `tests/test_serving.py` (6, real data), `test_measure_memory.py` (5), `test_cost_model.py` (4), `test_scale_10x.py` (4) | new |
| `reports/q4_*.json`, `q4_summary.md`, `design_note_q4.md`, `a2_q4_implementation.md` | new |
| `Makefile` | `q4`, `q4-check`, `q4-memory`, `q4-latency`, `q4-cost`, `q4-scale`, `q4-report` |

## Reproduce

Needs the Q3 checkpoints and saved test scores named in `config/serving.yaml` (under `data/`, not in git).

```bash
make q4          # ~12 min on an M4; AC power, other apps closed
make q4-report   # regenerate the markdown only
.venv/bin/pytest tests/test_serving.py tests/test_measure_memory.py tests/test_cost_model.py tests/test_scale_10x.py -q
```
