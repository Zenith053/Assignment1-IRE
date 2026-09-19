# Assignment 2, Q1 + Q2 — Implementation Report

Scope: **Q1 (click-history & session features)** and **Q2 (re-ranker)** only.
Q3 (baseline reproduction/ablation), Q4 (serving benchmark) and Q5 (full harness
integration) are not started. Nothing in this pass touches `scale: demo→small`
for EB-NeRD or the schema-changing columns (`session_id`, `context_article_id`)
that a fuller build would add to `IMPRESSION_COLUMNS` — both would force a full
data rebuild, and disk is at ~95% full. Everything here reads session/dwell-time
data directly from **raw** EB-NeRD parquet at feature-build time instead, the
same pattern `harness.load_leaky_popularity` already uses for Q9.

**Nothing has been committed.** All changes are on disk, uncommitted, in this
working tree.

---

## Final Q2 results (re-run 19 Sep on EB-NeRD **small**) — supersede the tables further below

The original results below were measured on EB-NeRD **demo** and a 1,200-impression MIND
sample, before the Q3 pass switched EB-NeRD to `scale: small`. They are kept for the history of
the analysis, but these are the numbers to cite. Both datasets: 10,000 impressions per split
(train / val / test, seed 13), GBDT and MLP trained on train, early-stopped on val, reported on
test; paired bootstrap 95% CI vs the best single A1 scorer (`semantic` on both), 10,000 resamples.

**Three fixes were needed to get these numbers:**

1. **Crash fix (`src/rerank/gbdt.py`).** LightGBM training segfaulted (exit 139) whenever faiss
   or torch was loaded first: LightGBM links whichever `libomp` is already in memory, and faiss
   and torch each bundle an incompatible copy. Reproduced in isolation: 4 threads exit -11,
   1 thread OK. Now `num_threads: 1` for training and prediction (also makes runs reproducible).
2. **Crash fix (`src/rerank/retriever.py`).** Multi-threaded FAISS search segfaulted once the MLP
   had loaded torch (the same clash Q4's `scale_10x.py` works around). `UnionRetriever` now sets
   `faiss.omp_set_num_threads(1)`.
3. **Metric fix (`src/rerank/evaluate_reranker.py`, found in the Q5 pass).** Universe B's
   "conditional on retrieval" figures filtered on `is not None`, but `M.ndcg`/`M.mrr` return
   0.0 for impressions without a retrieved click, so nothing was filtered and `end_to_end`
   applied recall twice. Impressions with a retrieved click are now selected explicitly, and
   Universe B reports all four metrics before/after with CIs and a paired CI (Q2.4).

### Universe A — re-rank the inview list (test split, 10,000 impressions)

| scorer | MIND AUC | MIND nDCG@10 | EB-NeRD small AUC | EB-NeRD small nDCG@10 |
|---|---|---|---|---|
| popularity (A1, train clicks) | 0.4955 | 0.2836 | 0.4441 | 0.3839 |
| BM25 (A1) | 0.5702 | 0.3445 | 0.5204 | 0.4450 |
| semantic (A1, before) | **0.6443** | **0.4033** | 0.5296 | 0.4538 |
| **GBDT (after)** | 0.6220 [0.6164, 0.6276] | 0.3865 | **0.7499** [0.7446, 0.7551] | **0.6208** |
| MLP (after) | 0.5907 | 0.3524 | 0.6065 | 0.4946 |

| paired Δ vs semantic, 95% CI | MIND | EB-NeRD small |
|---|---|---|
| GBDT AUC | −0.0223 [−0.0261, −0.0184] ✓ | **+0.2203** [+0.2126, +0.2283] ✓ |
| GBDT MRR | −0.0127 [−0.0160, −0.0093] ✓ | +0.1856 [+0.1779, +0.1932] ✓ |
| GBDT nDCG@5 | −0.0150 [−0.0184, −0.0116] ✓ | +0.2115 [+0.2035, +0.2195] ✓ |
| GBDT nDCG@10 | −0.0168 [−0.0199, −0.0137] ✓ | +0.1670 [+0.1604, +0.1735] ✓ |
| MLP AUC | −0.0535 [−0.0607, −0.0465] ✓ | +0.0769 [+0.0690, +0.0848] ✓ |
| MLP nDCG@10 | −0.0509 [−0.0569, −0.0448] ✓ | +0.0408 [+0.0344, +0.0471] ✓ |

(✓ = CI excludes zero.) The earlier findings hold at the larger, correct scale: re-ranking over
hand-crafted behavioural features **loses to A1 semantic on MIND** (significantly, on every
metric) and **wins by a wide margin on EB-NeRD** (+0.22 AUC; demo had shown +0.18).

**Top GBDT features (gain):** MIND — `train_clicks_log1p` 0.187, `semantic` 0.150,
`popularity_rank` 0.101, `semantic_rank_in_imp` 0.079, `candidate_position_norm` 0.074.
EB-NeRD — `rolling_clicks_24h` 0.265, `freshness_hours` 0.132, `rolling_clicks_168h` 0.091,
`hour_of_day` 0.075, `category_match_rate` 0.059. On EB-NeRD the two strongest features are
exactly the signals Q3 later added to NRMS (trailing clicks, article age); `hour_of_day` fell
from 2nd (demo) to 4th, consistent with the demo caveat below.

### Universe B — retrieve top-200 (BM25 ∪ FAISS ∪ popularity), then re-rank (3,000 test impressions)

| | MIND | EB-NeRD small |
|---|---|---|
| recall@200 (a click was retrieved) | 0.2353 (706 / 3,000) | 0.0960 (288 / 3,000) |
| AUC \| retrieved, stage-1 order (before) | **0.6728** [0.6570, 0.6893] | 0.4437 [0.4127, 0.4790] |
| AUC \| retrieved, GBDT (after) | 0.4942 [0.4768, 0.5123] | **0.9169** [0.8959, 0.9353] |
| nDCG@10 \| retrieved, before → after | 0.0597 → 0.0124 | 0.0166 → 0.4093 |
| paired Δ AUC \| retrieved | −0.1786 [−0.1946, −0.1630] ✓ | +0.4732 [+0.4334, +0.5115] ✓ |
| end-to-end nDCG@10 (recall × conditional), before → after | 0.0141 → 0.0029 | 0.0016 → 0.0393 |

Recall bounds the whole system: 76% (MIND) and 90% (EB-NeRD) of test clicks are never retrieved.
On what is retrieved, the Universe-A-trained GBDT is at chance on MIND (distribution shift from
inview negatives to retrieved ones) but near-perfect on EB-NeRD, where its trailing-click and
freshness features separate the one live article from 199 retrieved candidates that are mostly
stale or never shown.

### Reproduce

```bash
.venv/bin/python src/rerank/evaluate_reranker.py --config config/mind.yaml   --sample 10000 --retrieve-sample 3000
.venv/bin/python src/rerank/evaluate_reranker.py --config config/ebnerd.yaml --sample 10000 --retrieve-sample 3000
```
Writes `reports/rerank_{mind,ebnerd}.json` (about 2 minutes per dataset on an M4). Logs of the
19 Sep runs: `logs/q2_final_{mind,ebnerd}.log`.

---

## Files touched

| File | Change |
|---|---|
| `src/rerank/__init__.py` | new package |
| `src/rerank/timeline.py` | new — `RollingPopularity` (causal trailing click counts), `SessionContext` (EB-NeRD session/dwell features from raw parquet) |
| `src/rerank/candidates.py` | new — `CandidateSet`, `from_inview`, `from_retrieval` (the two evaluation universes) |
| `src/rerank/retriever.py` | new — `UnionRetriever`, stage-1 candidate generation (BM25 ∪ FAISS ∪ popularity) |
| `src/rerank/features.py` | new — the Q1 feature table (`FEATURE_NAMES`), `build_context`, `build_frame`, `score_base_scorers` |
| `src/rerank/gbdt.py` | new — LightGBM LambdaRank re-ranker (Q2 Option A) |
| `src/rerank/mlp.py` | new — listwise-softmax MLP re-ranker (Q2 Option B) |
| `src/rerank/evaluate_reranker.py` | new — CLI orchestrating training + before/after reporting for both universes |
| `src/eval/metrics.py` | added `paired_bootstrap_ci(a, b, ...)` — paired CI on a per-impression score difference |
| `tests/test_no_leakage.py` | added `test_rolling_popularity_is_strictly_causal`, `test_session_features_are_strictly_causal` (both parametrised over MIND/EB-NeRD like the existing five) |
| `tests/test_rerank_features.py` | new — 9 synthetic-data unit tests, no built dataset required |
| `requirements.txt` | added `lightgbm==4.7.0` |

Test suite: **42 passed, 2 skipped** (skips are both MIND session-feature tests,
correctly reporting N/A since `has_session_id: false`).

---

## Q1 — the feature table

26 features, grouped by the assignment's own subsections. A feature that is
architecturally unavailable on a dataset is **NaN**, never a fabricated value —
LightGBM splits on NaN natively; the MLP gets a paired missingness indicator
(`FeatureScaler` in `mlp.py`).

| Group | Features | MIND | EB-NeRD | Gate |
|---|---|---|---|---|
| Click-history (Q1.1) | `hist_len`, `hist_len_log1p`, `hist_n_distinct_categories`, `hist_category_entropy`, `hist_decay_mass` | ✓ | ✓ | — |
| | `hours_since_last_click` | NaN | ✓ | `has_history_timestamps` |
| | `mean_hist_read_time`, `mean_hist_scroll_pct` | NaN | ✓ | `has_session_id` (read from raw `history.parquet`'s `read_time_fixed`/`scroll_percentage_fixed`) |
| Session (Q1.2) | `candidate_position`, `candidate_position_norm`, `inview_size`, `hour_of_day`, `day_of_week` | ✓ | ✓ | — |
| | `session_impression_index`, `session_clicks_so_far`, `seconds_since_session_start` | NaN | ✓ | `has_session_id` (read from raw `train_behaviors`/`val_behaviors` parquet) |
| Article (Q1.3) | `train_clicks_log1p`, `popularity_rank`, `is_head`, `rolling_clicks_24h`, `rolling_clicks_168h`, `n_tokens` | ✓ | ✓ | — |
| | `freshness_hours` | NaN | ✓ | `has_published_time` |
| Category match (Q1.3) | `category_match`, `category_match_rate` | ✓ | ✓ | — |
| Interaction | `bm25`, `semantic`, `bm25_rank_in_imp`, `semantic_rank_in_imp`, `popularity_rank_in_imp` | ✓ | ✓ | — |

Deliberately excluded from the default frame: `hybrid` (A1's combiner is
fit on `val`; the re-ranker also trains on `train`/tunes on `val`, so including
it would leak val-fit information into a training feature — the GBDT sees
`bm25`/`semantic` separately and can recover any combination on its own), and
every column in `cfg.serving_time_unavailable` (current-row `read_time`,
`scroll_percentage`, `total_pageviews`, etc.) — that is Q9's job, not built here.

### The `read_time` distinction (worth stating explicitly)

Two different things share a column name. The **current** impression's
`read_time` is the outcome of the click being predicted — correctly
quarantined. A user's **past** dwell time on articles they already clicked,
before this impression fired (`mean_hist_read_time`, from `history.parquet`'s
`read_time_fixed`), is an ordinary behavioural feature and is what this
implementation computes. Never the same column.

### Two new causal mechanisms, both tested against real data

`RollingPopularity` and `SessionContext` are new relative to A1 — they read
*other* impressions, which A1's history-snapshot pairing does not cover. Each
has a **counterfactual** test, not a proxy:

- `test_rolling_popularity_is_strictly_causal` — builds the counter twice
  (once over the whole click timeline, once over only events strictly before a
  probe timestamp `t0`) and demands `counts_before(..., t0)` agrees. A leaky
  `searchsorted` side (`"right"` instead of `"left"`) or a non-trailing window
  cannot pass this by accident. **Passes on both real datasets.**
- `test_session_features_are_strictly_causal` — asserts session-relative
  indices and elapsed time are strictly increasing in real time order within a
  session, on real EB-NeRD raw data. **Passes on EB-NeRD; correctly skips with
  an N/A reason on MIND** (`has_session_id: false`).

---

## Q2 — re-ranker

### Two evaluation universes (`candidates.py`)

- **Universe A (`inview`)** — re-rank the impression's own inview list. What
  the leaderboards and A1's harness score. Used for the headline before/after
  numbers below.
- **Universe B (`retrieved`)** — `UnionRetriever` retrieves top-K (BM25 top-80
  ∪ FAISS-semantic top-100 ∪ popularity top-20, deduped, capped at K=200) per
  user, literally satisfying Q2.1. Validated end-to-end on real MIND data (see
  below) — recall is well below 100%, so results are reported as a
  recall-ceiling decomposition rather than a single misleading AUC.

### Re-rankers

- **GBDT** (`gbdt.py`) — LightGBM, `objective: lambdarank`, optimising nDCG
  directly rather than log-loss (A1's design note already names this as the
  logistic-regression hybrid's weakness). Fit on `train`, early-stopped on
  `val` nDCG@10, reported on `test` — the same split roles A1 uses everywhere.
- **MLP** (`mlp.py`) — a small listwise-softmax neural ranker over the same
  feature frame (Option B; NRMS itself is reserved for Q3, which reads raw
  article text rather than this hand-crafted table).

### A real bug found and fixed during this pass

The first run trained a GBDT that **lost to `semantic` alone by a large,
significant margin**, and its top feature by gain was `popularity_score` (raw
`train_clicks`). This reproduces something A1 already measured and documented:
*"blending [popularity or position] into the top-5 score makes it worse"* on
MIND (`design_note.md` §2) — a raw, unbounded count gives a gradient-boosted
tree far more candidate split points than a bounded cosine similarity, a
well-known tree bias toward high-cardinality features regardless of true
signal. **Fix:** replaced the raw `popularity_score` feature with
`popularity_rank_in_imp` (in-impression rank, symmetric with the existing
`bm25_rank_in_imp`/`semantic_rank_in_imp`), and tightened GBDT regularisation
(`num_leaves` 63→31, `min_data_in_leaf` 20→30, added `lambda_l2=1.0`,
`feature_fraction` 0.8→0.7). This closed most — not all — of the gap.

**A further diagnostic ruled out a plumbing bug.** A GBDT restricted to
*only* the `semantic` feature could not fully reproduce raw `semantic`'s own
AUC (0.6308 vs 0.6455 on a 6,000-impression MIND sample) — real histogram-
binning loss on an already near-continuous, near-optimal signal, not a bug in
how labels/features are aligned. Raising `max_bin` 255→1023 was tried as the
direct fix and **made both the single-feature and full-feature models worse**
(full-feature test AUC 0.6285→0.5966) — finer bins gave the *noisier*
behavioural features more precise ways to overfit, outweighing any resolution
gained on `semantic`. Reverted; this negative result is left as a comment in
`gbdt.py` rather than silently discarded.

---

## Results (Universe A, test split)

Sampled from the currently-built data on disk (MIND-small; EB-NeRD **demo**
scale — not yet switched to `small`). Paired bootstrap CI is against the best
single A1 scorer on that dataset's test split (`semantic` won on both).

### MIND (n = 1,200 per split)

| scorer | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| semantic (before) | 0.6476 | 0.3171 | 0.3420 | 0.3999 |
| bm25 | 0.5799 | 0.2636 | 0.2831 | 0.3425 |
| popularity | 0.5011 | 0.2208 | 0.2287 | 0.2894 |
| **gbdt (after)** | 0.6301 | 0.3041 | 0.3301 | 0.3908 |
| **mlp (after)** | 0.6579 | 0.3056 | 0.3310 | 0.3947 |

Paired vs. `semantic` (95% CI, 10,000 resamples):

| model | metric | Δ | 95% CI | significant |
|---|---|---|---|---|
| gbdt | AUC | −0.0175 | [−0.0320, −0.0032] | **yes** |
| gbdt | nDCG@10 | −0.0092 | [−0.0215, +0.0031] | no |
| mlp | AUC | +0.0103 | [−0.0050, +0.0258] | no |
| mlp | nDCG@10 | −0.0052 | [−0.0223, +0.0116] | no |

A larger cross-check (n=6,000, not persisted to the JSON artifact since each
run overwrites it) gave a consistent picture: GBDT AUC 0.6285 vs semantic
0.6455 (Δ=−0.0170, CI [−0.0211,−0.0129], significant). **Re-ranking loses to
semantic-alone on MIND**, which reproduces A1's own finding for its
logistic-regression hybrid (0.6401 vs 0.6441, though that gap was *not*
significant — MIND's semantic score is unusually strong and smooth, and every
tested way of combining it with weaker features (raw popularity/position in
A1; a full behavioural feature table here) has made it worse rather than
better on this specific dataset.

Top GBDT feature importances (gain): `hour_of_day` 0.140, `semantic` 0.137,
`train_clicks_log1p` 0.129, `candidate_position_norm` 0.072, `hist_len` 0.060.

### EB-NeRD demo (n = 3,500 per split)

| scorer | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| semantic (before) | 0.5401 | 0.3434 | 0.3770 | 0.4594 |
| bm25 | 0.5186 | 0.3274 | 0.3582 | 0.4424 |
| popularity | 0.4687 | 0.2805 | 0.3140 | 0.4040 |
| **gbdt (after)** | **0.7165** | **0.4766** | **0.5392** | **0.5828** |
| **mlp (after)** | 0.6181 | 0.3758 | 0.4300 | 0.5010 |

Paired vs. `semantic`:

| model | metric | Δ | 95% CI | significant |
|---|---|---|---|---|
| gbdt | AUC | **+0.1764** | [+0.1632, +0.1895] | **yes** |
| gbdt | nDCG@10 | +0.1234 | [+0.1122, +0.1344] | **yes** |
| mlp | AUC | +0.0780 | [+0.0656, +0.0901] | **yes** |
| mlp | nDCG@10 | +0.0416 | [+0.0314, +0.0517] | **yes** |

A large, significant, and well-explained win: A1 already measured EB-NeRD's
provided word2vec embeddings as weak (+0.02 AUC over random, vs MiniLM's +0.14
on MIND through identical code), so genuine behavioural signal — real
`session_id`, real per-click dwell time, real `published_time` — has far more
room to help here than on MIND. Top GBDT importances: `rolling_clicks_24h`
0.197, `hour_of_day` 0.108, `freshness_hours` 0.105, `popularity_rank` 0.066,
`category_match_rate` 0.064 — a materially different, more behaviourally-driven
ranking than MIND's.

**Caveat, stated plainly:** this ran at EB-NeRD's small `demo` scale (a few
thousand impressions across ~8 days). `hour_of_day` ranking second by gain
over such a short window could partly reflect memorising which specific hours
were trending on specific days rather than a generalisable diurnal pattern —
worth re-checking at `scale: small` before treating the magnitude as final.

---

## Universe B (retrieve-then-rank, Q2.1) — validated on real data

Run once on MIND (n=1,200, K=200) to confirm the pipeline works end-to-end
against real data (previously only unit-tested on synthetic candidates):

```
recall@200: 0.3342  (401/1,200 impressions retrieved the clicked article)
nDCG@10 | click retrieved:  stage-1 order = 0.0215   gbdt_inview on retrieved = 0.0099
end-to-end nDCG@10:         before = 0.0072          after = 0.0033
```

Two honest findings, not swept aside:

1. **Recall bounds the whole system.** Missing ~67% of clicks before any
   re-ranker runs caps end-to-end nDCG far below the Universe-A numbers above,
   regardless of re-ranker quality — consistent with A1's own recall@50
   measurements (0.079 MIND / 0.037 EB-NeRD, circulating pool).
2. **`gbdt_inview` applied to retrieved candidates does *worse* than simply
   keeping stage-1's own merge order** (0.0099 vs 0.0215). This is the
   distribution-shift cost the code explicitly anticipates: the GBDT was
   trained against editorially-curated inview negatives (hard, plausible
   near-misses) and does not generalise to retrieved negatives (mostly
   off-topic articles the union retriever pulled in). A model trained directly
   on retrieved candidates (`gbdt_retrieved`, wired but not yet trained in this
   pass) would be the fix.

---

## Reproduce

```bash
.venv/bin/pytest tests/ -q                                    # 42 passed, 2 skipped

.venv/bin/python src/rerank/evaluate_reranker.py \
    --config config/mind.yaml   --sample 6000 --skip-retrieval   # Universe A only, faster
.venv/bin/python src/rerank/evaluate_reranker.py \
    --config config/ebnerd.yaml --sample 3500 --skip-retrieval

.venv/bin/python src/rerank/evaluate_reranker.py \
    --config config/mind.yaml --sample 1200 --retrieve-k 200     # includes Universe B
```

Each run writes `reports/rerank_<dataset>.json` (overwritten per run — the
numbers above were captured from console output at the time each ran, cross-
checked against the JSON where a run's output was still the last one on disk).

## Not done in this pass

- Q9's leaky-feature arm (features built from `cfg.serving_time_unavailable`
  columns, for comparison) — the default frame is already free of them
  (asserted by a leakage test), but the explicit with/without ablation is not
  built.
- Q3 (NRMS baseline reproduction + ablation), Q4 (serving/latency benchmark),
  Q5 (full harness integration — slices, diversity/novelty/coverage,
  Codabench submission).
- `gbdt_retrieved` (a model trained directly on retrieved candidates, rather
  than applying the inview-trained model to them).
- EB-NeRD `scale: small`, and the `session_id`/`context_article_id` schema
  additions — both would force a full data rebuild against ~3.6GB free disk.
