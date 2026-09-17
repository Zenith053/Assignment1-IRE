# Assignment 2, Q3 — Implementation Report

Scope: **Q3 (baseline reproduced, then beaten)**, built on top of the Q1/Q2
branch. Status as of 17 Sep:

| Q3 requirement | Where | Status |
|---|---|---|
| Reproduce the official baseline on both datasets | NRMS, Steps 1–2 | ✓ 3 seeds on both datasets |
| Improve with one principled change | Popularity/freshness-aware NRMS, Step 3 | ✓ 3 seeds on both datasets |
| Ablation isolating the contribution | Step 3 variants | ✓ |
| Paired bootstrap 95% CI excluding zero | Step 4 (`src/baseline/q3_tables.py`) | ✓ `reports/q3_summary.md` |

**Nothing is committed yet.** All changes are on disk in the working tree of
`parth/a2-work`. The full generated tables are in
[`reports/q3_summary.md`](q3_summary.md); this document explains how they were
produced and what they mean.

---

## TL;DR for the team

1. **EB-NeRD is now built at `scale: small`**, not demo. The EB-NeRD numbers in
   `a2_q1_q2_implementation.md` / `reports/rerank_ebnerd.json` are from demo and
   are not comparable — Q2 needs a re-run on small (command at the end).
2. **NRMS reproduced** (PyTorch port, ebnerd-benchmark hyperparameters, 3 seeds).
   Test AUC: **MIND 0.6237 ± 0.0022**, **EB-NeRD small 0.5508 ± 0.0027**.
   NRMS is *below* A1 semantic on MIND (−0.019, significant) but *above* it on
   EB-NeRD (+0.024, significant).
3. **Improvement = PP-Rec-style popularity/freshness signals blended into NRMS.**
   - **EB-NeRD: +0.175 AUC [+0.173, +0.176]**, +0.128 nDCG@10 — significant on
     every seed and every metric, spread across seeds only ±0.001.
   - **MIND: +0.0051 AUC [+0.004, +0.006]**, significant on all four metrics when
     averaged over seeds — but **1 of 3 seeds is worse than its baseline**, so
     the MIND gain is small and not robust to training randomness.
4. **Ablation (EB-NeRD):** popularity carries nearly all of the gain (+0.171);
   freshness alone is also strong (+0.093) but largely redundant with popularity
   (adds only ~+0.003 MRR/nDCG on top). Learned per-user gate ≈ plain sum.
5. **The MIND variant (plain sum, no gate) was chosen on validation AUC, not test.**
   Seed 13 of the plain-sum model looked much better on test, so two more seeds
   were trained and the choice was made by the pre-stated rule (mean val AUC over
   3 seeds). Its test advantage shrank from +0.013 to +0.005 once all seeds were in.
6. **Correction to the earlier draft:** freshness is *not* useless. Click rate vs
   article age is an inverted U (peak at 2–4 h), which a "newer is better" check
   misses entirely.

---

## Step 0 — data scale and environment

**Why:** Q2's EB-NeRD data was the *demo* bundle (20,501 train impressions,
1,533 users). A neural ranker with ~11–17M parameters overfits that; the
*small* bundle is ~10×.

**What changed:**
- `config/ebnerd.yaml`: `scale: demo` → `scale: small` (one line; every raw path
  uses `{scale}`).
- Re-ran the existing, unmodified pipeline:
  `src/data/clean.py` → `src/data/split.py` → `src/data/feature_store.py` (~12 s).
- Demo tables backed up to `data/_backup_ebnerd_demo/` (git-ignored, 31 MB).
- Created `.venv` (Python 3.12, `requirements.txt` unchanged).

| EB-NeRD | demo (before) | small (now) |
|---|---|---|
| Articles | 11,777 | 20,738 |
| Train impressions | 20,501 | 192,884 |
| Val impressions | 4,223 | 40,003 |
| Test impressions | 25,356 | 244,647 |
| Users with history | ~1,590 | ~15,000 |

Split windows: train 18–23 May, val 24–25 May, test 25 May–1 Jun;
`boundaries ordered and disjoint: OK`. Existing tests still pass on small.

---

## Step 1 — NRMS data loader (`src/baseline/news_data.py`)

Turns the processed tables into the integer arrays NRMS consumes. Reads only
`data/processed/` and `data/feature_store/`; never re-derives the split.

| Array | Shape | Content |
|---|---|---|
| `NewsTokens.tokens` | (N+1, 30) | title token ids; row 0 = padding article |
| `NewsTokens.vocab` | (V,) | compact id → xlm-roberta token id |
| `SplitTensors.user_history` | (users, 20) | last 20 clicked article rows, most recent last, left-padded with 0 |
| `SplitTensors.cand_rows` / `labels` / `offsets` | flat | inview candidates + click labels, same flatten idiom as `CandidateSet` |
| `SplitTensors.cand_features` | (C, F) | Step 3 signals, attached by the trainer when enabled |
| `TrainSamples` | (clicks, 5) | 1 click + 4 non-clicks **from the same impression**, resampled every epoch; `mask` marks padded slots; `features` travel with their candidates |

Design points:
- **Tokenizer:** `FacebookAI/xlm-roberta-base` for *both* datasets (Danish +
  English). No special tokens. Truncate to 30.
- **Compact vocabulary:** titles use 20,155 (MIND) / 12,134 (EB-NeRD) of the
  250k tokens, so only those rows of the pretrained embedding matrix are kept.
  The cache holds no pickled objects, so it loads under any numpy version.
- **Leakage:** histories come from `feature_store/<ds>/user_profiles.parquet`
  (built per split from the certified snapshot). The loader only selects the
  row for the split being built and raises if another split's rows are passed.

Data statistics (`reports/q3_data_{mind,ebnerd}.json`):

| | MIND | EB-NeRD small |
|---|---|---|
| Training rows (clicks) | 141,558 | 193,617 |
| Candidates per impression (median) | 23 | 8 |
| Title tokens (median) / truncated at 30 | 16 / 1.4% | 11 / 0.02% |
| Test users with < 20 history clicks | 59% | 16% |
| Test users with no history | 2.8% | 0% |

EB-NeRD shows ~8 candidates per impression vs 23 on MIND, so AUC is not
comparable *across* datasets.

---

## Step 2 — NRMS baseline (`src/baseline/nrms.py`, `src/baseline/train_nrms.py`)

### Architecture (Wu et al., EMNLP 2019)

```
title tokens → word embeddings (768, xlm-roberta init, trainable)
            → dropout 0.2 → multi-head self-attention (20 heads × 20 = 400)
            → dropout 0.2 → additive attention pooling (200)   = news vector (400)

last 20 news vectors → multi-head self-attention (20 × 20)
                     → additive attention pooling (200)         = user vector (400)

score = user · candidate news vector
```

Parameters: MIND 17,042,208 (15,479,808 word embeddings + 1,562,400 attention);
EB-NeRD 10,882,080 (9,319,680 + 1,562,400).

### Training
- Softmax over (1 click + 4 non-clicks), cross-entropy on the click.
- Adam, lr 1e-4, batch 32, fresh negatives each epoch (seeded `(seed, epoch)`).
- Early stopping on AUC over a fixed 20,000-impression val sample, patience 1.
  Epoch caps from the seed-13 baseline curves: MIND 3, EB-NeRD 4.
- Test: **full** test split. Inference encodes each article once, then one
  user-encoder pass + dot products.
- Artifacts: `reports/q3/<ds>_<run>.json` (hparams, per-epoch log, test metrics),
  `data/feature_store/<ds>/q3_runs/<run>_test_scores.npz` (per-candidate scores).

### Deviations from the reference implementation (state these in the report)
1. **PyTorch port** of the TensorFlow ebnerd-benchmark NRMS; hyperparameters matched.
2. **Padding is masked out of every softmax.** The reference lets padded tokens
   and padded history slots take attention weight; with 59% of MIND users
   having < 20 clicks that would average over empty slots.
3. Title only; history length 20 on both datasets (the original MIND NRMS used
   GloVe and history 50).
4. Evaluated on **our temporal split** (A1's `split.py`), not the official
   dev/test protocol — not directly comparable with published numbers.

### Baseline results (3 seeds, full test splits)

| | MIND (73,152 impressions) | EB-NeRD small (244,647) |
|---|---|---|
| Test AUC per seed (13, 14, 15) | 0.6232, 0.6217, 0.6261 | 0.5539, 0.5488, 0.5498 |
| **Test AUC mean ± std** | **0.6237 ± 0.0022** | **0.5508 ± 0.0027** |
| MRR | 0.2959 | 0.3471 |
| nDCG@5 | 0.3214 | 0.3865 |
| nDCG@10 | 0.3835 | 0.4645 |
| Best epoch | 3, 3, 3 | 4, 4, 4 |

**Against A1, rescored on the identical test impressions** (paired, 95% CI):

| | MIND | EB-NeRD |
|---|---|---|
| A1 semantic AUC | 0.6423 (matches A1's report exactly) | 0.5273 |
| NRMS − A1 semantic | **−0.0186** [−0.0209, −0.0164] | **+0.0236** [+0.0218, +0.0253] |
| NRMS − A1 BM25 | +0.0541 | +0.0296 |

On MIND, frozen MiniLM (pretrained on over a billion sentence pairs) beats NRMS
learning text understanding from 141k clicks. On EB-NeRD, A1 only had the
weaker provided word2vec vectors, and NRMS wins.

---

## Step 3 — the improvement: popularity/freshness-aware NRMS

### Motivation (why this is "principled")
1. News decays within hours: median age of a shown EB-NeRD article is < 5 h.
2. Q2's GBDT ranked `rolling_clicks_24h` as its top feature on EB-NeRD.
3. A1's popularity baseline scored *below* 0.5 because its counts were frozen
   at train time — stale by test time. Trailing counts fix exactly that.
4. NRMS is content-only and structurally cannot see either signal.
5. The design follows a published model: PP-Rec (Qi et al., 2021).

### Model
```
content = user · news                                   (NRMS, unchanged)
signal  = MLP(F → 64 → 1)(log1p clicks 1h, 24h, 168h [, log1p age hours])
g       = sigmoid(w · user + b)                         (per-user gate)
score   = (1 − g) · content + g · signal                (--gate learned)
score   = content + signal                              (--gate sum)
```
All parts train jointly with the same loss as the baseline. New layers are
created only when signals are enabled, *after* the encoders, so plain NRMS
initialises identically and old checkpoints load (tested).

### Signals (`src/baseline/candidate_signals.py`)
- **Popularity:** every click in train/val/test (never sampled), counted per
  article in `[t − window, t)` — strictly before the impression. Vectorised
  `ClickTimeline` (packed int64 keys, two `searchsorted` calls for any number of
  queries) because EB-NeRD test has 2.2M candidates.
- **Freshness (EB-NeRD only):** `log1p(max(0, impression time − published_time))`.
  0.03% of train candidates have a publish time up to 0.7 h *after* the
  impression (clock skew); clamped to 0. MIND has no publish time.

### Leakage checks
- **Equivalence test:** `ClickTimeline` counts equal
  `RollingPopularity.counts_before` exactly on real data, both datasets, all
  three windows — so Q2's counterfactual causality test covers these counts too.
- Boundary unit test: a click at exactly `t` is excluded; half a second later it
  is included.
- Negative sampling was refactored to carry features; the baseline's sampled
  negatives are **identical** to the original sampler on all 670k rows checked.

**Signals alone, no model** (EB-NeRD / MIND test, 20,000 impressions):

| Scorer | EB-NeRD AUC | MIND AUC |
|---|---|---|
| random | 0.498 | 0.501 |
| clicks, last 1h | 0.702 | 0.597 |
| **clicks, last 24h** | **0.735** | 0.581 |
| clicks, last 24h, window ending **60 min before** the impression | 0.692 | 0.563 |
| clicks, last 1h, excluding the user's own earlier clicks | 0.703 | 0.597 |
| newer article ranked higher (monotone) | 0.502 | N/A |
| article age closest to ~4 h ranked higher | 0.664 | N/A |

Lagging the counts an hour barely hurts (a same-moment leak would collapse to
~0.5); the user's own clicks touch 2.6% of EB-NeRD candidates and removing them
changes nothing. These counts are realistic at serving time (a streaming
per-article counter — relevant for Q4).

**Click rate by article age (EB-NeRD test, 30k impressions):**

| age | < 30m | 30m–1h | 1–2h | 2–4h | 4–8h | 8–24h | 1–3d | 3–30d | > 30d |
|---|---|---|---|---|---|---|---|---|---|
| share of candidates | 6.8% | 8.4% | 14.6% | 16.9% | 12.2% | 14.2% | 7.4% | 6.2% | 13.3% |
| click rate | 0.072 | 0.085 | 0.110 | **0.120** | 0.114 | 0.103 | 0.045 | 0.029 | 0.013 |

Brand-new articles have not caught on yet; old ones are stale. The signal MLP
can learn this bump; a monotone "newer is better" rule cannot.

---

## Training runs (18 total)

| Dataset | Variant | Seeds | Where |
|---|---|---|---|
| both | `nrms` | 13, 14, 15 | Mac (13, 14), Kaggle (15) |
| MIND | `nrms_pop` (gate) | 13, 14, 15 | Mac (13, 14), Kaggle (15) |
| MIND | `nrms_pop_sum` | 13, 14, 15 | Kaggle |
| EB-NeRD | `nrms_popfresh` (gate) | 13, 14, 15 | Mac (13), Kaggle (14, 15) |
| EB-NeRD | `nrms_pop`, `nrms_fresh`, `nrms_popfresh_sum` | 13 | Kaggle |

- Queue: `tools/run_q3.sh [mind|ebnerd]` (resumable; skips runs whose report
  exists). `PYTHON=python` for non-venv machines.
- Kaggle: `tools/make_kaggle_bundle.sh` packs code + ~160 MB/dataset of processed
  data and caches; `tools/kaggle_q3.ipynb` runs one worker per T4 GPU and zips
  results. Results were merged with `unzip -n` after checking every overlapping
  report was byte-identical.
- **Hardware mix:** runs were split between Apple `mps` and CUDA T4 (table above),
  so seed and hardware are partly confounded; state this in the report. Every
  run scored identical test impressions/candidates/labels (checked), so the
  paired comparisons are unaffected.
- Time: M4 (`mps`) ~32–35 min per MIND run, ~48–65 min per EB-NeRD run, GPU at
  97–98% utilisation. Kaggle T4: 8–17 min per run. Mixed precision (+3–6%) and
  larger batches (+23–29%) were measured and not adopted.

---

## Step 4 — results, ablation and significance (`src/baseline/q3_tables.py`)

`make q3-tables` (~3 min) reads every run report and saved test score, verifies
all runs share identical test arrays, computes official per-impression metrics,
rescores A1's scorers on the same impressions, and writes
`reports/q3_summary.{md,json}`.

**Choosing "the improvement":** among pre-stated candidates (gate, plain sum)
with all three seeds, the one with the highest **mean validation AUC**. Test
metrics never enter the choice.

| | gate val AUC | plain sum val AUC | selected |
|---|---|---|---|
| MIND | 0.6317 (`nrms_pop`) | **0.6348** (`nrms_pop_sum`) | plain sum |
| EB-NeRD | **0.7400** (`nrms_popfresh`) | 0.7399, 1 seed only (`nrms_popfresh_sum`) | gate |

**Significance:** per-impression paired bootstrap, 10,000 resamples, 95%
percentile CI, all comparisons in a dataset on one shared set of resamples.
`src/eval/metrics.paired_bootstrap_ci_many` was added for this: the existing
`paired_bootstrap_ci` would allocate a 10,000 × 244,647 index matrix (~19 GB)
on EB-NeRD. Its output matches `paired_bootstrap_ci` (tested).

### Headline: improved vs NRMS

| | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| **EB-NeRD** NRMS (mean of 3 seeds) | 0.5508 | 0.3471 | 0.3865 | 0.4645 |
| **EB-NeRD** `nrms_popfresh` (mean of 3 seeds) | **0.7256** | **0.4860** | **0.5486** | **0.5928** |
| Δ seed-averaged, 95% CI | **+0.1748** [+0.1733, +0.1761] | +0.1389 [+0.1375, +0.1404] | +0.1622 [+0.1607, +0.1636] | +0.1283 [+0.1270, +0.1295] |
| Δ per seed (13 / 14 / 15), AUC | +0.171 / +0.177 / +0.177, all significant | | | |
| **MIND** NRMS (mean of 3 seeds) | 0.6237 | 0.2959 | 0.3214 | 0.3835 |
| **MIND** `nrms_pop_sum` (mean of 3 seeds) | **0.6288** | **0.2972** | **0.3236** | **0.3860** |
| Δ seed-averaged, 95% CI | **+0.0051** [+0.0043, +0.0060] | +0.0013 [+0.0004, +0.0021] | +0.0022 [+0.0013, +0.0032] | +0.0025 [+0.0016, +0.0034] |
| Δ per seed (13 / 14 / 15), AUC | +0.0128 ✓ / **−0.0027 ✓ (worse)** / +0.0053 ✓ | | | |

How to read the MIND row honestly: the seed-averaged CI excludes zero on all
four metrics, but that CI only captures *which test impressions were sampled*,
not *training randomness*. Across seeds the plain-sum model's AUC spread is
±0.0088 — larger than the gain — and seed 14 is significantly worse than its
own baseline. On MIND seed 14, validation also disagreed with test (val said
plain sum beat NRMS, 0.6313 vs 0.6248; test said the opposite). The MIND gain
is real on average but small and unstable; the EB-NeRD gain is neither.

### Ablation

**EB-NeRD** (seed 13, vs NRMS 0.5539):

| variant | AUC | Δ vs NRMS | Δ vs full model (`nrms_popfresh`) |
|---|---|---|---|
| + popularity | 0.7247 | +0.1708 ✓ | AUC +0.0002 ✗; MRR −0.0030 ✓, nDCG@10 −0.0025 ✓ |
| + freshness | 0.6469 | +0.0930 ✓ | −0.0776 ✓ |
| + both, gate (full) | 0.7246 | +0.1707 ✓ | — |
| + both, plain sum | 0.7234 | +0.1695 ✓ | AUC −0.0011 ✓; MRR/nDCG ✗ |

**MIND** (no publish time, so popularity is the whole change):

| comparison | AUC | nDCG@10 |
|---|---|---|
| gate (`nrms_pop`), mean of 3 seeds − NRMS | +0.0024 [+0.0016, +0.0033] ✓ | +0.0011 ✓ |
| plain sum − gate, mean of 3 seeds | +0.0027 [+0.0021, +0.0033] ✓ | +0.0014 ✓ |

Reading:
1. **Popularity does almost all the work on EB-NeRD.** Freshness on top leaves
   AUC unchanged and adds only ~0.003 at the top of the ranking — an article in
   its 2–4 h peak is also the one collecting clicks.
2. **Freshness alone is strong (+0.093)**, via the inverted-U age curve.
3. **The per-user gate does not earn its keep.** Equal to plain sum on EB-NeRD;
   slightly *worse* than plain sum on MIND, where popularity is weak and the
   gate appears to damp it too much.
4. Models with popularity on EB-NeRD peak after epoch 1–3 (baseline: epoch 4):
   the signal path is learned almost immediately, and further epochs only
   overfit the text part.

---

## Files

| File | Change |
|---|---|
| `config/ebnerd.yaml` | `scale: demo` → `small` |
| `.gitignore` | added `logs/` |
| `Makefile` | `q3`, `q3-train`, `q3-tables` targets |
| `src/eval/metrics.py` | added `paired_bootstrap_ci_many` (existing functions untouched) |
| `src/baseline/__init__.py` | new package |
| `src/baseline/news_data.py` | new — tokens, histories, candidates, negative sampling |
| `src/baseline/nrms.py` | new — NRMS + optional signal MLP and gate, fast split scorer |
| `src/baseline/candidate_signals.py` | new — causal click counts, article age |
| `src/baseline/train_nrms.py` | new — train / early stop / test / artifacts; variant flags |
| `src/baseline/q3_tables.py` | new — Step 4 tables and CIs |
| `tools/run_q3.sh` | new — ordered, resumable training queue (dataset filter, `PYTHON`) |
| `tools/make_kaggle_bundle.sh`, `tools/kaggle_q3.ipynb` | new — run the queue on Kaggle |
| `tests/test_nrms_data.py` | new — 9 tests + 1 N/A skip |
| `tests/test_nrms_model.py` | new — 7 tests |
| `tests/test_candidate_signals.py` | new — 10 tests |
| `tests/test_metrics.py` | +2 tests for `paired_bootstrap_ci_many` |
| `reports/q3_data_{mind,ebnerd}.json` | data statistics |
| `reports/q3/*.json` | per-run results (18) |
| `reports/q3_summary.{md,json}` | Step 4 tables |

Full test suite: **70 passed, 3 skipped**.

Not in git (under `data/`, ignored): token and embedding caches, per-run test
scores and A1 rescoring cache in `data/feature_store/<ds>/q3_runs/`, and model
checkpoints (`.pt`) for the runs trained on the Mac (NRMS seeds 13–14 both
datasets, MIND `nrms_pop` seeds 13–14, EB-NeRD `nrms_popfresh` seed 13 — usable
for Q4 latency). Kaggle runs were brought back without checkpoints.

---

## Reproduce

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Step 0: EB-NeRD at small (scale: small in config/ebnerd.yaml)
for s in clean split feature_store; do .venv/bin/python src/data/$s.py --config config/ebnerd.yaml; done

# Steps 1-4: token caches, the 18 training runs (resumable), tables
make q3
# or separately:
tools/run_q3.sh            # ~11 h on an M4; tools/run_q3.sh mind|ebnerd for one dataset
make q3-tables             # ~3 min

.venv/bin/pytest tests/ -q
```

---

## Needed from Q1/Q2 side

1. **Re-run Q2 on EB-NeRD small** so the design note compares like with like:
   ```bash
   .venv/bin/python src/rerank/evaluate_reranker.py --config config/ebnerd.yaml --sample 3500 --skip-retrieval
   ```
   Useful comparison for the note: Q2's GBDT (hand-crafted features incl.
   rolling popularity) vs Q3's popularity-aware NRMS (learned text + the same
   kind of signal).

## Still to do for Q3

- Design-note text: deviations, the MIND "NRMS < semantic" finding, the
  inverted-U freshness curve, the popularity leakage checks, the gate null
  result, and the MIND seed-instability caveat.
- Commit.
