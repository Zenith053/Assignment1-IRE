# Design Note — Lexical & Semantic Retrieval on MIND and EB-NeRD

CS4.406 Assignment 1, Part I. All numbers are from `reports/*.json`, produced by
`make all` and reproducible with `make data && make test`.

## 1. What I built

A five-stage pipeline (`download → clean → split → feature store → retrieval →
evaluation`) that runs identically on both datasets. The load-bearing decision is
that **only `clean.py` knows a dataset name**. Two adapters emit one schema:

| Table | Form | Why |
|---|---|---|
| `articles` | wide | natural key, no repetition |
| `impressions` | nested (`inview_ids`/`clicked_ids` as lists) | the impression is the evaluation unit |
| `history` | long (one row per click) | leakage checks need individual events |

`impressions` stays nested because every metric is computed per impression; storing
it long would force a `groupby` over millions of groups to rebuild structure that
was just discarded. `explode_impressions()` materialises the long view on demand.

Everything downstream branches on **capability flags** (`has_body`,
`has_published_time`, `has_provided_embeddings`), never on the dataset name. Where
a dataset genuinely lacks a capability the harness prints `N/A` — MIND has no
`published_time`, so the freshness-restricted pool is reported as unavailable
rather than silently substituted.

## 2. Choices and alternatives

**BM25 implemented directly, not `rank_bm25`.** The index is a real inverted index
(`term → postings`) materialised as a sparse matrix of precomputed BM25 document
weights, so scoring is one sparse matrix product. `rank_bm25` scores one document
at a time in Python; at 65k documents × 17k queries that is hours, versus 54s.

**pandas throughout, not polars.** One DataFrame API for both datasets. The cost is
object-dtype list columns from parquet, confined to `src/common/io.py`.

**Exact FAISS (`IndexFlatIP`) as the reference, HNSW as the ANN.** At this corpus
size exact search is milliseconds, so HNSW is measured rather than needed — see §4.

**Per-dataset encoders.** EB-NeRD uses the shipped Danish word2vec vectors; MIND is
encoded with `all-MiniLM-L6-v2`. A single multilingual encoder would have made
cross-dataset numbers comparable, but would have discarded the provided vectors and
needed 11 minutes of GPU time on a 4 GB card for Danish text it handles worse.
**Consequence, stated plainly: semantic recall is not comparable across datasets.**
Only BM25-vs-semantic *within* a dataset, and BM25 across datasets, are fair.

## 3. The leakage bug worth reporting

EB-NeRD ships one history snapshot per split directory, each covering the 21 days
*before* that split. The validation snapshot therefore spans `05-04 → 05-25`, which
is **the entire train impression window** (`05-18 → 05-25`).

My first `clean.py` collapsed the two with `drop_duplicates(keep="last")`, keeping
validation history for every user. Train impressions could then see clicks that
happened during and after them — a straight future-click leak. Both snapshots are
now kept and tagged, and `split.py` pairs each split with the newest snapshot that
*ends before it begins*:

| split | history ends | split starts | gap |
|---|---|---|---|
| train | 05-18 06:59:51 | 05-18 07:00:03 | +12 s |
| val | 05-18 06:59:51 | 05-24 00:00:29 | +5 d |
| test | 05-25 06:59:54 | 05-25 07:00:15 | +21 s |

The old behaviour gave train a **negative 7-day gap**.
`tests/test_no_leakage.py::test_history_snapshot_predates_split` fails if it regresses.

Two other data facts that changed the code: MIND numbers impressions from 1 in
*both* files, so all 73,152 dev ids collide with train ids (the schema carries a
unique `impression_id` and the raw `source_impression_id` the submission must echo);
and 7,233 MIND articles contain a double-quote, which pandas' default
`QUOTE_MINIMAL` silently swallows.

## 4. Observations

**Ranking (test split, AUC with 95% bootstrap CI, n=20,000 / 5,000):**

| scorer | MIND | EB-NeRD |
|---|---|---|
| random | 0.4988 | 0.5034 |
| popularity | 0.4955 | 0.4691 |
| bm25 | 0.5671 [0.563, 0.571] | 0.5060 [0.497, 0.515] |
| semantic | **0.6301 [0.626, 0.634]** | 0.5039 [0.495, 0.513] |
| hybrid (α=0.5) | 0.6209 | **0.5098 [0.501, 0.519]** |

**Lexical vs semantic flips between datasets.** Semantic wins decisively on MIND
(0.630 vs 0.567, non-overlapping CIs); on EB-NeRD the two are statistically
indistinguishable and BM25 is marginally ahead. The likely cause is embedding
quality, not language difficulty: MiniLM is trained for semantic similarity, while
the EB-NeRD word2vec document vectors are averaged static embeddings that blur
topical distinctions. This is the strongest argument for encoding Danish locally
with XLM-R as follow-up work.

**Popularity scores at or below random within an impression** (0.4955 MIND, 0.4691
EB-NeRD) — and 0.4751 on EB-NeRD's *val* split, adjacent to train, so news decay is
not the explanation. The cause is that the inview list is **already curated by the
production system**: globally popular articles appear in nearly every candidate list
as negatives, so popularity has no discriminative power left inside it. Candidate
generation and within-list ranking are genuinely different tasks, and a strong
candidate-generation signal can be actively harmful as a ranker.

The head/tail slice confirms this from the other side — when the clicked article *is*
a head article, popularity's nDCG@10 jumps to 0.549 (MIND) and 0.908 (EB-NeRD),
against ~0.21 and ~0.33 for the content scorers.

**Retrieval recall is low in absolute terms and pool choice dominates it.**

| dataset | pool | size | BM25 r@50 | semantic r@50 | random r@50 |
|---|---|---|---|---|---|
| MIND | all | 65,238 | 0.0062 | 0.0076 | 0.0008 |
| MIND | circulating | 4,174 | 0.0483 | **0.0751** | 0.0120 |
| EB-NeRD | all | 11,777 | 0.0095 | 0.0055 | 0.0042 |
| EB-NeRD | circulating | 2,634 | **0.0261** | 0.0216 | 0.0190 |

Only 8.2% (MIND) and 23.2% (EB-NeRD) of the catalogue is in circulation during the
test window, so full-catalogue recall understates what a serving system would do —
but the `circulating` pool is derived from the evaluation split and is therefore an
optimistic bound, not a deployable filter. Both are reported. Even at best, content
similarity to click history is a weak signal for news: recency and editorial
placement, which this pipeline does not model, carry most of the signal.

**Almost nothing carries over between train and test.** Only **3.9%** (MIND) and
**1.3%** (EB-NeRD) of test-window clicks land on an article that was clicked during
training. This is the structural fact behind several results above: any signal fitted
to train-window *items* is near-useless at test time, which is why popularity scores
below random rather than merely weakly. It also bounds what this pipeline can achieve
at all - both retrievers rank article *content*, and the content that matters is
mostly content the training window never saw. A production system closes this gap with
recency and editorial placement, neither of which is modelled here.

**Recency-weighted pooling did not help** (EB-NeRD 0.0216 → 0.0226, MIND 0.0751 →
0.0710). A negative result: exponential decay over the last 20 clicks discards
topical breadth that mean pooling keeps.

**Cold-start users score *higher* than warm users** on both datasets (MIND nDCG@10
0.409 vs 0.388). This is an artefact worth naming: cold users have shorter inview
lists, so a correct guess is likelier by chance. EB-NeRD's shortest history is 5
clicks, so the absolute "<5 clicks" definition selects nobody there — the harness
uses a within-dataset bottom-quartile band so the slice is non-empty on both.

## 5. Where it breaks at 10×

| Component | Behaviour at 10× | Fix |
|---|---|---|
| BM25 score matrix | `retrieve()` densifies a `batch × n_docs` block. At 650k docs a 256-row batch is 666 MB. | Shrink batch, or WAND/block-max to skip low-scoring postings |
| `explode_impressions` | 5.8M rows for MIND train today; 58M at 10× exceeds 14 GB RAM | Chunk by impression range, or move to a columnar out-of-core engine |
| FAISS flat index | 65k × 384 floats = 100 MB now; 1 GB at 10× still fits, but exact search is linear in corpus size | HNSW — already implemented and measured |
| HNSW recall | **Already degrading**: 0.926 on the 4,174-article pool but **0.769** on the full 65,238 corpus at `efSearch=64` | Raise `efSearch`/`M`, and re-measure — the default is not safe at scale |
| Article encoding | 11 min for 65k on 4 GB, with a CUDA OOM at batch 256 | Batch 128, fp16, or shard across runs; embeddings are cached by article-id hash |
| Per-user query cache | One query per user held in RAM; 940k users at 10× | Shard by user, or compute queries streaming |

The HNSW row is the one that already bites: ANN recall against the exact index falls
from 0.926 to 0.769 purely by growing the pool 16×, at fixed parameters. Anyone
scaling this must re-tune `efSearch` rather than trusting the default.

## 6. What I would do next, in priority order

0. **Done: top-k similarity aggregation instead of mean-pooling.** Scoring a
   candidate by the mean of its 5 highest similarities to the user's history beats
   pooling that history into one vector, because mean-pooling averages away the
   niche interest that explains the click. Measured on MIND val (k swept 1..20,
   peak at 5) and confirmed by the official scorer on MINDsmall_dev:

   | metric | mean-pool | top-5 |
   |---|---|---|
   | AUC | 0.6299 | **0.6414** |
   | MRR | 0.3041 | **0.3117** |
   | nDCG@5 | 0.3311 | **0.3400** |
   | nDCG@10 | 0.3907 | **0.3999** |

   Two features that looked promising and are not: **candidate position** scores
   0.4984/0.5016 alone, i.e. exactly random, so MIND randomises the inview order and
   the usual position prior is unavailable; and **popularity** scores 0.5001. Blending
   either into the top-5 score makes it worse, which is consistent with the 3.9%
   train-to-test item carryover reported above.

1. **Re-encode EB-NeRD with a real multilingual model.** The provided word2vec
   document vectors give +0.0005 AUC over random; MiniLM gives MIND +0.1312 through
   the identical code path. The encoder, not the language, is the bottleneck, and
   `has_provided_embeddings: false` already routes to a local encoder.
2. **Model recency explicitly.** Given 1-3% item carryover, publication age plausibly
   dominates content similarity. `published_time` and its capability flag are already
   plumbed through.
3. **Learn the hybrid weight.** At a fixed α=0.5 the blend *loses* to semantic alone on
   MIND (0.6209 vs 0.6301), so the current setting actively destroys signal. Sweep α on
   val, per dataset.
4. **Evaluate on the full split** rather than the 20,000-impression sample, and raise
   HNSW `efSearch` to recover the recall lost at full corpus size.

## 7. Validation against the official scorer

Microsoft's `evaluate.py` (msnews/MIND) was run locally against my MIND submission,
using `MINDsmall_dev` labels as `ref/truth.txt`. This is an independent check of my
metric implementations, not just my file format:

| metric | official evaluate.py | my harness | |
|---|---|---|---|
| AUC | 0.6299 | 0.6301 | agree |
| nDCG@5 | 0.3311 | 0.3317 | agree |
| nDCG@10 | 0.3907 | 0.3918 | agree |
| MRR | **0.3041** | **0.3491** | **disagreed** |

Residual differences on the three agreeing metrics are the 20,000-impression harness
sample versus the full 73,152.

**The MRR gap was a real bug in my code.** Microsoft's `mrr_score` sums `1/rank` over
*every* relevant item and divides by the number of positives; mine returned the
reciprocal rank of the *first* hit only. The two are identical on single-click
impressions and diverge on the 27.9% of MIND that is multi-click, which inflated my
MRR by 0.045. `src/eval/metrics.py` now implements the official definition, and
`tests/test_metrics.py` pins both the multi-positive case and the single-positive
case where the formulas must coincide.

After the fix, the harness reports semantic MRR **0.3043** against the official
**0.3041** - agreement within sampling noise. Correcting it moved every MIND MRR
down by 0.025-0.045, since the inflation scaled with how often a scorer put a
second positive high in the list.

This is the clearest argument in this project for validating against a reference
implementation rather than trusting one's own reading of a metric definition: the
three metrics I got right gave no hint that the fourth was wrong.

## 8. Codabench

- **MIND** (comp. 13967): the competition scores the **Official Test** phase, so the
  submission is built on `MINDlarge_test` (2,370,727 impressions, 120,961 articles,
  unlabelled) via `src/submission/predict_raw_mind.py`, semantic scorer.
  `submission_mind_test.zip` (MINDsmall_dev, 73,152 rows) is retained as the
  locally-scoreable version, since MINDlarge_test has no public labels.
  Submissions are limited to one per day, so both were format-validated locally
  first.
- **EB-NeRD** (comp. 2469): **validation-split dry run only**
  (`submission_ebnerd_val.zip`, 4,223 rows). The scored leaderboard requires
  `ebnerd_testset` (1.5 GB), deliberately not downloaded; `make ebnerd-testset`
  fetches it and the same writer produces a real submission unchanged.

*Leaderboard screenshots to be attached after upload.*
