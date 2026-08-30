# Design Note — Lexical & Semantic Retrieval on MIND and EB-NeRD

CS4.406 Assignment 1, Part I.
**Code: <https://github.com/Zenith053/Assignment1-IRE>**

All numbers are from `reports/*.json`, produced by `make all` and reproducible with
`make data && make test`.

## 1. Architecture

A five-stage pipeline (`download → clean → split → feature store → retrieval →
evaluation`) that runs identically on both datasets. The load-bearing decision is
that **only `clean.py` knows a dataset name**. Two adapters emit one schema:
`articles` (wide, one row/article), `impressions` (nested, `inview_ids`/
`clicked_ids` as lists — the impression is the evaluation unit, so storing it long
would force a `groupby` over millions of groups to rebuild structure just
discarded), and `history` (long, one row/click — leakage checks need individual
events). `explode_impressions()` materialises a long view on demand.

Everything downstream branches on **capability flags** (`has_body`,
`has_published_time`, `has_provided_embeddings`), never on the dataset name. Where
a dataset genuinely lacks a capability the harness prints `N/A` — MIND has no
`published_time`, so the freshness-restricted pool is reported unavailable rather
than silently substituted.

## 2. Choices and alternatives

**BM25 implemented directly, not `rank_bm25`.** A real inverted index
(`term → postings`) materialised as a sparse matrix of precomputed BM25 weights,
so scoring is one sparse matrix product — `rank_bm25` scores one document at a
time; at 65k docs × 17k queries that is hours, versus 54s.

**pandas throughout, not polars** — one DataFrame API for both datasets; the cost
is object-dtype list columns from parquet, confined to `src/common/io.py`.

**Exact FAISS (`IndexFlatIP`) as reference, HNSW as the ANN** — at this corpus size
exact search is milliseconds, so HNSW is measured rather than needed (§5).

**Top-5 similarity pooling, not mean-pooling.** A candidate is scored by the mean
of its 5 highest similarities to individual history clicks, rather than by cosine to
a single mean-pooled history vector — which averages away the niche interest that
explains the click. k was swept 1..20 on MIND val (peak at 5, 0.6449 vs 0.6305 for
mean-pool) and confirmed on the official scorer (0.6299 → 0.6414) and on the real
leaderboard at 2.37M-impression scale (0.6447 → **0.6567**, §6). Two features were
measured and **rejected**: candidate position (0.4984 alone — MIND shuffles inview
order, so the usual position prior does not exist here) and popularity (0.5001);
blending either into the top-5 score makes it worse, consistent with the 3.9% item
carryover in §4.

**Learned hybrid, not a fixed α.** A logistic regression over per-impression
min-max-normalised `(bm25, semantic)`, fit on val and applied frozen to test, shared
by the harness and both submission paths via `src/retrieval/hybrid.py`. It removes
the failure mode of a badly-chosen constant — α=0.5 lost outright to semantic alone
on MIND (0.6209 vs 0.6301) — and ties the best single scorer rather than beating it.
Justification and the learned coefficients are in §4.

**Per-dataset encoders**: EB-NeRD uses the shipped Danish word2vec vectors; MIND is
encoded with `all-MiniLM-L6-v2`. A single multilingual encoder would make
cross-dataset numbers comparable but discards the provided vectors and costs 11 GPU
minutes for Danish text it handles worse. **Consequence: semantic recall is not
comparable across datasets** — only BM25-vs-semantic *within* a dataset, and BM25
across datasets, are fair comparisons. The measured cost of that choice is large:
word2vec buys EB-NeRD **+0.02 AUC** over random where MiniLM buys MIND **+0.14**
through identical code, making a local multilingual re-encode the highest-value
change still outstanding.

## 3. Validation

Every correctness claim in this note is pinned either by a test or by an external
reference implementation.

**Behaviour-window boundary (Q9).** EB-NeRD ships one history snapshot per split
directory, each covering the 21 days *before* that split, so the validation
snapshot (`05-04→05-25`) spans the entire train window (`05-18→05-25`). Collapsing
the two would expose train impressions to later clicks, so both are kept and
tagged, and `split.py` pairs each split with the newest snapshot that **ends before
it begins**:

| split | history ends | split starts | gap |
|---|---|---|---|
| train | 05-18 06:59 | 05-18 07:00 | +12 s |
| val | 05-18 06:59 | 05-24 00:00 | +5 d |
| test | 05-25 06:59 | 05-25 07:00 | +21 s |

All gaps positive. `tests/test_no_leakage.py::test_history_snapshot_predates_split`
asserts this per split; `::test_no_impression_appears_in_two_splits` asserts the
splits partition the impressions exactly; `::test_popularity_is_fit_on_train_only`
asserts no evaluation-split clicks enter the popularity feature.

**Metrics checked against the official scorer.** Microsoft's `evaluate.py`
(vendored unmodified at `tools/evaluate_official.py`) run against the MIND
submission with `MINDsmall_dev` as ground truth — an independent check of the
metric implementations, not merely of file format:

| metric | official `evaluate.py` | this harness |
|---|---|---|
| AUC | 0.6414 | 0.6375 |
| MRR | 0.3117 | 0.3085 |
| nDCG@5 | 0.3400 | 0.3373 |
| nDCG@10 | 0.3999 | 0.3973 |

Residuals are uniformly ~0.003 and one-directional, consistent with the harness's
20,000-impression sample against the full 73,152. MRR follows the official
definition — the mean of `1/rank` over *every* click, not the first alone — which
matters because **27.9% of MIND impressions are multi-click**; the two definitions
coincide only on single-click impressions. `tests/test_metrics.py` pins both cases,
along with tie handling (tied scores must give AUC exactly 0.5) and undefined AUC
(all-positive or all-negative impressions return `None` rather than 0.5).

**Schema constraints taken from the raw data.** MIND numbers impressions from 1 in
*both* files, so all 73,152 dev ids collide with train ids: the schema carries a
unique `impression_id` alongside the raw `source_impression_id` that submissions
must echo back. 7,233 MIND articles contain a double-quote inside title or
abstract, which pandas' default `QUOTE_MINIMAL` silently swallows, so the TSV
reader uses `QUOTE_NONE` with `keep_default_na=False`.

## 4. Observations

**Ranking (test split, AUC, n=20,000/5,000). Semantic uses top-5 pooling; hybrid
is a logistic regression over (bm25, semantic) fit on val, replacing a fixed
α=0.5 blend (§2):**

| scorer | MIND | EB-NeRD |
|---|---|---|
| random | 0.4988 | 0.4986 |
| popularity | 0.4955 | 0.4685 |
| bm25 | 0.5671 | 0.5098 |
| semantic | **0.6375** | **0.5195** |
| hybrid (learned) | 0.6337 | 0.5169 |

**Why the learned hybrid ranks below semantic alone.** On the offline test splits
(MINDsmall_dev 73,152 impressions sampled to 20,000; EB-NeRD validation 20,000) it
trails the best single scorer by **+0.0038 (MIND) and +0.0026 (EB-NeRD)** — and in
both cases **the 95% bootstrap CIs overlap, so the deficit is not statistically
significant** (MIND 0.6337 [0.630, 0.638] vs 0.6375 [0.634, 0.642]). The honest
reading is that the blend ties with semantic, not that it loses to it.

Three things explain why it fails to *win*:

1. **Objective mismatch.** `LogisticRegression` minimises log-loss, which rewards
   calibrated click probabilities, whereas AUC only cares about the ordering within
   an impression. A combiner can fit the data better and rank slightly worse; that
   is exactly what happens here. A pairwise/ranking objective (LambdaRank) would be
   the right fix, and is a bigger change than reweighting.
2. **Only two features.** With just `(bm25, semantic)` there is little for a linear
   model to exploit beyond picking a weight ratio, and the better single feature
   already carries almost all the signal.
3. **Fit/apply distribution shift.** Weights are fit on val and applied frozen to a
   later time window, so any drift between the two costs accuracy.

**The learned weights are themselves a result.** They independently recover the
per-dataset picture from §4:

| dataset | coef bm25 | coef semantic | ratio |
|---|---|---|---|
| MIND | 0.767 | 1.516 | semantic **2.0×** |
| EB-NeRD | 0.316 | 0.201 | **bm25 1.6×** |

Fit only on click labels, the regression weights semantic twice as heavily as BM25
on MIND, and *inverts* that on EB-NeRD — matching the slice table, where BM25 wins
EB-NeRD's cold users and its circulating-pool recall. The blend is retained because
it removes the failure mode of a badly-chosen fixed α (α=0.5 lost outright to
semantic on MIND, 0.6209 vs 0.6301) at no significant cost, and because it
generalises without hand-tuning per dataset.

### Lexical vs. semantic, by slice (Q3.5)

**AUC, test split. `*` = 95% bootstrap CIs do not overlap.**

| slice | n (MIND) | BM25 | semantic | n (EB) | BM25 | semantic |
|---|---|---|---|---|---|---|
| all | 20,000 | 0.5671 | **0.6375** * | 20,000 | 0.5098 | **0.5195** * |
| warm users | 16,304 | 0.5721 | **0.6457** * | 18,270 | 0.5088 | **0.5206** * |
| cold users | 3,696 | 0.5449 | **0.6011** * | 1,730 | **0.5198** | 0.5081 |
| head clicks | 572 | 0.5890 | **0.6156** | 102 | 0.5099 | **0.5990** |
| tail clicks | 19,428 | 0.5664 | **0.6381** * | 19,898 | 0.5098 | **0.5191** * |

**Semantic wins on 9 of 10 slices, and the exception is informative.** On EB-NeRD's
**cold users BM25 leads (0.5198 vs 0.5081)** — the only reversal anywhere. With a
short history there are too few vectors to pool into a meaningful centroid, so
semantic degrades toward noise, while BM25 still matches literal tokens from the
handful of titles available. The same slice on MIND does *not* reverse (0.601 vs
0.545), which points at the embeddings rather than at cold-start itself: MiniLM
stays useful on thin evidence where EB-NeRD's averaged word2vec statics do not.

**Margin size separates the datasets.** MIND's gap is +0.070 AUC; EB-NeRD's is
+0.010 — seven times smaller. Both use identical code, so this is embedding
quality, not language difficulty: MiniLM is trained for semantic similarity, while
EB-NeRD's provided word2vec document vectors are averaged statics that blur topical
distinctions. This is the single strongest argument for re-encoding Danish locally
(§2). Under the earlier mean-pooling the two arms were statistically
indistinguishable on EB-NeRD with BM25 slightly ahead; top-5 pooling flipped it.

**On retrieval rather than ranking, the ordering reverses on EB-NeRD** (table
below): BM25 takes the circulating pool at r@50 0.0261 vs semantic's 0.0216, while
MIND stays semantic-first (0.0751 vs 0.0483). Retrieving from a 2,634-article pool
and re-ranking ~12 shown candidates are different problems, and the weaker Danish
embeddings lose the first while still edging the second.

**Popularity scores at or below random within an impression** (0.4955 MIND, 0.4685
EB-NeRD) because the inview list is **already popularity-curated by the production
system** — globally popular articles appear mostly as negatives, leaving popularity
no discriminative power inside a list, even though it separates head vs tail
impressions cleanly from the outside (head-click nDCG@10: 0.549 MIND / 0.885
EB-NeRD for popularity, vs ~0.21–0.39 for content scorers).

**Q9 serving-time ablation (EB-NeRD only, MIND has no such columns).**
`total_pageviews` is an article-level count aggregated over its *entire lifetime*
— future relative to any given split — so ranking with it directly is a serving-time
violation. Swapping it in for the honest train-only popularity feature: **AUC
0.4685 → 0.5960, +0.1275**, beating every legitimate scorer including hybrid
(0.5169). `load_leaky_popularity` builds this feature only when
`serving_time_unavailable` is declared, and is never used by any real scorer;
`tests/test_no_leakage.py::test_serving_time_ablation_declared_not_faked` pins
both that MIND correctly reports it unavailable and that the feature genuinely
varies. The size of the inflation is itself the finding: a feature this "good" is
a red flag, not a win.

**Recall is low and pool choice dominates it** (still mean-pooled — porting top-5
pooling here is a remaining step):

| dataset | pool | size | BM25 r@50 | semantic r@50 | random r@50 |
|---|---|---|---|---|---|
| MIND | all | 65,238 | 0.0062 | 0.0076 | 0.0008 |
| MIND | circulating | 4,174 | 0.0483 | **0.0751** | 0.0120 |
| EB-NeRD | all | 11,777 | 0.0095 | 0.0055 | 0.0042 |
| EB-NeRD | circulating | 2,634 | **0.0261** | 0.0216 | 0.0190 |

Only 8.2%/23.2% of the catalogue circulates during the test window, so
full-catalogue recall understates a serving system — but `circulating` is derived
from the eval split itself, an optimistic bound rather than a deployable filter;
both are reported. **Known gap:** `recall_at_k` still builds a mean-pooled user
vector, so this table measures a different configuration from the AUC table above;
porting top-5 pooling into the retrieval path is the clearest unfinished item.

**Almost nothing carries over between train and test** — only 3.9% (MIND) / 1.3%
(EB-NeRD) of test-window clicks land on a train-clicked article, bounding what any
content-only retriever can achieve and explaining why popularity scores below
random rather than merely weakly. It also suggests publication **age** plausibly
dominates content similarity, and recency is the main signal this pipeline does not
model; `published_time` is already plumbed through for EB-NeRD.

**Recency-weighted pooling did not help** (EB-NeRD r@50 0.0216→0.0226, MIND
0.0751→0.0710) — exponential decay over 20 clicks discards topical breadth mean
pooling keeps.

**Cold-start users score higher than warm** on both (MIND semantic nDCG@10: cold
0.410 vs warm 0.394) — an artefact of shorter inview lists making a correct guess
likelier, not better modelling. EB-NeRD's shortest history is 5 clicks, so an
absolute "<5" threshold selects nobody there; the harness uses a within-dataset
bottom-quartile band instead.

## 5. Where it breaks at 10×

| Component | Behaviour at 10× | Fix |
|---|---|---|
| BM25 score matrix | 650k docs × 256-row batch densifies to 666 MB | Shrink batch, or WAND/block-max |
| `explode_impressions` | 5.8M rows today; 58M at 10× exceeds 14 GB RAM | Chunk by impression range, or out-of-core |
| FAISS flat index | 100 MB now, 1 GB at 10×, but search is linear in corpus size | HNSW — already implemented and measured |
| HNSW recall | **Already degrading**: 0.926 on 4,174 articles, **0.769** on the full 65,238 at `efSearch=64` | Raise `efSearch`/`M`, re-measure — default unsafe at scale |
| Article encoding | 11 min for 65k on a 4 GB GPU, CUDA OOM at batch 256 | Batch 128, fp16, or shard; cached by article-id hash |
| Per-user query cache | One query/user in RAM; 940k users at 10× | Shard by user, or stream |

The HNSW row already bites: ANN recall against the exact index falls from 0.926 to
0.769 purely by growing the pool 16× at fixed parameters — raising `efSearch` and
re-measuring, alongside evaluating on the full split rather than a
20,000-impression sample, are the two outstanding items here.

## 6. Codabench

Both leaderboards were submitted to and scored.

**MIND** (comp. 13967, Official Test phase): `MINDlarge_test` (2,370,727
impressions, 120,961 articles, unlabelled), via `src/submission/predict_raw_mind.py`.

| submission | method | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|
| `submission_mind_large_test.zip` | semantic, mean-pool | 0.6447 | — | — | — |
| `submission_mind_large_test_top5.zip` | semantic, top-5 pool | **0.6567** | 0.3235 | 0.3495 | 0.4054 |

Top-5 pooling's local win reproduces on the real leaderboard at full scale: **+0.012
AUC** over mean-pooling.

<img src="MIND_submission.png" width="450" alt="MIND submission history">
<img src="MIND_leaderboard.png" width="450" alt="MIND leaderboard placement">

**EB-NeRD** (comp. 2469): scored on the real `ebnerd_testset`, not a validation dry
run — the original plan was to skip the 1.5 GB download, but the testset was
fetched via `make ebnerd-testset` and a genuine submission produced instead. Via
`src/submission/predict_ebnerd_test.py`, semantic mean-pool: **AUC 0.5149**.

<img src="EB-NERD_submission.png" width="450" alt="EB-NeRD submission history">
<img src="EB-NERD_leaderboard.png" width="450" alt="EB-NeRD leaderboard placement">
