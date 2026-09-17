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
explains the click. The two are not rival methods but the endpoints of one
parameter (§4, *Pooling the history*), and the optimum is interior: re-swept over
k = 1..50 on both datasets under the current full-history configuration, k=5 buys
**+0.010 AUC on MIND and +0.037 on EB-NeRD** over mean-pooling, both intervals
clear of zero. It holds at scale — 0.6447 → **0.6567** on the MIND leaderboard over
2.37M impressions (§6) — and it is what makes carrying an untruncated history free,
since a stale click that matches nothing never enters the top 5. Two features were
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
| AUC | 0.6427 | 0.6423 |
| MRR | 0.3121 | 0.3121 |
| nDCG@5 | 0.3405 | 0.3405 |
| nDCG@10 | 0.4005 | 0.4005 |

All 73,152 dev impressions on both sides, at the same configuration (full history,
top-5 pooling). **Three of four metrics agree exactly to four decimals.** AUC differs
by 0.0004 for a structural reason: the submission format carries *ranks*, not scores,
so writing it collapses tied scores into a strict order, while the harness scores
from raw floats and averages tied ranks. 2.9% of MIND impressions contain at least
one tie; converting scores to ranks was measured to move AUC by +0.0002 on a 5,000
impression sample, which brackets the residual.

The comparison only holds because the submission writer and the harness score
identically — a constraint worth stating, since `--method semantic` previously
mean-pooled while the harness reported top-5, and the two differed by 0.0066 AUC on
this exact file. `--pooling` now governs both paths. MRR follows the official
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

**Ranking (AUC on the complete held-out split as shipped — all 73,152 MIND
`MINDsmall_dev` impressions and all 25,356 EB-NeRD `validation/` impressions, no
subsampling. Semantic uses top-5 pooling; hybrid is a logistic regression over
(bm25, semantic) fit on the full val split, replacing a fixed α=0.5 blend (§2):**

| scorer | MIND | EB-NeRD |
|---|---|---|
| random | 0.5012 [0.499, 0.503] | 0.4973 [0.493, 0.501] |
| popularity | 0.4950 [0.494, 0.496] | 0.4684 [0.467, 0.469] |
| bm25 | 0.5696 [0.567, 0.572] | 0.5242 [0.520, 0.528] |
| semantic | **0.6423** [0.640, 0.644] | 0.5319 [0.528, 0.536] |
| hybrid (learned) | 0.6388 [0.637, 0.641] | **0.5358** [0.532, 0.540] |

**Where the learned hybrid stands.** Over the complete held-out splits it divides
the two datasets: it **leads on EB-NeRD** (0.5358 vs semantic's 0.5319, +0.0039) and
trails on MIND (0.6388 vs 0.6423, −0.0035). **In both cases the 95% bootstrap CIs overlap**,
so neither the lead nor the deficit is statistically significant, and the honest
reading is that the blend ties the best single scorer on both datasets while being
the better of the two on EB-NeRD across every slice except cold users.

Two things bound how much a blend can win here:

1. **Objective mismatch.** `LogisticRegression` minimises log-loss, which rewards
   calibrated click probabilities, whereas AUC only cares about the ordering within
   an impression. A combiner can fit the data better and rank slightly worse. A
   pairwise/ranking objective (LambdaRank) would be the right fix, and is a bigger
   change than reweighting.
2. **Only two features.** With just `(bm25, semantic)` there is little for a linear
   model to exploit beyond picking a weight ratio, and on MIND the better single
   feature already carries almost all the signal — which is precisely why the blend
   helps on EB-NeRD, where the two arms are much closer (+0.008 apart, CIs
   overlapping) and genuinely complementary.

**The learned weights are themselves a result.** They independently recover the
per-dataset picture from §4:

| dataset | coef bm25 | coef semantic | ratio |
|---|---|---|---|
| MIND | 0.804 | 1.649 | semantic **2.1×** |
| EB-NeRD | 0.195 | 0.440 | semantic **2.3×** |

Fit on the full val splits — 61,894 impressions (MIND) and 4,223 (EB-NeRD).

Fit only on click labels, the regression weights semantic roughly twice as heavily
as BM25 on both datasets. Under the earlier truncated history it *inverted* on
EB-NeRD (bm25 1.6×); that inversion was an artefact of the window rather than a
property of the data — top-5 pooling over 20 clicks has little to choose from, and
over a full 154-click history it has a great deal, so the semantic arm gained
+0.0127 AUC and the combiner reweighted accordingly. Note the global weights do not
capture the cold-user slice, where BM25 still leads on EB-NeRD; a single linear
blend cannot express a per-slice preference. The blend is retained because it
removes the failure mode of a badly-chosen fixed α (α=0.5 lost outright to semantic
on MIND, 0.6209 vs 0.6301) at no significant cost, and because it generalises
without hand-tuning per dataset.

### Pooling the history: how many clicks should a candidate be matched against?

Mean-pooling and top-k pooling are one estimator with one knob, not two designs.
A candidate `c` is scored against a history `H` by the mean of its `k` largest
cosine similarities to individual clicks,

```
score(c) = (1/k) * sum of the k largest cos(c, h) over h in H
```

whose endpoints are the two familiar strategies. **k=1 is max-pooling** — score by
the single best-matching click. **Any k ≥ |H| is exactly mean-pooling**, because the
mean of a candidate's similarities to every click is its dot product with the mean
history vector, and re-normalising that vector to unit length only multiplies every
candidate in the impression by one per-user constant, which cannot reorder them.
That is an identity, not an approximation, and the sweep reproduces it bit for bit:
at k=10⁵ the AUC equals an independently computed mean-pooled-vector baseline to
every digit (MIND 0.6407864, EB-NeRD 0.5218714). So the question is not *which
pooling*, but *where on the selectivity axis to sit* — and the answer is neither end.

**Semantic AUC by k, val split, full history** (`tools/sweep_pooling_k.py` →
`reports/sweep_pooling_k_{mind,ebnerd}_val.json`; MIND n=19,502, EB-NeRD n=4,223):

| k | 1 | 2 | 3 | 4 | **5** | 6 | 8 | 10 | 15 | 20 | 30 | 50 | ≥\|H\| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MIND | 0.6395 | 0.6480 | 0.6499 | 0.6507 | **0.6508** | 0.6507 | 0.6498 | 0.6490 | 0.6474 | 0.6457 | 0.6439 | 0.6422 | 0.6408 |
| EB-NeRD | 0.5436 | 0.5499 | 0.5535 | 0.5577 | 0.5590 | 0.5600 | 0.5604 | **0.5613** | 0.5603 | 0.5586 | 0.5540 | 0.5474 | 0.5219 |

Both curves are unimodal, and both ends are worse than the middle. Paired bootstrap
on the per-impression AUC differences (1,000 resamples; pairing on the impression
cancels the shared difficulty of the list, so the CI answers "does k=5 beat this",
not "do two means differ"):

| comparison | MIND ΔAUC | EB-NeRD ΔAUC |
|---|---|---|
| k=5 − mean-pool | **+0.0100** [0.0079, 0.0123] | **+0.0371** [0.0275, 0.0475] |
| k=5 − k=1 (max-pool) | **+0.0114** [0.0087, 0.0137] | **+0.0154** [0.0099, 0.0214] |
| k=5 − k=20 | **+0.0051** [0.0032, 0.0071] | +0.0004 [−0.0046, 0.0054] |
| best k − k=5 | 0 (best k = 5) | +0.0023 [−0.0010, 0.0054] (best k = 10) |

**Each end fails for its own reason.** k=1 stakes the whole score on one nearest
click, so a single generic or near-duplicate headline decides the ranking and
nothing averages the noise out; it is the weaker end on MIND (−0.0114) despite
being the most "personalised" setting available. Mean-pooling fails the other way,
and the failure scales with history length: a user with 264 clicks has a mean vector
that has drifted toward the corpus centroid and discriminates almost nothing. The
clearest evidence is EB-NeRD's longest-history quartile, where mean-pooling scores
**0.4964 — below random** — while k=5 on the same impressions scores 0.5350.

**The right k tracks history length, which is why the two datasets peak in
different places.** MIND's median history is 20 clicks, EB-NeRD's is 264 — so k=5
consumes a quarter of a typical MIND profile but under 2% of an EB-NeRD one, and
EB-NeRD's optimum drifts upward accordingly:

| history quartile | median \|H\| | k=1 | k=5 | k=20 | mean-pool |
|---|---|---|---|---|---|
| MIND q1 (shortest) | 5 | 0.6147 | **0.6186** | 0.6163 | 0.6163 |
| MIND q2 | 14 | 0.6485 | **0.6604** | 0.6535 | 0.6535 |
| MIND q3 | 30 | 0.6526 | **0.6678** | 0.6603 | 0.6543 |
| MIND q4 (longest) | 70 | 0.6453 | **0.6606** | 0.6564 | 0.6426 |
| EB-NeRD q1 (shortest) | 54 | 0.5648 | **0.5791** | 0.5546 | 0.5354 |
| EB-NeRD q2 | 192 | 0.5529 | 0.5682 | **0.5789** | 0.5313 |
| EB-NeRD q3 | 343 | 0.5342 | 0.5536 | **0.5599** | 0.5243 |
| EB-NeRD q4 (longest) | 628 | 0.5225 | 0.5350 | **0.5413** | 0.4964 |

The band where a larger k wins is exactly the band with more history to select from.
Read down the mean-pool column instead and the other half of the story appears: on
EB-NeRD it decays monotonically as histories lengthen (0.5354 → 0.5313 → 0.5243 →
0.4964), passing below random in the longest quartile, while every top-k column
stays well above it — extra history helps only a scorer that can ignore most of it.
Note also that in MIND's shortest band
k=20 and mean-pooling are *numerically identical*: with fewer than 20 clicks
`min(k, |H|)` already takes everything, so top-k pooling degrades into mean-pooling
precisely for the cold users who need it most. That is the mechanism behind the one
BM25 reversal in the slice table below.

**k=5 is kept for both datasets** even though EB-NeRD peaks at 10, because the
difference is not significant (+0.0023, CI crosses zero) and a per-dataset k would
reintroduce exactly the hand-tuned constant that the learned hybrid removes for
α (§2). The honest reading of the sweep is that anything in 4–10 is
indistinguishable on MIND and anything in 5–20 on EB-NeRD; k=5 sits in both
plateaus. A history-adaptive rule — k proportional to √|H|, or a similarity
*threshold* rather than a fixed count — is the obvious next step and is untested.

**The cost is that there is no query vector, which is why retrieval still
mean-pools.** Mean-pooling produces one vector per user: scoring is a single dot
product per candidate and the vector can be handed straight to FAISS to search the
whole catalogue. Top-k pooling produces a *function*, not a point — it needs the
full |candidates| × |H| similarity block plus a partial sort per impression. For
re-ranking a shown inview list that is cheap — the whole sweep, thirteen values of k
over 19,502 MIND impressions, scores in about 10s, and EB-NeRD's 264-click profiles
in under 3s. For retrieval it is not indexable at all:
top-50 from the catalogue under this score means either |H| ANN queries merged per
user or a full scan, which is why the recall@K table still uses mean-pooled user
vectors and why that gap is a design consequence rather than an oversight.

**It reproduces off the val split it was tuned on.** The same choice was worth
0.6299 → 0.6414 AUC on Microsoft's official scorer against `MINDsmall_dev`, and
0.6447 → **0.6567** on the live MIND leaderboard over 2.37M unlabelled impressions
(§6) — both under the earlier truncated-history configuration, so those absolute
values predate the tables here while the ordering they establish is the same one the
sweep above recovers. Re-scored under the current configuration, the official scorer
returns **AUC 0.6427** on the full `MINDsmall_dev` (§3).

### Lexical vs. semantic, by slice (Q3.5)

**AUC, test split. `*` = 95% bootstrap CIs do not overlap.**

| slice | n (MIND) | BM25 | semantic | n (EB) | BM25 | semantic |
|---|---|---|---|---|---|---|
| all | 73,152 | 0.5696 | **0.6423** * | 25,356 | 0.5242 | **0.5319** |
| warm users | 59,784 | 0.5762 | **0.6506** * | 23,157 | 0.5249 | **0.5340** * |
| cold users | 13,368 | 0.5401 | **0.6049** * | 2,199 | **0.5173** | 0.5101 |
| head clicks | 2,108 | 0.6057 | **0.6234** | 121 | 0.5865 | **0.7143** * |
| tail clicks | 71,044 | 0.5685 | **0.6428** * | 25,235 | 0.5239 | **0.5310** |

**Semantic wins on 9 of 10 slices, and the exception is informative.** On EB-NeRD's
**cold users BM25 leads (0.5173 vs 0.5101)** — the only reversal anywhere. With a
short history there are too few vectors for top-5 pooling to select among, so
semantic degrades toward noise, while BM25 still matches literal tokens from the
handful of titles available. The same slice on MIND does *not* reverse (0.605 vs
0.540), which points at the embeddings rather than at cold-start itself: MiniLM
stays useful on thin evidence where EB-NeRD's averaged word2vec statics do not.

**On EB-NeRD the two arms are close to indistinguishable.** With the full history
BM25 gained more (+0.0150 AUC) than semantic (+0.0127), narrowing gaps that were
comfortably significant under the truncated window: on the full split only `warm
users` and `head clicks` separate at 95%, and `all`, `cold users` and `tail clicks`
have overlapping CIs. Lexical retrieval benefits more from extra history because
every additional headline adds matchable terms, whereas top-5 pooling already
ignores all but the five best-matching clicks. `head_clicks` separates them
decisively (0.5865 vs 0.7143), on only 121 impressions.

**Margin size separates the datasets.** MIND's gap is +0.073 AUC; EB-NeRD's is
+0.008 — nine times smaller. Both use identical code, so this is embedding quality,
not language difficulty: MiniLM is trained for semantic similarity, while EB-NeRD's
provided word2vec document vectors are averaged statics that blur topical
distinctions. This is the single strongest argument for re-encoding Danish locally
(§2).

**On retrieval rather than ranking, the ordering reverses on EB-NeRD** (table
below): BM25 takes the circulating pool at r@50 0.0367 vs semantic's 0.0239 — a
1.5× lead, wider than under the truncated window — while MIND stays semantic-first
(0.0790 vs 0.0474). Retrieving from a 2,634-article pool
and re-ranking ~12 shown candidates are different problems, and the weaker Danish
embeddings lose the first while still edging the second.

**Popularity scores at or below random within an impression** (0.4950 MIND, 0.4684
EB-NeRD) because the inview list is **already popularity-curated by the production
system** — globally popular articles appear mostly as negatives, leaving popularity
no discriminative power inside a list, even though it separates head vs tail
impressions cleanly from the outside (head-click nDCG@10: 0.539 MIND / 0.897
EB-NeRD for popularity, vs ~0.21–0.53 for content scorers).

**Q9 serving-time ablation (EB-NeRD only, MIND has no such columns).**
`total_pageviews` is an article-level count aggregated over its *entire lifetime*
— future relative to any given split — so ranking with it directly is a serving-time
violation. Swapping it in for the honest train-only popularity feature: **AUC
0.4684 → 0.5969, +0.1285**, beating every legitimate scorer including hybrid
(0.5358). `load_leaky_popularity` builds this feature only when
`serving_time_unavailable` is declared, and is never used by any real scorer;
`tests/test_no_leakage.py::test_serving_time_ablation_declared_not_faked` pins
both that MIND correctly reports it unavailable and that the feature genuinely
varies. The size of the inflation is itself the finding: a feature this "good" is
a red flag, not a win.

**Recall is low and pool choice dominates it** (still mean-pooled — porting top-5
pooling here is a remaining step):

| dataset | pool | size | BM25 r@50 | semantic r@50 | random r@50 |
|---|---|---|---|---|---|
| MIND | all | 65,238 | 0.0064 | **0.0081** | 0.0008 |
| MIND | circulating | 4,174 | 0.0474 | **0.0790** | 0.0120 |
| EB-NeRD | all | 11,777 | **0.0109** | 0.0062 | 0.0042 |
| EB-NeRD | circulating | 2,634 | **0.0367** | 0.0239 | 0.0190 |
| EB-NeRD | fresh | 10,932 | **0.0115** | 0.0066 | 0.0046 |

Only 8.2%/23.2% of the catalogue circulates during the test window, so
full-catalogue recall understates a serving system — but `circulating` is derived
from the eval split itself, an optimistic bound rather than a deployable filter;
both are reported. EB-NeRD's `fresh` pool (articles published before the impression)
tracks `all` closely, so the freshness restriction costs almost nothing here.
**Known gap:** `recall_at_k` still builds a mean-pooled user vector, so this table
measures a different configuration from the AUC table above; porting top-5 pooling
into the retrieval path is the clearest unfinished item, and the hardest, since the
score has no single query vector to hand FAISS (*Pooling the history*, above).

**Almost nothing carries over between train and test** — only 3.9% (MIND) / 1.3%
(EB-NeRD) of test-window clicks land on a train-clicked article, bounding what any
content-only retriever can achieve and explaining why popularity scores below
random rather than merely weakly. It also suggests publication **age** plausibly
dominates content similarity, and recency is the main signal this pipeline does not
model; `published_time` is already plumbed through for EB-NeRD.

**Recency-weighted pooling did not help** — it now hurts on both datasets
(circulating r@50: MIND 0.0790→0.0716, EB-NeRD 0.0239→0.0228). Exponential decay
with a 5-click half-life discards the topical breadth that mean pooling keeps, and
the effect is larger over a full history than it was over a 20-click window.

**Cold-start users score higher than warm** on both (MIND semantic nDCG@10: cold
0.410 vs warm 0.398) — an artefact of shorter inview lists making a correct guess
likelier, not better modelling. EB-NeRD's shortest history is 5 clicks, so an
absolute "<5" threshold selects nobody there; the harness uses a within-dataset
bottom-quartile band instead.

## 5. Where it breaks at 10×

| Component | Behaviour at 10× | Fix |
|---|---|---|
| BM25 score matrix | 650k docs × 256-row batch densifies to 666 MB | Shrink batch, or WAND/block-max |
| `explode_impressions` | 5.8M rows today; 58M at 10× exceeds 14 GB RAM | Chunk by impression range, or out-of-core |
| FAISS flat index | 100 MB now, 1 GB at 10×, but search is linear in corpus size | HNSW — already implemented and measured |
| HNSW recall | **Already degrading**: 0.928 on 4,174 articles, **0.769** on the full 65,238 at `efSearch=64` | Raise `efSearch`/`M`, re-measure — default unsafe at scale |
| Article encoding | 11 min for 65k on a 4 GB GPU, CUDA OOM at batch 256 | Batch 128, fp16, or shard; cached by article-id hash |
| Per-user query cache | One query/user in RAM; 940k users at 10× | Shard by user, or stream |

The HNSW row already bites: ANN recall against the exact index falls from 0.928 to
0.769 purely by growing the pool 16× at fixed parameters. Raising `efSearch`/`M`
and re-measuring is the outstanding item here; the ranking tables in §4 now run on
the complete held-out splits rather than a 20,000-impression sample, so sampling no
longer contributes uncertainty to them.

## 6. Codabench

Both leaderboards were submitted to and scored. **Both submissions predate the
removal of the truncation window (§2)** — MIND ran at N=50, EB-NeRD at N=20 — so the
scores below correspond to the earlier configuration, not to the §4 tables.
Regenerating and resubmitting both is the outstanding item here; the offline gains
from full history (+0.007 MIND AUC, +0.020 EB-NeRD) suggest the direction but do not
predict the leaderboard.

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
