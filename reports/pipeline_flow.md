# Pipeline Flow — Raw Data to Results

How this repo turns raw MIND / EB-NeRD files into evaluated rankings and
Codabench submissions. Traces the actual code (not just `PLAN.md`'s intent),
stage by stage, in execution order.

```
raw files (data/raw/)
   │  src/data/download.py
   ▼
unified schema (data/processed/<ds>/{articles,impressions,history}.parquet)
   │  src/data/clean.py
   ▼
temporal train/val/test split (data/processed/<ds>/{train,val,test}/, split_meta.json)
   │  src/data/split.py
   ▼
feature store (data/feature_store/<ds>/{articles,user_profiles}.parquet, stats.json)
   │  src/data/feature_store.py
   ▼
retrieval recall@K            evaluation harness              submissions
reports/recall_bm25_*.json    reports/eval_*.json             reports/submissions/*
reports/recall_semantic_*.json
   │  src/retrieval/{bm25,semantic,popularity,pool}.py
   │  src/eval/{metrics,harness}.py
   │  src/submission/generate_predictions.py (+ predict_raw_mind.py, predict_ebnerd_test.py)
   ▼
reports/design_note.md  (synthesises everything above)
```

Everything is driven by one of two YAML configs (`config/mind.yaml`,
`config/ebnerd.yaml`) and orchestrated by `Makefile` (`make data`, `make all`,
`make test`). Only `src/data/clean.py` contains dataset-specific code —
everything downstream branches on **capability flags** declared in the config
(`has_body`, `has_published_time`, `has_provided_embeddings`, …), never on the
dataset name. A capability a dataset lacks is reported as `N/A`.

---

## 1. Download — `src/data/download.py`

Standard-library-only (runs before `pip install`), driven by `config/source.json`
(a manifest of sources, not read in this pass but referenced by the module).

- Idempotent: `is_present()` checks declared marker files (or a non-empty
  `extract_dir`) and skips a source that's already on disk.
- Streams downloads in 1 MiB chunks with resume support via a `.part` file and
  HTTP `Range` requests; handles redirects by hand because HuggingFace's
  pre-signed CDN redirect rejects a forwarded `Authorization` header.
- Verifies SHA-256 against the manifest when pinned, extracts zips with
  zip-slip protection (`_safe_members`), optionally strips a single wrapper
  folder, and records provenance (url, size, sha256, timestamp) to
  `data/raw/manifest.json`.
- MIND lives in a gated HuggingFace repo — auth resolves a bearer token from
  `HF_TOKEN` or `~/.cache/huggingface/token`.
- `--group core|optional|all`, `--id`, `--dataset`, `--list`, `--force`,
  `--dry-run` select what to fetch. `ebnerd_testset` (1.5 GB, needed only for a
  scored EB-NeRD leaderboard submission) is `optional`, fetched via
  `make ebnerd-testset`.

Output: `data/raw/{mind,ebnerd}/...` plus `data/raw/manifest.json`.

## 2. Clean — `src/data/clean.py`

**The only dataset-specific module.** Two adapters, `load_mind()` and
`load_ebnerd()`, each read their native format and return three DataFrames in
one unified schema defined in `src/common/io.py`:

| Table | Form | Columns |
|---|---|---|
| `articles` | wide, one row/article | `article_id, title, abstract, body, category, subcategory, entities(list[str]), published_time` |
| `impressions` | nested, one row/impression | `impression_id, source_impression_id, user_id, timestamp, inview_ids(list), clicked_ids(list), source_split` |
| `history` | long, one row/click | `user_id, article_id, timestamp, position, snapshot` |

Key adapter decisions:

- **`article_id`/`user_id` cast to `str`** here — the one boundary where MIND
  (string ids) and EB-NeRD (int64 ids) differ; a dtype mismatch would silently
  join to empty downstream.
- **MIND**: headerless TSV read with `QUOTE_NONE` (7,233 articles contain a
  raw double-quote that the default `QUOTE_MINIMAL` would swallow) and
  `keep_default_na=False` (so an empty abstract isn't coerced to NaN and a
  literal "NA" headline isn't coerced to missing). Impressions are numbered
  from 1 independently in the train and dev files, so all 73,152 dev ids
  collide with train ids — `clean.py` assigns a fresh unique `impression_id`
  and preserves the original as `source_impression_id`, which the Codabench
  submission must echo back. The packed `N123-1 N456-0` impressions field is
  split into `inview_ids`/`clicked_ids` by `_parse_mind_impressions`. History
  is order-only (no click timestamps) — one snapshot (`snapshot="all"`) built
  from each user's first-seen history string.
- **EB-NeRD**: parquet read directly. `abstract` ← `subtitle`, `category` ←
  `category_str` (the readable label; the raw integer id is dropped),
  `entities` ← `ner_clusters`. Two history snapshots are read — one per split
  directory (`train/`, `validation/`) — and **kept separately, tagged by
  `snapshot`**, rather than merged. This matters: the validation snapshot
  covers the *entire* 21-day window before validation, which fully overlaps
  the train impression window; collapsing the two would let train impressions
  see clicks that happened during/after them (see §3, the leakage bug the
  design note documents finding this way).
- `check_referential_integrity()` reports what fraction of inview/clicked/history
  ids resolve to a known article — a near-zero number here is the standard
  symptom of an id dtype mismatch, so it's printed on every run and warned
  loudly below 50%.

Output: `data/processed/<dataset>/{articles,impressions,history}.parquet`,
validated against the schema by `write_table()` before being written.

## 3. Split — `src/data/split.py`

Strictly temporal — never random, since a random split would let a model see
clicks that happen after the impression it's scored on.

- `held_out_source` (config: `dev` for MIND, `val` for EB-NeRD) names the
  `source_split` value that becomes **test** directly — both datasets ship a
  separate held-out file, so no date arithmetic is needed for the test cut.
- The remainder is cut into train/val either at explicit day bounds (MIND:
  `val: ["2019-11-13","2019-11-14"]`) or by holding back the last `val_days`
  (EB-NeRD, default 2) of the available window.
- **`choose_history_snapshot()`** picks, per split, the newest history
  snapshot whose max timestamp is `<=` that split's start — this is the
  mechanism that prevents the EB-NeRD leak described above. For MIND (no
  history timestamps) it just returns the single snapshot.
- `assert_ordered_and_disjoint()` fails loudly if split windows overlap or run
  out of order — `train.t_max < val.t_min < ... `.

Output: `data/processed/<ds>/{train,val,test}/impressions.parquet` and
`data/processed/<ds>/split_meta.json` (per-split counts, exact `[t_min,t_max]`,
which history snapshot was paired with it, disjointness flag).

## 4. Feature store — `src/data/feature_store.py`

Builds the two artifacts every retrieval/eval module reads afterward. All
popularity/threshold statistics are **fit on train only** and reused unchanged
on val/test — the module's docstring calls out fitting on the eval split as
"the easiest way to manufacture a leaderboard score that does not survive
contact with serving."

- `build_article_features()`: joins `title`+`abstract` (configurable via
  `text.index_fields`; body is excluded even where available, so BM25 stays
  comparable across datasets), tokenizes it (`src/common/text.tokenize`,
  language-aware), counts each article's train-split clicks, ranks by that
  count (ties broken by id) and flags the top 20% of the *clicked* subset as
  `is_head` (never-clicked articles are always tail).
- `build_user_profiles()`: for each split, reads the history snapshot that
  `split.py` certified as safe, sorts by `position` (restores chronological
  order even without timestamps), and rolls it into one ordered `clicked_ids`
  list per user, with `n_clicks`, `last_click_time`, `is_cold`
  (`n_clicks < 5`), and `is_low_history` (bottom quartile *within this
  dataset* — needed because EB-NeRD's shortest history is 5 clicks, so the
  absolute cold threshold selects nobody there).

Output: `data/feature_store/<ds>/articles.parquet`,
`data/feature_store/<ds>/user_profiles.parquet`, `stats.json` (catalog size,
popularity concentration, per-split cold/warm counts).

## 5. Retrieval — `src/retrieval/{bm25,semantic,popularity,pool}.py`

### Candidate pools — `pool.py`
Shared by BM25 and semantic recall. Three pools, each an explicit, reported
choice because recall@K depends heavily on what the retriever is allowed to
return:
- `all` — the whole catalogue (pessimistic, honest).
- `circulating` — articles that appeared in ≥1 impression during the split
  (closer to a serving pool, but derived from the eval split itself, so it's
  an optimistic bound — always reported alongside `all`).
- `fresh` — articles published before the split window closes; needs
  `has_published_time` (EB-NeRD only, MIND reports `N/A`).

### BM25 — `bm25.py`
Implemented directly rather than via `rank_bm25` (too slow in Python at 65k
docs × tens of thousands of queries).

- `BM25Index._build()` tokenizes each document, builds a sparse term-frequency
  matrix, computes Robertson IDF (`+0.5` guard), and **precomputes the full
  BM25 document-weight matrix** (`k1=1.2, b=0.75`, configurable) so that
  scoring a query is one sparse matrix–vector product rather than a Python
  loop over postings.
- `retrieve(queries, k, pool)` — top-K over the whole corpus (or a restricted
  pool), densifying one batch at a time via `argpartition` (avoids a full sort
  of 65k columns). Feeds recall@K.
- `score_pairs(queries, query_idx, doc_idx)` — scores specific
  (query, document) pairs via a row-wise dot product of aligned sparse
  matrices, without building a full score matrix. This is what the eval
  harness and submission scripts actually call to score an impression's
  in-view candidates — despite `PLAN.md`/the module docstring describing a
  planned `score_inview()` method, no such method exists on `BM25Index`;
  `score_pairs()` is the mechanism that does that job in practice (see
  `src/eval/harness.py:159-162`).
- `build_queries()` concatenates the tokens of a user's last `L` clicked
  articles (default 20) into one query per user, so many impressions sharing
  a user reuse the same query.
- `recall_at_k()` measures the fraction of ground-truth clicks landing in the
  top-K, averaged per impression. Run via `main()` for K∈{50,100,200} on a
  20,000-impression sample (`--sample`), over each requested pool, alongside a
  random-retriever baseline (`k / pool_size`) for a fair floor.

Output: `reports/recall_bm25_<ds>_<split>.json`.

### Semantic — `semantic.py`
Embedding source is picked by the `has_provided_embeddings` capability flag,
not the dataset name:
- **EB-NeRD**: loads the shipped Ekstra Bladet word2vec `document_vector.parquet`,
  aligns rows to the article table, zero-fills (and excludes from the index)
  articles with no vector.
- **MIND**: encodes `title + ". " + abstract` locally with
  `sentence-transformers/all-MiniLM-L6-v2` (fp16-capable, GPU if available),
  cached to `.npy` and keyed on an exact article-id-list match so re-runs skip
  the GPU pass.

- `build_user_vectors()`: mean-pools (or, with `--recency-weighted`,
  exponentially decays by clicks-ago with a configurable half-life) the
  embeddings of a user's last `L` clicks into one L2-normalized query vector.
- Retrieval uses FAISS `IndexFlatIP` (exact inner product = cosine on
  normalized vectors) as the reference; `--ann` additionally builds an
  `IndexHNSWFlat(M=32, efSearch=64)` and reports ANN-vs-exact recall overlap
  plus search latency for both.
- Same `recall_at_k()` / pool / random-baseline protocol as BM25.

Output: `reports/recall_semantic_<ds>_<split>[_recency].json`.

**Cross-dataset caveat (stated repeatedly in the code and design note):**
because MIND and EB-NeRD use different encoders, semantic recall is only
comparable BM25-vs-semantic *within* one dataset — not across datasets.

### Popularity — `popularity.py`
Not a throwaway baseline — the design note treats it as the sanity floor every
content-based scorer must beat. `PopularityRanker` scores an article by its
train-split click count (computed once, reused everywhere); `top_k()` and
`score_articles()` are the two entry points used by the submission script and
the eval harness respectively.

## 6. Evaluation harness — `src/eval/{metrics.py,harness.py}`

### `metrics.py` — pure per-impression functions
- `auc()` — ROC AUC via the rank-sum identity with tie-averaged ranks; returns
  `None` (not 0.5) for degenerate impressions where every label is identical,
  so those don't bias the mean.
- `mrr()` — sums `1/rank` over **every** positive and divides by the positive
  count (the official MIND-scorer definition), not first-hit-only — the
  in-code comment notes first-hit-only overstated MRR by 0.045 on MIND's
  27.9%-multi-click impressions. (§7 of the design note documents this as a
  bug caught by validating against Microsoft's official `evaluate.py`.)
- `ndcg(k)` — binary-gain nDCG, ideal DCG computed from `min(n_pos, k)`.
- `intra_list_diversity()` — `1 − mean pairwise cosine similarity` of the
  top-10's embeddings.
- `category_entropy()` — Shannon entropy (bits) over top-10 categories,
  reported alongside diversity because the two can disagree (lexically varied
  but one-section).
- `novelty()` — mean `−log2 p(item)` over the top-10 using **train-split**
  popularity, `+1` Laplace-smoothed.
- `bootstrap_ci()` — vectorized percentile bootstrap (default `B=1000`) over a
  precomputed per-impression metric array, so every metric shares one resample
  draw and CIs stay comparable across metrics.

### `harness.py` — orchestration
For a config/split, evaluates five scorers together (`random`, `popularity`,
`bm25`, `semantic`, `hybrid = α·minmax(bm25) + (1−α)·minmax(semantic)`,
default `α=0.5`) so they share one sample and one bootstrap draw:

1. Builds the BM25 index and per-user query matrix, and loads/encodes
   embeddings + mean-pooled user vectors — same building blocks as §5.
2. Flattens every impression's in-view candidates into one array once
   (`flat_ids`/`flat_doc_rows`/`flat_user_rows`), then scores each scorer over
   that flat array in one vectorized pass (`index.score_pairs` for BM25,
   `np.einsum` for semantic cosine) rather than looping per impression.
3. Regroups scores back per impression and computes the hybrid blend inside
   each impression (min-max normalizing BM25/semantic scores per-list, since
   raw BM25 and cosine live on different scales).
4. Computes four **slices** per scorer: `all`, `cold_users`/`warm_users` (by
   the feature store's `is_low_history` flag), `head_clicks`/`tail_clicks`
   (whether any clicked article is in the popularity head band).
5. `summarise()` wraps every metric with a bootstrap 95% CI.

Output: `reports/eval_<ds>_<split>.json` plus a console table (per-scorer
AUC/MRR/nDCG@5/nDCG@10/coverage, and per-slice nDCG@10 comparison).

## 7. Submissions — `src/submission/`

Codabench line format: `<impression_id> [rank_1,...,rank_n]`, ranks a 1-based
permutation in the original in-view order, highest score → rank 1.

- **`generate_predictions.py`** — the standard path, for splits that went
  through `clean → split → feature_store`. `score_impressions()` scores with
  `bm25`, `semantic`, or `popularity`; `validate_submission()` checks row
  count, no duplicate impression ids, and that every rank list is a clean
  permutation *before* writing, so format bugs are caught locally rather than
  on the leaderboard. Writes `prediction_<ds>_<split>.txt` zipped as
  `submission_<ds>_<split>.zip` with `prediction.txt` at the archive root.
  Defaults to `test` for MIND and `val` for EB-NeRD (the validation-only dry
  run, since `ebnerd_testset` is optional/not downloaded by default).
- **`predict_raw_mind.py`** — inference-only path for `MINDlarge_test` (2.37M
  impressions, unlabelled, no train split, so it can't go through the normal
  pipeline). Streams `behaviors.tsv` in chunks to bound memory. Supports a
  `topk` pooling mode (score a candidate by the mean of its 5
  highest-similarity matches against the user's whole history, rather than
  mean-pooling history into one vector first) — measured on MIND val to beat
  mean-pooling (AUC 0.6414 vs 0.6299), confirmed against the official scorer.
  BLAS thread count is pinned to 1 before `numpy` import (`OMP_NUM_THREADS`
  etc.) because the top-k scorer's many tiny matmuls saturate cores on thread
  start/sync overhead rather than arithmetic.
- **`predict_ebnerd_test.py`** — the equivalent inference-only path for the
  1.5 GB `ebnerd_testset` bundle (opt-in via `make ebnerd-testset`). Note the
  RecSys 2024 scorer expects the file inside the zip named `predictions.txt`
  (plural) — different from MIND's `prediction.txt`.

Output: `reports/submissions/{prediction,submission}_*.{txt,zip}`.

## 8. Cross-checking against the official scorer — `tools/evaluate_official.py`

A vendored copy of Microsoft's `evaluate.py` (MIND competition's own AUC/
MRR/nDCG scorer), run locally against this project's own submission +
`MINDsmall_dev` labels as an independent check of the harness's metric math —
not just the file format. This is how the MRR bug in §6 was caught: three
metrics agreed with the official scorer and one (MRR) was off by 0.045,
pinning down a real first-hit-vs-sum-of-hits bug rather than a sampling
artifact.

## 9. Anti-gaming checks — `tests/`

- `test_no_leakage.py` — asserts every profile's contributing click
  timestamps predate the impression they're used for (where timestamps
  exist), asserts split windows are disjoint/ordered, and asserts popularity
  was fit on train only.
- `test_split_boundary.py` — includes `test_clicked_is_subset_of_inview`, i.e.
  every clicked id must also appear in that impression's inview list.
- `test_metrics.py` — pins the multi-positive MRR formula against the
  single-positive case where first-hit-only and sum-of-hits must coincide.

Run via `make test` (`pytest tests/ -v`).

## 10. Design note — `reports/design_note.md`

Synthesises every JSON report above into the ≤4-page writeup: architecture and
schema choices, alternatives rejected, the EB-NeRD history-snapshot leakage
bug and its fix, headline AUC/recall tables per dataset, the official-scorer
cross-check (§6/§8 above), a 10× scale-analysis table (where the dense
`|queries|×|corpus|` score path, in-RAM FAISS flat index, and per-user query
cache each break), and prioritized next steps (top-k pooling — already
implemented and measured; re-encoding EB-NeRD with a real multilingual model;
modeling recency explicitly; learning the hybrid α; evaluating the full
split).

---

## Current headline results (test split, from `reports/design_note.md`)

| | MIND | EB-NeRD demo |
|---|---|---|
| Best AUC | semantic **0.630** [0.626, 0.634] | hybrid **0.510** [0.501, 0.519] |
| BM25 AUC | 0.567 [0.563, 0.571] | 0.506 [0.497, 0.515] |
| Popularity AUC | 0.496 | 0.469 |
| Best recall@50 (circulating pool) | semantic 0.075 | bm25 0.026 |

Notable findings baked into the pipeline's design: popularity scores *at or
below random* inside an impression (candidate lists are already pre-curated,
so popularity has no discriminative power left — but strongly separates
head/tail slices from the outside); only 3.9% (MIND) / 1.3% (EB-NeRD) of
test-window clicks land on an article clicked during training, which bounds
what any content-only retriever can achieve; and recency-weighted pooling
made semantic recall *worse*, not better.

## Known gaps / honesty notes

- EB-NeRD's Codabench submission is a **validation-split dry run** — the
  scored RecSys 2024 leaderboard needs the 1.5 GB `ebnerd_testset`, fetched
  only via the opt-in `make ebnerd-testset`.
- Semantic recall is **not comparable across datasets** (different encoders);
  only BM25-vs-semantic within a dataset is a fair comparison.
- The `circulating` pool is derived from the evaluation split itself, so its
  recall numbers are an optimistic bound, not a deployable filter — always
  reported next to the honest `all`-catalogue number.
- `score_inview()`, named in `PLAN.md` and in `bm25.py`'s own module
  docstring as the method that would score an impression's candidates, was
  never implemented as a standalone method — `BM25Index.score_pairs()` covers
  that role in the harness and submission code instead.
