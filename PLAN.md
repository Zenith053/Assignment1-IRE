# Assignment 1 — Part I: Lexical & Semantic Retrieval (MIND + EB-NeRD)

## Context

CS4.406 Assignment 1 (due 2026-08-27) requires a reproducible retrieval pipeline over two news
datasets, covering Q1–Q6: data pipeline, BM25 retrieval, embedding retrieval, an evaluation
harness with slices and bootstrap CIs, Codabench submissions, and a ≤4-page design note.

Current repo state: raw data is **already downloaded** (505 MB under [data/raw/](data/raw/) —
MINDsmall train+dev, ebnerd_demo, ebnerd_small, Ekstra_Bladet_word2vec). A [Makefile](Makefile),
[README.md](README.md), [requirements.txt](requirements.txt) and [.gitignore](.gitignore) exist,
but `src/` and `config/` are **empty directories** — every module the Makefile invokes has to be
written. There are no commits on `master` yet.

Grading is on pipeline correctness, system design, ablation rigour, scale analysis,and
design-note clarity — never leaderboard rank. The plan optimises for those.

### Decisions taken with the user

- **EB-NeRD Codabench**: validation-only dry run. Build a format-correct submission writer,
  validate it on the held-out split, do **not** download `ebnerd_testset.zip`. Leave the download
  behind an opt-in `make ebnerd-testset` target.
  ⚠️ This means **Q5's EB-NeRD half stays unfulfilled** (no scored leaderboard screenshot for
  competition 2469). Flipping to a real submission later is one `make` target plus ~4 GB disk.
- **Embeddings**: EB-NeRD uses the provided word2vec document vectors; MIND is encoded locally
  with a sentence-transformer on the GTX 1650 Ti.
- **Scale**: everything is config-driven; default to `ebnerd_demo`. Final-run scale (demo vs.
  small) is a one-line config change, decided later.

### Environment

Python 3.12 venv (`/usr/bin/python3.12`). System default is 3.14 with only numpy; 3.12 has the
safest wheel coverage for the whole stack. GPU is a GTX 1650 Ti (4 GB) — fine for MiniLM
inference in fp16, not for training.

---

## Verified data facts (drive the split config)

| | MIND-small | EB-NeRD demo |
|---|---|---|
| train window | 11/09–11/14/2019 | `train/` (7 days) |
| test window | dev = 11/15/2019, **labels present** | `validation/` (following 7 days) |
| articles | 51 282 / 42 416 rows (train/dev `news.tsv`) | `articles.parquet` |
| history | ordered id list, **no timestamps** | `history.parquet`, timestamped |

Confirmed parquet columns:
- `articles`: `article_id, title, subtitle, body, category, category_str, subcategory,
  published_time, last_modified_time, premium, article_type, ner_clusters, entity_groups, topics,
  sentiment_label, sentiment_score, total_inviews`
- `behaviors`: `impression_id, article_id, impression_time, read_time, scroll_percentage,
  device_type, article_ids_inview, article_ids_clicked, user_id, session_id, is_sso_user, gender,
  postcode, is_subscriber, next_read_time, next_scroll_percentage`
- `history`: `user_id, article_id_fixed, impression_time_fixed, read_time_fixed,
  scroll_percentage_fixed` (parallel lists — explode to get one row per click)
- `document_vector.parquet`: `article_id, document_vector`

---

## Layout

```
config/{mind,ebnerd}.yaml        # paths, split dates, scale switch, hyperparams
src/
  common/{config,io,text}.py     # yaml+paths; parquet/tsv IO + explode_impressions; EN+DA tokenizer
  data/{download,clean,split,feature_store}.py
  retrieval/{bm25,semantic,popularity}.py
  eval/{metrics,harness}.py
  submission/generate_predictions.py
tests/{test_no_leakage,test_split_boundary,test_metrics}.py
reports/                          # metric tables + plots for the design note
```

`requirements.txt` grows to: numpy, scipy, pandas, pyarrow, scikit-learn, faiss-cpu,
torch, sentence-transformers, pyyaml, tqdm, pytest, matplotlib.

Every module takes `--config config/<ds>.yaml` so the two datasets share one code path.

**pandas everywhere** — one DataFrame API for both datasets. A polars/pandas split would mean two
dialects for the same operations, and MIND is TSV, which is pandas-native anyway.

The one real cost: pandas reads parquet list columns (`article_ids_inview`, `article_id_fixed`,
`document_vector`) as object-dtype columns of Python lists — memory-hungry and slow to explode.
Do not fight it. For the nested-heavy files, read with `pyarrow.parquet` directly and convert
explicitly (`np.vstack` for the 300-dim document vectors), keeping pandas as the DataFrame layer
rather than the parquet reader in hot paths. All of this is confined to `src/common/io.py`.

---

## Storage format: long vs wide

Decided per table, according to how each is consumed.

| Table | Canonical form | Why |
|---|---|---|
| `articles` | **wide** — one row per article | Natural key, no repetition |
| `impressions` | **nested** — one row per impression, `inview_ids`/`clicked_ids` as list columns | The impression is the evaluation unit |
| `history` | **long** — one row per click event | Leakage checks and recency decay need individual events |

**Why `impressions` stays nested.** AUC, MRR and nDCG are all computed *per impression* over its
candidate list, and the submission format is a per-impression ranked list. Stored long, every
metric would begin with `groupby(impression_id)` over millions of groups — slow in pandas, and it
reassembles structure that was just discarded. Long form also repeats `user_id`/`timestamp` per
candidate; MIND-small averages ~37 candidates per impression, so exploding train alone is ~5.8M
rows.

Long form *is* better for feature joins, popularity counts and leakage assertions, so
`src/common/io.py` exposes `explode_impressions(df) -> (impression_id, article_id, label)` to
materialise that view on demand. Derived, never stored.

**Why `history` is long.** One row per `(user_id, article_id, timestamp, position)`. This is the
form the no-future-click assertion needs, and it drops straight out of exploding EB-NeRD's
parallel `*_fixed` lists. The nested rollup that BM25 query construction and mean-pooling actually
consume is built once into `user_profiles` in the feature store.

---

## Reconciling the two datasets

**The unified schema is a contract.** Only two adapters — `load_mind()` and `load_ebnerd()` in
`src/data/clean.py` — know anything dataset-specific; both return the same three DataFrames.
Everything downstream (split, feature store, BM25, semantic, eval, submission) is dataset-agnostic
and must never branch on dataset name.

| | MIND | EB-NeRD | Reconciliation |
|---|---|---|---|
| Format | TSV, headerless | Parquet | Adapter layer |
| `article_id` | `"N55528"` str | int64 | **Cast to str at the boundary** — mismatched dtypes join to empty silently |
| Candidates | packed `N123-1 N456-0` | separate inview/clicked columns | Parse MIND's packed field into two lists |
| History | ordered ids, **no timestamps** | parallel `*_fixed` lists, timestamped | `timestamp` nullable, `position` always present |
| Abstract | `abstract` | `subtitle` | Rename |
| Category | str | int + `category_str` | Use `category_str`; keep raw ids separately |
| Entities | JSON with `Label` | `ner_clusters` | Both → `entities: list[str]` |
| Body | **absent** in MIND-small | full `body` | Index `title + abstract` only, so BM25 is comparable across datasets |
| `published_time` | **absent** | present | Capability flag (below) |
| Clicks per impression | usually exactly 1 | often multiple | Metrics must handle multi-positive: MRR = first relevant, nDCG accumulates all gains |
| Provided embeddings | none | word2vec / BERT | Per-dataset embedding source in config |

Two differences do not reduce to renaming, and they drive real design decisions:

**Missing capabilities are declared, not faked.** Each config carries flags such as
`has_published_time: false` and `has_body: false`. Code checks the flag; the harness emits `N/A`
rather than a fabricated number. Concretely, the freshness-restricted retrieval ablation runs on
EB-NeRD only, and the report says so instead of quietly reporting an unrestricted number as if it
were restricted.

**Embeddings deliberately stay un-unified.** Danish vs English forces per-dataset encoders
(provided word2vec for EB-NeRD, MiniLM for MIND). The consequence must be stated plainly in the
design note: **semantic recall@K is not comparable across datasets** — only BM25-vs-semantic
*within* a dataset is a fair comparison. Cross-dataset claims are limited to lexical retrieval,
where `title + abstract` and BM25 are genuinely the same procedure on both sides.

---

## Q1 — Reproducible pipeline

**`download.py`** — idempotent: check for the expected unpacked marker file, skip if present, else
fetch+unzip. Existing raw data must survive a re-run untouched. Record a manifest
(`data/raw/manifest.json`: url, bytes, sha256, timestamp). Optional targets: multilingual-BERT
embeddings, `ebnerd_testset`.

**`clean.py`** — two adapters, `load_mind()` and `load_ebnerd()`, collapse both datasets into the
one schema, written as parquet under `data/processed/<ds>/`. **All `article_id`/`user_id` values
are cast to `str` here**, at the only boundary where the dtype difference exists.

- `articles` (**wide**): `article_id, title, abstract, body, category, subcategory,
  entities(list[str]), published_time, dataset`
  - MIND: `abstract` ← abstract col, `body` ← null (MIND-small ships no body), `entities` ←
    `Label` fields parsed out of the title/abstract entity JSON, `published_time` ← null.
  - EB-NeRD: `abstract` ← `subtitle`, `category` ← `category_str`, `entities` ← `ner_clusters`.
  - MIND articles = union of train + dev `news.tsv`, deduped on `article_id`.
- `impressions` (**nested**): `impression_id, user_id, timestamp, inview_ids(list),
  clicked_ids(list)`
  - MIND: parse `N123-1`/`N123-0` tokens into inview + clicked; parse `%m/%d/%Y %I:%M:%S %p`.
  - EB-NeRD may carry several clicked ids per impression; never assume exactly one.
- `history` (**long**): `user_id, article_id, timestamp, position` — explode EB-NeRD's parallel
  `*_fixed` lists; for MIND emit ordered rows with a null `timestamp` (ordering via `position` is
  the only temporal signal available).

**`split.py`** — strictly temporal, config-driven:
- MIND: train = 11/09–11/12, val = 11/13–11/14, test = the dev file (11/15).
- EB-NeRD: read the actual `impression_time` range at build time; last `val_days` (default 2) of
  `train/` → val, the rest → train, `validation/` → test. **Use `validation/history.parquet` for
  test users** — pairing test impressions with train history is a subtle leak.
- Writes `data/processed/<ds>/{train,val,test}/impressions.parquet` plus a `split_meta.json`
  recording each split's exact `[t_min, t_max]`.

**`feature_store.py`** — `data/feature_store/<ds>/`:
- `articles.parquet` (cleaned fields + tokenized text + train-split popularity rank)
- `article_embeddings.npy` + `article_ids.npy` (row-aligned)
- `user_profiles.parquet`: `user_id, split, clicked_ids(ordered), n_clicks, last_click_time`
- `stats.json`: catalog size, popularity distribution, cold/warm threshold
- Popularity and IDF are computed **from the train split only** and reused by val/test.

**`make data`** rebuilds all of the above from `data/raw/` in one command.

---

## Q2 — BM25 lexical retrieval (`src/retrieval/bm25.py`)

Implement BM25 directly — do not use `rank_bm25` (far too slow at 51 K docs × tens of thousands of
queries).

1. **Inverted index**: tokenize `title + " " + abstract` (lowercase, strip punctuation, digit
   normalisation, stopwords; Danish stopword list + `æøå`-safe regex for EB-NeRD). Build
   `postings: term -> (doc_ids array, tf array)` — the explicit inverted index the spec asks for —
   then materialise it as a `scipy.sparse.csc_matrix` of pre-computed BM25 document weights so
   scoring is one sparse mat-vec. `k1=1.2, b=0.75` (configurable).
2. **Query**: concatenate titles of the user's last `L` clicked articles (default `L=20`; ablate
   `L ∈ {5, 20, 50}`). Cache per user — many impressions share a user.
3. **Two scoring modes**, both needed downstream:
   - `retrieve()` — top-K over the whole corpus → feeds recall@K.
   - `score_inview()` — score only the impression's inview list → feeds Q4 ranking metrics and Q5.
4. **recall@K for K ∈ {50, 100, 200}**: fraction of ground-truth clicked articles that appear in
   the top-K. Report on val and test, and add a **freshness-restricted** variant (corpus limited
   to articles published before the impression, EB-NeRD only) as an ablation — the unrestricted
   number is optimistic relative to a real serving corpus.

Write results to `reports/recall_bm25_<ds>.json`.

---

## Q3 — Semantic retrieval (`src/retrieval/semantic.py`)

1. **Embeddings**
   - EB-NeRD: load `document_vector.parquet`, align rows to the article table, L2-normalize.
     Articles missing a vector get a zero row and are excluded from the index (log the count).
   - MIND: `sentence-transformers/all-MiniLM-L6-v2` over `title + ". " + abstract`, fp16,
     `batch_size=256`, on GPU. ~65 K unique articles → a couple of minutes. **Cache to
     `.npy`** and skip re-encoding if the cache matches the article-id hash.
2. **ANN index**: FAISS `IndexFlatIP` on normalized vectors (exact — corpus is small) as the
   reference, plus `IndexHNSWFlat` (`M=32, efSearch=64`) as the true ANN. Report ANN-vs-exact
   recall and query latency; that comparison is the "build an ANN index" deliverable *and* a
   scale-analysis data point.
3. **User representation**: mean-pooled, L2-normalized embeddings of the last `L` clicked
   articles. Ablate against a **recency-weighted** pool (exponential decay, half-life config) —
   news decays fast, so this should measurably matter.
4. **recall@K for K ∈ {50, 100, 200}**, same protocol as BM25.
5. **Lexical vs. semantic comparison**: same slices as Q4 (cold-start vs. warm, head vs. tail),
   plus a **hybrid** score fusion (min-max normalize both scores, `α`-weighted sum, sweep `α`) to
   show whether the two arms are complementary.

---

## Q4 — Evaluation harness (`src/eval/`)

`metrics.py` — pure functions over `(y_true, y_score)` per impression:
- **AUC**: per-impression ROC AUC averaged over impressions (standard MIND protocol); skip
  degenerate impressions where all labels are equal, and log how many were skipped.
- **MRR**: reciprocal rank of the first clicked item.
- **nDCG@5, nDCG@10**.
- **Intra-list diversity**: `1 − mean pairwise cosine similarity` of the top-10 items' embeddings
  (report a category-entropy variant too, since the two disagree in interesting ways).
- **Novelty**: mean `−log2 p(item)` over the top-10, with `p` from **train-split** popularity.
- **Coverage**: distinct recommended items across all impressions ÷ catalog size.

`harness.py` — orchestration:
- **Slices** (do both, they answer different questions): cold-start (`n_clicks < 5`) vs. warm, and
  head (top-20% train popularity) vs. tail articles.
- **Bootstrap 95% CI**: resample impressions with replacement, `B=1000`, percentile interval.
  Vectorize over a pre-computed per-impression metric array so all metrics share one resample
  draw — cheap, and the CIs stay comparable across metrics.
- Runs over every scorer: **popularity baseline, BM25, semantic, hybrid**. The popularity baseline
  is not optional — it is the sanity check that says whether the content arms do anything.
- Emits `reports/eval_<ds>.json` + a markdown table for the design note.
- Subsample impressions via `eval_sample_n` (default 20 000, seeded) if full-set runs are slow;
  the CIs make the sampling honest.

---

## Q9 — Anti-gaming (build this alongside Q4, not after)

`tests/test_no_leakage.py` — the spec explicitly requires this test:
- For every (user, impression) profile used by a scorer, assert every contributing click timestamp
  is `< impression_time` (EB-NeRD, where timestamps exist).
- Assert split windows are disjoint and ordered: `train.t_max ≤ val.t_min ≤ val.t_max ≤ test.t_min`.
- Assert popularity/IDF artifacts were fit on train only — hash-check them against a train-only
  recomputation.

**Serving-time ablation**: EB-NeRD's `total_inviews`/`total_pageviews` are future-aggregated, and
`read_time`/`scroll_percentage`/`next_read_time`/`next_scroll_percentage` are post-click. Report
the metric table **with and without** a popularity feature derived from these, and quote the
inflation delta in the design note.

---

## Q5 — Submissions (`src/submission/generate_predictions.py`)

- **MIND** (competition 13967): `prediction.txt` with `<impression_id> [r1,r2,...]` — the 1-based
  rank of each inview article in original order — zipped. Generated for the test split
  (MINDsmall_dev). **First action: open the competition page and confirm the expected input file
  and split**, since a wrong assumption here silently costs the submission.
- **EB-NeRD** (competition 2469): same writer, run on the **validation split as a dry run** per the
  decision above. Add an opt-in `make ebnerd-testset` target that downloads the 1.5 GB test bundle
  so a real submission is one command away.
- A `validate_submission()` function checks row count, id coverage and rank permutation validity
  before writing — catches format errors locally instead of on the leaderboard.

---

## Q6 — Design note (≤4 pages)

Draft in `reports/design_note.md` from the emitted JSON tables, covering: architecture and schema
choices; alternatives rejected (`rank_bm25` vs. sparse BM25, exact vs. HNSW, mean vs.
recency-weighted pooling, provided vs. self-computed embeddings); observations (lexical vs.
semantic per slice, MIND vs. EB-NeRD differences, cold-start behaviour); and the 10× scale
analysis — where it breaks: the dense `|queries| × |corpus|` score path, in-RAM FAISS flat index,
per-user query caching, and the single-process BM25 build. Include the MIND leaderboard
screenshot; state plainly that the EB-NeRD submission was a validation-only dry run.

---

## Build order

1. Env + `requirements.txt` + `src/common/` skeleton; commit early, commit often (Q8).
2. Q1 pipeline end-to-end on **MIND** (simpler schema) → verify `make data`.
3. Extend Q1 to EB-NeRD demo → `split_meta.json` sanity check.
4. `tests/test_split_boundary.py` + `test_no_leakage.py` — before any modelling.
5. Q2 BM25 + recall@K on both.
6. Q3 embeddings + FAISS + recall@K on both.
7. Q4 harness (metrics → slices → bootstrap) over all four scorers.
8. Q9 serving-time ablation table.
9. Q5 submission writer; verify MIND format, submit, screenshot.
10. Q6 design note; final `make all` from a clean `data/processed/`.

---

## Verification

```bash
# clean rebuild from raw (the one-command claim)
rm -rf data/processed data/feature_store && make data

# leakage + boundary + metric unit tests must pass
pytest tests/ -v

# retrieval + eval
make retrieval && make eval

# submission format check without uploading
python src/submission/generate_predictions.py --config config/mind.yaml --validate-only
```

Correctness checks beyond the tests:
- Popularity baseline AUC should land around 0.50–0.60 on MIND; near 0.50 means the label parsing
  is wrong, well above 0.65 means something is leaking.
- BM25 and semantic recall@200 must exceed recall@50 monotonically, and both must beat a random
  baseline by a wide margin.
- `split_meta.json` windows must be disjoint and in chronological order.
- FAISS HNSW recall vs. `IndexFlatIP` should be ≥0.95 at `efSearch=64`; much lower means the
  vectors were not normalized before inner-product search.
- Re-running `make data` twice must produce byte-identical feature-store parquet files.
