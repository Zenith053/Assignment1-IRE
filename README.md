# CS4.406 Assignment 1 — Lexical & Semantic Retrieval on MIND and EB-NeRD

A reproducible pipeline that ranks candidate articles in an impression by click
likelihood, using BM25 over article text and embedding similarity over click
history, with a sliced, bootstrapped evaluation harness.

**Repository:** <https://github.com/Zenith053/Assignment1-IRE>
```bash
git clone git@github.com:Zenith053/Assignment1-IRE.git    # SSH
git clone https://github.com/Zenith053/Assignment1-IRE.git # HTTPS
```

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Python 3.12 is used for wheel coverage. A GPU is optional — it only speeds up
MIND's article encoding (≈11 min on a GTX 1650 Ti, cached afterwards).

MIND lives in a **gated** HuggingFace repo. Accept the terms once at
<https://huggingface.co/datasets/yjw1029/MIND>, then either run
`huggingface-cli login` or `export HF_TOKEN=hf_...`.

## One-command rebuild

```bash
make data     # download -> clean -> split -> feature store, both datasets
make all      # the above, plus retrieval, evaluation and submissions
make test     # leakage, split-boundary and metric tests
```

Every stage is idempotent: `make data` over an already-populated `data/raw/`
re-downloads nothing.

## Pipeline

| Stage | Module | Output |
|---|---|---|
| Download | `src/data/download.py` | `data/raw/` (driven by `config/source.json`) |
| Clean | `src/data/clean.py` | `data/processed/<ds>/{articles,impressions,history}.parquet` |
| Split | `src/data/split.py` | `data/processed/<ds>/{train,val,test}/`, `split_meta.json` |
| Features | `src/data/feature_store.py` | `data/feature_store/<ds>/` |
| Lexical | `src/retrieval/bm25.py` | `reports/recall_bm25_*.json` |
| Semantic | `src/retrieval/semantic.py` | `reports/recall_semantic_*.json` |
| Evaluate | `src/eval/harness.py` | `reports/eval_*.json` |
| Submit | `src/submission/generate_predictions.py` | `reports/submissions/` |

Each module takes `--config config/{mind,ebnerd}.yaml`. Only `clean.py` contains
dataset-specific code; everything downstream is dataset-agnostic and branches on
declared **capability flags** (`has_body`, `has_published_time`, …) rather than
on the dataset name. A capability a dataset lacks is reported as `N/A`, never
silently faked.

## Scale

`config/ebnerd.yaml` has `scale: demo`. Change it to `small` to rerun at ~10×.

## Results (held-out split, evaluated in full)

The test split is the partition each dataset ships as held out — MIND's `dev` file
and EB-NeRD's `validation/` directory — scored end to end with no subsampling.
`split.py` never touches it; it only subdivides the shipped *train* file into
train + val so hyperparameters are never tuned on the reported split.

| | MIND | EB-NeRD demo |
|---|---|---|
| Impressions scored | 73,152 (all of `MINDsmall_dev`) | 25,356 (all of `validation/`) |
| Best AUC | **semantic 0.6423** [0.640, 0.644] | **hybrid 0.5358** [0.532, 0.540] |
| Hybrid AUC (learned) | 0.6388 [0.637, 0.641] | 0.5358 [0.532, 0.540] |
| Semantic AUC | 0.6423 [0.640, 0.644] | 0.5319 [0.528, 0.536] |
| BM25 AUC | 0.5696 [0.567, 0.572] | 0.5242 [0.520, 0.528] |
| Popularity AUC | 0.4950 [0.494, 0.496] | 0.4684 [0.467, 0.469] |
| Best recall@50 (circulating pool) | semantic 0.079 | bm25 0.037 |
| Codabench leaderboard AUC | **0.6567** (MINDlarge_test) | **0.5149** (ebnerd_testset) |

Semantic uses top-5 similarity pooling over the user's **full** click history —
there is no truncation window, which was measured to cost AUC monotonically on
EB-NeRD (0.5029 at 5 clicks, 0.5094 at 20, 0.5242 uncapped). k=5 is the peak of a
1..50 val sweep (`make sweep`); mean-pooling is that sweep's k≥|history| endpoint
and costs 0.010 AUC on MIND, 0.037 on EB-NeRD. Hybrid is a logistic
regression over (bm25, semantic) fit on the val split, replacing an earlier fixed-α
blend; it ties semantic on MIND and leads it on EB-NeRD, with overlapping CIs in
both cases. Full numbers, slices and confidence intervals are in `reports/`, and
the analysis is in `reports/design_note.md`.

The two Codabench figures were produced before the truncation window was removed
and have not been resubmitted.

## Known limitations

- **Semantic recall is not comparable across datasets** — Danish and English
  force different encoders. Only BM25-vs-semantic *within* a dataset is a fair
  comparison.
- The `circulating` candidate pool is derived from the evaluation split, so it
  is an optimistic bound rather than a deployable filter. Reported alongside
  the honest full-catalogue number.
- `recall_at_k` (Q2.4/Q3.4) still measures mean-pooled semantic similarity;
  top-5 pooling is only wired into the ranking harness and submissions so far,
  because it yields no single query vector for FAISS to index — retrieving under
  it needs one ANN query per history click, merged.
- The Codabench scores above were produced before the truncation window was
  removed (MIND at N=50, EB-NeRD at N=20) and have not been resubmitted; the
  offline tables in `reports/` are full-history.
