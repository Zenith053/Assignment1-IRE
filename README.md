# CS4.406 Assignment 1 — Lexical & Semantic Retrieval on MIND and EB-NeRD

A reproducible pipeline that ranks candidate articles in an impression by click
likelihood, using BM25 over article text and embedding similarity over click
history, with a sliced, bootstrapped evaluation harness.

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

## Results (test split)

| | MIND | EB-NeRD demo |
|---|---|---|
| Best AUC | **semantic 0.6375** [0.634, 0.642] | semantic 0.5195 [0.515, 0.524] |
| Hybrid AUC (learned) | 0.6337 [0.630, 0.638] | 0.5169 [0.512, 0.521] |
| BM25 AUC | 0.5671 [0.563, 0.571] | 0.5098 [0.505, 0.514] |
| Popularity AUC | 0.4955 | 0.4685 |
| Best recall@50 (circulating pool) | semantic 0.075 | bm25 0.026 |
| Codabench leaderboard AUC | **0.6567** (MINDlarge_test) | **0.5149** (ebnerd_testset) |

Semantic uses top-5 similarity pooling; hybrid is a logistic regression over
(bm25, semantic) fit on the val split, replacing an earlier fixed-α blend. Full
numbers, slices and confidence intervals are in `reports/`, and the analysis is
in `reports/design_note.md`.

## Known limitations

- **Semantic recall is not comparable across datasets** — Danish and English
  force different encoders. Only BM25-vs-semantic *within* a dataset is a fair
  comparison.
- The `circulating` candidate pool is derived from the evaluation split, so it
  is an optimistic bound rather than a deployable filter. Reported alongside
  the honest full-catalogue number.
- `recall_at_k` (Q2.4/Q3.4) still measures mean-pooled semantic similarity;
  top-5 pooling is only wired into the ranking harness and submissions so far.
