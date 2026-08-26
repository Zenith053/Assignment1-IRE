# AI Usage Log

Assignment 1 deliverable Q7.4. Chronological record of every prompt given to the
AI assistant (Claude Opus 5, via Claude Code CLI), what each produced, and which
code is AI-generated versus human-written.

Session dates: 2026-08-21 to 2026-08-24.

---

## Attribution summary

**All code in `src/`, `tests/`, `config/`, the `Makefile` and `requirements.txt`
is AI-generated**, written by the assistant in response to the prompts below, and
reviewed/directed by the author.

**Human-written / human-decided:**

| Item | Contribution |
|---|---|
| `fd2de72` "implemented script to collect the data" | pre-existing commit before the session |
| Directory scaffold (`src/`, `config/`, `data/`) and original `Makefile`/`README.md`/`.gitignore` | authored before the session |
| Raw data acquisition (505 MB pre-downloaded) | performed by the author |
| **All design decisions listed in §Decisions** | author's choices, made in response to options presented |
| Prompt direction, scope, corrections | author throughout |

**Third-party, vendored unmodified:** `tools/evaluate_official.py` — Microsoft's
scorer from <https://github.com/msnews/MIND/blob/master/evaluate.py>, included so
results reproduce offline. Not written by the author or the AI.

---

## Decisions the author made (not AI choices)

These were put to the author as explicit questions; the answers changed what was built.

1. **EB-NeRD Codabench** → "Validation-only dry run" (later reversed: author asked
   for a real testset submission).
2. **Embeddings** → "Provided for EB-NeRD + own for MIND".
3. **EB-NeRD scale** → "work on demo then i will decide for final".
4. **Storage format** → author asked for the long/wide rationale, then directed
   **"Use pandas for both as we want a clean common pipeline"**, overriding the
   assistant's initial polars-for-EB-NeRD recommendation.
5. **Comment style** → author directed one-line inline comments on important syntax
   alongside docstrings.
6. **Git branch** → "Rename to main"; **"Commit and push everything"**.

---

## Prompt log

### 1. Planning

**P1.** *(Assignment PDF attached)* `come up with a detailed plan to complete part 1`
→ Explored repo, verified data schemas, checked wheel availability, asked three
scoping questions, wrote the implementation plan.

**P2.** *(answers to scoping questions)* `Validation-only dry run` /
`Provided for EB-NeRD + own for MIND` / `work on demo then i will decide for final`

**P3.** `as of now just save the plan to Assignment1`
→ Plan copied to `PLAN.md`; no other files touched.

### 2. Data acquisition

**P4.** `create download.py also create a seperate source.json file which should
include links, source, for all the data we require to collect and download.py must
go through source.json to download all the data files in one place`
→ `config/source.json` (12 sources), `src/data/download.py` (stdlib-only,
idempotent, resumable). Discovered the MIND HF repo is now **gated** (401).

**P5.** `just to tell you , i also prefer one line comment for important syntaxes
along with docstirng`
→ Comment style applied to `download.py` and all subsequent code; saved to memory.

**P6.** `what is 'driven entirely by the manifes'`
→ Explained the phrase; found and fixed a contradiction (`choices=["mind","ebnerd"]`
was hardcoded in argparse).

### 3. Schema and cleaning

**P7.** `in the plan clarify me few things 1> what should i use long format or wide
format for storing 2> Use pandas for both as we want a clean common pipleline which
can work on both dataset / also tell me since we are using same pipleline for two
different kind of dataset how are solving the differences in the dataset`
→ Added "Storage format: long vs wide" and "Reconciling the two datasets" sections
to the plan; switched from polars to pandas throughout.

**P8.** `now create clean.py`
→ `src/common/{config,io,text}.py`, `config/{mind,ebnerd}.yaml`, `src/data/clean.py`.
Found: MIND dev/train impression-id collision; 7,233 articles containing quotes;
27.9% multi-click.

**P9.** `did you create clean.py`
→ Confirmed created but **untested** (dependency install had been killed); reran.

### 4. Version control

**P10.** `change the current remote repo to git@github.com:Zenith053/Assignment1-IRE.git`

**P11.** `when i do push it says it has no remote branch`
→ Diagnosed missing upstream. Found `.gitignore`'s unanchored `data/` was silently
excluding `src/data/` — `download.py` and `clean.py` would never have been committed.

### 5. Splitting and the leakage bug

**P12.** `lets do split.py`
→ `src/data/split.py`. Found a **real leak in the assistant's own `clean.py`**:
EB-NeRD's validation history snapshot spans the entire train impression window.

**P13.** `continue where you left off`

### 6. Full build

**P14.** `go on and complete the assignment part 1`
→ `feature_store.py`, `text.py`, `bm25.py`, `semantic.py`, `popularity.py`, `pool.py`,
`metrics.py`, `harness.py`, `generate_predictions.py`, three test modules, README,
design note. 27 tests passing.

### 7. Validation against official tooling

**P15.** `is it tested against the official test data?`
→ Answered **no**; neither official test set was downloaded and nothing had been
submitted.

**P16.** `should i submit the zip of codabench`
→ Advised **not yet** — format unverified.

**P17.** `https://github.com/msnews/MIND/blob/master/evaluate.py take a look at thier
evaluation script, can you determine if we need to run it on large data?`
→ Ran Microsoft's scorer locally. **Found a real bug: MRR was computed first-hit-only
instead of averaging over all positives** — overstated by 0.045 on MIND.

**P18.** `do the prediction on mind large and create a zip`
→ Downloaded MINDlarge_test (2.37M impressions); wrote `predict_raw_mind.py` with
streaming (the naive approach needed ~135 GB).

**P19.** `is the zip for large test done`
**P20.** `where is the zip?`
**P21.** `can i submit submission_min_large_test.zip on codabench`
→ Verified integrity; corrected an earlier error (10 submissions/day, not 1).

### 8. EB-NeRD submission and tuning

**P22.** `now how can we make it even better` *(interrupted)* +
`Also create a zip for codabench submission for ebnerd`
→ Downloaded `ebnerd_testset`; wrote `predict_ebnerd_test.py`; 13.5M impressions,
required a second streaming rewrite.

**P23.** `how can i imporve the score, for mind the results came at 0.633 for auc`
→ Ablated user-representation strategies. **Top-5 similarity aggregation: +0.0115 AUC**
confirmed by the official scorer. Measured and rejected position and popularity features.

**P24.** `can you check what is running on my computer which is maxing my cpu`
→ Identified the assistant's own job saturating 8/12 cores; fixed by capping BLAS threads.

**P25.** `what does score_inview does`
→ *Not answered at the time; answered in §Appendix below.*

**P26.** `create a detailed list of all prompts that i gave you`
→ This document.

---

## Appendix: answer to P25 — what `score_inview` does

The plan named a `score_inview()` method; in the delivered code that role is filled by
**`BM25Index.score_pairs()`** in [`src/retrieval/bm25.py`](../src/retrieval/bm25.py).

The pipeline needs two different scoring modes:

| mode | method | question it answers | feeds |
|---|---|---|---|
| retrieval | `retrieve()` | "of all articles, which top-K best match this user?" | recall@K (Q2/Q3) |
| in-impression | `score_pairs()` | "given the candidates actually shown, how should they be ordered?" | AUC/MRR/nDCG (Q4), submissions (Q5) |

`score_pairs(queries, query_idx, doc_idx)` takes aligned arrays of (query, document)
pairs and returns one BM25 score per pair, computing a row-wise dot product of two
sparse matrices in batches rather than materialising a full query×document score
matrix. That distinction matters: scoring all 65,238 MIND articles against 17,147 user
queries is 1.1 billion cells, whereas only ~5.8M (query, candidate) pairs are actually
needed for the ranking metrics.

The two modes measure genuinely different tasks, which is why the results diverge —
popularity is a strong candidate generator yet scores **below random** as a
within-impression ranker, because the inview list is already popularity-curated by the
production system.
