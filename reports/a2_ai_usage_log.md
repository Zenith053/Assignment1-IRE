# Assignment 2 — AI usage log

Covers Assignment 2. The Assignment 1 log is `reports/ai_usage_log.md`.

| | |
|---|---|
| Tool | Claude Code (Anthropic Claude Opus 5), run locally in the repo |
| Transcript | Chat exports are attached with the submission; this file summarises, per question, what was asked, what the tool produced, and how each output was checked |
| Code authorship | Every source file listed below was typed by the tool and reviewed by us. No file in Assignment 2 was written by hand. What was human was the direction: choosing the models and the experiment design, cutting scope, deciding what counts as an honest claim, and rejecting or correcting the tool where it was wrong |

Nothing below was accepted because it looked plausible. Every number in the design
note is backed by a test, a cross-check against an independent implementation, or a
reproduction of an already-known value — the "verified by" line under each question
says which.

---

## Q1 — Behavioural features

**Asked for:** a feature set built only from information available before the
impression being scored, with the leakage boundary enforced in code rather than
trusted.

**Produced:** `src/rerank/features.py` (30 features), `src/rerank/timeline.py`
(a vectorised trailing-click counter), and the accompanying tests.

**Direction given:** counts must come from a strictly half-open window `[t-w, t)`,
never `[t-w, t]`, so the impression's own click can never enter its own features.
The tool's first counter was correct but too slow at EB-NeRD scale, so it was asked
to pack the timeline into sorted integer keys and count by binary search.

**Verified by:** a counterfactual test on real data — appending a future click and
confirming no feature value moves — plus an equality test between the fast counter
and the straightforward implementation.

---

## Q2 — Re-rankers over the retrieved set

**Asked for:** GBDT and MLP re-rankers over the Q1 features, evaluated in two
universes: re-ranking the given inview list (A) and re-ranking what our own
retriever returns (B).

**Produced:** `src/rerank/gbdt.py`, `mlp.py`, `candidates.py`, `retriever.py`,
`evaluate_reranker.py`.

**Direction given:** Universe B must report metrics conditional on the click
actually being retrieved, otherwise recall gets counted twice and the re-ranker is
blamed for the retriever's misses. The tool was also asked to keep the retriever a
union of BM25, FAISS and popularity rather than a single source, so that the
candidate pool keeps circulating.

**Where the tool was wrong, and how it was fixed:** the conditional-metric filter
it first wrote kept every impression, because the metric helper returns `0.0`
rather than `None` when no positive is present. The bug surfaced while cross-reading
the Q5 numbers; the fix selects impressions where the label vector actually contains
a retrieved click.

**Two crashes it diagnosed:** LightGBM segfaulted whenever FAISS or Torch was
already imported, and FAISS aborted once the MLP had loaded Torch. Both come from
three packages each bundling their own OpenMP runtime. We rejected the tool's first
suggestion (`KMP_DUPLICATE_LIB_OK`) because silently loading two runtimes can give
wrong results, not just crashes; the fix was to pin both libraries to a single
thread, and to run FAISS-only work in a Torch-free child process where that was not
enough.

**Verified by:** reproduction of each crash before and after the fix, and metric
values cross-checked against the Assignment 1 harness.

---

## Q3 — NRMS baseline and a principled improvement

**Asked for:** first an explanation of NRMS from first principles — what multi-head
self-attention, pooling and embeddings actually do — before any code was written,
then the implementation.

**Produced:** `src/baseline/news_data.py` (loader and sampler),
`nrms.py` (the model: news encoder, user encoder, dot-product scorer),
`candidate_signals.py` (popularity and freshness signals),
`train_nrms.py` (training CLI), `q3_tables.py` (aggregation and confidence
intervals), `freshness_curve.py`, and the tests for each.

**Direction given:** small datasets only — no large-set training, because the
compute was not available. The improvement was chosen as PP-Rec-style popularity
plus freshness over a category-aware encoder, because it addresses a failure we had
actually measured. Three seeds wherever a gain is claimed. Most importantly: the
improved variant is selected on **validation** AUC, never on test.

**Where the tool was wrong:** it predicted freshness alone would be useless on
EB-NeRD, reasoning from a monotone "newer ranks higher" rule that scores 0.5009 AUC.
The trained freshness model reached 0.647. Asked to explain the contradiction, it
produced the click-rate-against-age curve, which is an inverted U peaking at 2–4
hours — a monotone rule cannot capture that, but a learned signal can. Both the
claim and the reasoning behind it were corrected.

**A selection trap avoided:** with one seed, the plain-sum variant looked best on
test. Running two more seeds and selecting on validation instead shrank the
advantage from +0.013 to +0.005, which is the number the report states.

**Verified by:** the ablation with paired bootstrap confidence intervals, and the
MIND scores reproduced against Microsoft's official scorer.

---

## Q4 — Serving and scale

**Asked for:** the pipeline analysed as a served system rather than an offline
script, built and explained one phase at a time (setup, single-request path, memory,
latency, cost, 10× scale, write-up).

**Produced:** `src/serving/` — the request pipeline, memory measurement, latency
benchmark, queueing cost model, and the 10× scaling study.

**Direction given:** the serving path must be proven to serve the same model the
offline evaluation scored, not a re-implementation that happens to be close.

**Where the tool was wrong:** it attributed the slow path to re-cutting the BM25
matrix. Asked to measure the stages separately rather than reason about them, the
real cost turned out to be a token dictionary rebuilt over every article on each
request (~20 ms against 0.35 ms for the BM25 step). The same class of mistake — a
per-call rebuild inside a loop — appeared again in an analysis script and caused a
25-minute hang; both were found by measuring, not by reading.

**Also corrected:** the draft over-claimed in four places, quoting a queueing result
that had never been simulated, calling the timings repeatable "within a few percent"
without evidence, and presenting a MIND-only latency and a MIND-only cost as if they
covered both datasets. All four were narrowed to what was measured.

**Verified by:** the single-request path reproducing saved offline scores to within
4e-6 with identical top-1, and the queue simulator checked against closed-form
M/M/1 results.

---

## Q5 — Extended evaluation and Codabench

**Asked for:** evaluation beyond accuracy (diversity, novelty, coverage, cold/warm
and head/tail slices), and submissions to both Codabench competitions.

**Produced:** `src/eval/evaluate_twostage.py`, `src/eval/q5_extended.py`,
`src/submission/predict_nrms.py`, `validate_zip.py`, `codabench_analysis.py`.

**Direction given:** read the official submission rules on the competition pages
before generating anything, and verify the inference path on labelled data before
trusting it on a hidden set.

**The finding that changed the submission:** the hidden test sets carry no clicks,
so the trailing-popularity features our best Q3 model depends on do not exist there.
Rather than assume how bad that is, the tool was asked to measure it on our own
labelled split: the model collapses to 0.4810 AUC with only pre-test-period clicks
and 0.4854 with none — below chance. The click-free models were submitted instead.

**Verified by:** the MIND inference path reproducing our offline AUC on the dev
split (0.6232, and 0.6235 under Microsoft's official `evaluate.py`), and every zip
checked line by line against its source file by an independently written validator —
which did catch real defects, a stray folder inside the archive and a row order that
no longer matched the input.

---

## Q6 — Design note

**Asked for:** a design note as a PDF with its LaTeX source, covering everything the
assignment asks for, using the metrics already stored from Q1–Q5.

**Produced:** `src/report/build_design_note.py`, which generates the `.tex`, the
figures and the compiled PDF.

**Direction given:** no number may be typed into the note by hand — every value is
read from the stored result JSONs at build time, so the note cannot drift from the
experiments. Claims are stated with their confidence intervals or not at all.

**Where the tool was wrong:** it carried a feature count of 26 across from an
implementation write-up; checking the actual feature list showed 30.

---

## Q9 — Serving-time ablation

**Asked for:** a measurement of how much the model is flattered by features that
would not exist at serving time.

**Produced:** a `--leaky` arm in `evaluate_reranker.py` that adds the article-level
aggregates (total inviews, pageviews, read time) which are only known after the
fact.

**Result:** the EB-NeRD GBDT rises from 0.7499 to 0.7605 AUC, an inflation of
+0.0105 with a 95% interval of [+0.0074, +0.0136] that does not cross zero.

---

## Limits of this log

Prompts are summarised here, not transcribed; the full conversation exports are
attached separately. Attribution is at file granularity — line-level authorship
inside a file is not tracked.
