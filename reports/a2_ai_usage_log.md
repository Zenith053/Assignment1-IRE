# Assignment 2 — AI usage log

Covers Assignment 2 only. A1's log is `reports/ai_usage_log.md` (unchanged).

| | |
|---|---|
| Tool | Claude Code (Anthropic Claude Opus 5), run locally in the repo |
| Sessions | 16–20 Sep 2026, Parth Dhawale (Q3, Q4, Q2 fixes, Codabench, Q6/Q9 write-up) |
| | Peeyush Prashant used the same tool for Q1, Q2 and Q5; their own prompt log covers those sessions |
| Transcript | Chat exports attached with the submission (Moodle); this file summarises what was asked, what the tool produced, and how each output was checked |

**How to read the "authorship" column below.** *AI-written, human-directed and reviewed* means the human set the goal,
chose the approach when options were offered, reviewed the diff and the numbers, and asked for corrections; the tool
typed the code. Nothing was accepted because it looked plausible: every result below is backed by a test, a
cross-check against an independent implementation, or a reproduction of an already-known number.

---

## Per-question authorship

| Part | Files | Authorship |
|---|---|---|
| Q1 features | `src/rerank/features.py`, `src/rerank/timeline.py`, `tests/test_rerank_features.py`, leakage tests | AI-written, human-directed (Peeyush) |
| Q2 re-rankers | `src/rerank/gbdt.py`, `mlp.py`, `candidates.py`, `retriever.py`, `evaluate_reranker.py` | AI-written, human-directed (Peeyush); crash + metric fixes by Parth's session (below) |
| Q3 NRMS + improvement | `src/baseline/*` (`news_data.py`, `nrms.py`, `candidate_signals.py`, `train_nrms.py`, `q3_tables.py`, `freshness_curve.py`), `tests/test_nrms_*.py`, `tests/test_candidate_signals.py` | AI-written, human-directed (Parth) |
| Q4 serving & scale | `src/serving/*`, `tests/test_serving.py`, `test_measure_memory.py`, `test_cost_model.py`, `test_scale_10x.py`, `config/serving.yaml` | AI-written, human-directed (Parth) |
| Q5 extended evaluation | `src/eval/evaluate_twostage.py` | AI-written, human-directed (Peeyush) |
| Q5 Codabench | `src/submission/predict_nrms.py`, `validate_zip.py`, `codabench_analysis.py` | AI-written, human-directed (Parth) |
| Q6 design note | `src/report/build_design_note.py`, `reports/design_note/*` | AI-written, human-directed (Parth) |
| Q9 | `--leaky` arm in `evaluate_reranker.py` | AI-written, human-directed (Parth) |
| A1 code reused unchanged | `src/data/*`, `src/retrieval/*`, `src/eval/harness.py`, `src/eval/metrics.py` (one function added in A2) | A1 (see A1 log) |

Human-written in A2: no file was typed by hand. The humans chose the models, the experiment design, the scope
cuts, and every "is this honest?" decision, and rejected or redirected the tool where it was wrong (examples below).

---

## What was asked, in order (Parth's sessions)

**16 Sep — Q3 setup and baseline.** Asked to plan Q3, then to explain the assignment, NRMS and each step from first
principles before implementing. Directed: EB-NeRD to be rebuilt at `scale: small` first; small datasets only (no
large-set training); Kaggle used for extra seeds when the Mac GPU became the bottleneck.
Produced: the data loader, the PyTorch NRMS port, the training CLI, the queue script, the Kaggle bundle/notebook.

**17 Sep — Q3 improvement, ablation, CIs; then Q4.** Directed: popularity + freshness as the principled change
(chosen over a category-aware encoder); 3 seeds where a gain is claimed, 1 for the other ablation arms; the
improved variant to be selected on *validation* AUC, never test. Then Q4 phase by phase (setup, one-request
pipeline, memory, latency, cost, 10×, write-up), with an explanation of each phase before it was built.

**18–19 Sep — Codabench, Q2 re-run, Q6.** Directed: check the official submission rules on the competition pages
before generating anything; verify the inference path on labelled data first; do "whatever is necessary" for Q2;
build the design note as a PDF with a LaTeX source; leave the leaderboard screenshots to be added by hand.

**20 Sep — gaps.** Asked to check a list of gaps raised by the teammate's tooling, add the missing `make` targets,
build the Q9 arm for the A2 re-ranker, and write this log.

---

## Corrections the human made to AI output

These are the points where the tool was wrong or incomplete and was redirected. They are listed because they are
the reason the numbers in the report can be trusted.

1. **Wrong explanation of a measured result.** The tool first attributed the slow `as_is` request path to
   re-cutting the BM25 matrix. Measured separately, the cost was `build_queries` rebuilding a token dictionary over
   every article (~20 ms vs 0.35 ms). The claim was corrected in the Q4 report and the design note.
2. **A prediction contradicted by data.** The tool predicted freshness alone would be useless on EB-NeRD from a
   monotone "newer is better" check (AUC 0.502). The trained `nrms_fresh` reached 0.647, and the click-rate curve
   showed the relation is an inverted U peaking at 2–4 h. Both the claim and the reasoning were corrected.
3. **Overstated sentences in the draft note.** Four claims (a queueing result never simulated, "within a few
   percent" repeatability, a MIND-only p99 and a MIND-only cost presented as covering both datasets) were caught in
   review and narrowed to what was measured.
4. **A wrong count copied from a teammate's doc.** "26 features" was carried into the design note; checking
   `FEATURE_NAMES` showed 30. Fixed.
5. **Test-set selection avoided.** When the plain-sum MIND variant looked best on test with one seed, the human
   directed two more seeds and selection on validation instead — the test-set advantage then shrank from +0.013 to
   +0.005, which is what the report states.
6. **Bugs the tool found in its own or the teammate's code, then fixed:** Universe B's "conditional on retrieval"
   metrics filtered nothing (recall applied twice); a `result` variable used before assignment in the Q9 arm; an
   O(n) dictionary rebuilt inside a 5.5M-iteration loop; a memory-blow-up in the bootstrap at EB-NeRD scale.

## How outputs were verified

| Claim | Check |
|---|---|
| Q3 model is what the serving path serves | One-request path reproduces saved offline scores to 4e-6, identical top-1 |
| Submission files are correct | Verified on labelled splits first (MIND dev AUC 0.6232 = offline; Microsoft's official `evaluate.py` 0.6235), then every zip validated line-by-line against the source file |
| No future-click leakage | Counterfactual test on real data; vectorised counter tested equal to the Q1 implementation |
| Metric implementations | A1's harness vs Microsoft's official scorer; new bootstrap tested against `paired_bootstrap_ci` and against M/M/1 queueing theory |
| Design-note numbers | Generated from the stored result JSONs by `src/report/build_design_note.py`, not typed |
| Everything runs | `pytest tests/` — 89 passed, 3 skipped (skips are features MIND does not have) |

## Limits of this log

Prompts are summarised, not transcribed; the full conversation exports are attached separately. Line-level
attribution inside a file is not tracked — the granularity here is file plus the review record above.
