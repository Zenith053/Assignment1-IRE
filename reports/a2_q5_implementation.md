# Assignment 2, Q5 — Implementation Report

Scope: **Q5 (extended evaluation of the full two-stage pipeline)**, built on the `q4-serving` branch
(`origin/parth/q4-serving`, which already carries Q1+Q2, Q3 and Q4). Leaderboard submission/zip
generation is explicitly out of scope for this pass — handled separately.

| Q5 requirement | Where | Headline |
|---|---|---|
| All metrics (AUC, MRR, nDCG@5, nDCG@10, diversity, novelty, coverage) | `src/eval/evaluate_twostage.py` | reported per scorer, per slice, both datasets |
| ≥2 slices: cold-start vs. warm, head vs. tail | `build_slices()` | 6 slices: `cold_users`/`warm_users` (true cold-start), `low_history_users`/`high_history_users` (quartile fallback), `head_clicks`/`tail_clicks` |
| Bootstrap 95% CI for all reported metrics | reuses `M.bootstrap_ci` (A1) | every metric, every slice, both scorers |
| For the **full two-stage pipeline** | `UnionRetriever` (stage 1) → Universe-A-trained GBDT (stage 2) | not a re-evaluation of the single-stage A1 scorers `harness.py` already covers |

Full outputs: [`reports/eval_twostage_mind_test.json`](eval_twostage_mind_test.json),
[`reports/eval_twostage_ebnerd_test.json`](eval_twostage_ebnerd_test.json). Neither dataset's
result has been committed — see *Status* at the end.

---

## What was built

`src/eval/harness.py` (A1) already has every extended metric (diversity, novelty, coverage),
slicing, and bootstrap-CI machinery Q5 asks for — but only for single-stage scorers ranking an
impression's own inview list. It has no retrieval step and doesn't know about the GBDT. Q5 asks
specifically for these metrics over the **two-stage** system Q2 built (retrieve top-K, then
re-rank), so `src/eval/evaluate_twostage.py` (new) wires the same metric primitives through
`CandidateSet(universe="retrieved")` instead:

1. Train the Universe-A GBDT (`build_universe_a_frame` + `gbdt.train`, unchanged from Q2's train/val split roles).
2. Stage 1: `UnionRetriever.retrieve()` — top-200 per test user (BM25 ∪ FAISS-semantic ∪ popularity).
3. Stage 2: score the retrieved candidates with the Universe-A-trained booster (the same
   distribution-shift comparison `evaluate_reranker.py` reports for Universe B).
4. `extended_metrics()` — AUC/MRR/nDCG@5/nDCG@10 plus diversity/cat-entropy/novelty per impression,
   coverage over the whole run; reuses `M.intra_list_diversity`, `M.category_entropy`, `M.novelty`
   verbatim from `src/eval/metrics.py` (A1), same as `harness.py` does.
5. `build_slices()` — `is_cold`/`is_low_history` from the feature-store user profiles, `is_head`
   from the article feature store; identical definitions to `harness.py`'s, with one correction
   (see *Finding 1* below).
6. `summarise()` — point + 95% bootstrap CI, reused pattern from `harness.py`/`evaluate_reranker.py`.

Two scorers are compared, matching Q2's before/after naming so results read alongside
`reports/rerank_<ds>.json`: `stage1_order` (semantic similarity, standing in for the retriever's
own ranking — `UnionRetriever.retrieve()` itself returns a deduplicated *set*, not a score) and
`gbdt` (the trained booster).

## Findings

**1. `cold_users` was mislabeled in the existing harness, and stays genuinely N/A on EB-NeRD.**
`harness.py`'s `"cold_users"` slice is actually built from `is_low_history` (a dataset-relative
quartile), not `is_cold` (the assignment's actual cold-start definition, <5 train clicks).
`evaluate_twostage.py` uses the correct `is_cold` for `cold_users`/`warm_users`, and reports
`low_history_users`/`high_history_users` alongside as the always-non-empty stand-in. On EB-NeRD
(now `scale: small`, min history ≈ 5 clicks) this makes `cold_users` **N/A — 0 impressions** —
an honest result, not a bug, and exactly what was anticipated before this was measured.

**2. A real bug in Q2's Universe-B "conditional on retrieval" numbers.** `M.ndcg()`/`M.mrr()`
never return `None` for a zero-click impression — they return `0.0` (only `M.auc()` returns
`None`). `evaluate_reranker.py`'s Universe-B code filters on `if v is not None` to compute
"nDCG@10 | click retrieved" — which filters nothing, since every value is a real float. The
previously reported "conditional" figures (e.g. MIND stage1_order 0.0215, gbdt 0.0099) were
actually **unconditional** means with zeros baked in for every non-retrieved impression, and
`end_to_end = recall@K × that` **double-counted** the recall penalty rather than correctly
decomposing it. `evaluate_twostage.py` fixes this locally (`*_given_retrieved` filters
`cand.labels_by_imp[i].sum() > 0` *before* averaging); `evaluate_reranker.py` itself still has
the bug and was not touched in this pass — flagging for a follow-up fix there.

**3. The old "conditional" figures were mislabeled, not wrong as numbers.** Since a non-retrieved
impression contributes exactly `0.0` to `ndcg@10`, the mathematical identity
`unconditional_mean = recall@K × conditional_mean` always holds. The previously reported
"nDCG@10 | click retrieved" values are therefore exactly the *unconditional* mean by another
name — real, just mislabeled — and dividing by `recall@K` recovers the true conditional figure,
which is substantially higher on both datasets and every scorer (2.2×–8.2× higher across the
four scorer/dataset combinations here). The `end_to_end` figures compounded this by applying
`recall@K` a second time on top.

**4. EB-NeRD's GBDT survives the distribution shift dramatically better than MIND's, confirmed
at `scale: small` (not `demo`).** `stage1_order` AUC is 0.4930 (at chance) on EB-NeRD's retrieved
candidates; `gbdt` reaches 0.8454. MIND shows the opposite pattern already documented in
`reports/a2_q1_q2_implementation.md`: `gbdt` (0.5705) is *worse* than `stage1_order` (0.6696).

## Results — Universe B (retrieved top-200), `all` slice

| dataset | scorer | AUC (n defined) | MRR | nDCG@5 | nDCG@10 | nDCG@10\|retrieved | diversity | novelty | coverage |
|---|---|---|---|---|---|---|---|---|---|
| MIND | stage1_order | 0.6696 [0.655,0.683] (960) | 0.0192 | 0.0150 | 0.0197 | 0.0717 | 0.682 | 15.695 | 0.029 |
| MIND | gbdt | 0.5705 [0.556,0.584] (960) | 0.0087 | 0.0042 | 0.0060 | 0.0218 | 0.852 | 10.045 | 0.017 |
| EB-NeRD (small) | stage1_order | 0.4930 [0.467,0.520] (446) | 0.0045 | 0.0023 | 0.0040 | 0.0316 | 0.116 | 11.992 | 0.073 |
| EB-NeRD (small) | gbdt | 0.8454 [0.826,0.863] (446) | 0.0277 | 0.0274 | 0.0329 | 0.2583 | 0.136 | 12.277 | 0.076 |

On MIND, `stage1_order`'s conditional nDCG@10 (0.0717) is genuinely *higher* than `gbdt`'s
(0.0218) — the distribution-shift cost documented in `a2_q1_q2_implementation.md` is real and
larger than the earlier (buggy) figures suggested, not an artifact of the fix. On EB-NeRD the
direction flips the other way, by an even larger margin (0.0316 → 0.2583).

## Results — slices (gbdt scorer, MIND / EB-NeRD)

| slice | MIND n | MIND recall@200 | MIND nDCG@10\|retrieved | EB-NeRD n | EB-NeRD recall@200 | EB-NeRD nDCG@10\|retrieved |
|---|---|---|---|---|---|---|
| all | 3,500 | 0.274 | 0.0218 | 3,500 | 0.127 | 0.258 |
| cold_users | 377 | 0.199 | 0.0125 | **N/A (0)** | — | — |
| warm_users | 3,123 | 0.283 | 0.0226 | 3,500 | 0.127 | 0.258 |
| low_history_users | 637 | 0.204 | 0.0144 | 295 | 0.112 | 0.235 |
| high_history_users | 2,863 | 0.290 | 0.0229 | 3,205 | 0.129 | 0.260 |
| head_clicks | 100 | 0.500 | 0.1869 | 17 | 0.882 | 0.356 |
| tail_clicks | 3,400 | 0.268 | 0.0127 | 3,483 | 0.124 | 0.255 |

Both datasets: head-click impressions retrieve far better than tail (0.500 vs 0.268 MIND; 0.882
vs 0.124 EB-NeRD) — unsurprising, since `UnionRetriever` includes a popularity arm, but it
quantifies exactly how much of the retrieval ceiling is "the article was popular enough to be
in someone's top-20" rather than genuine personalized matching.

## Caveats

- **Sample size.** 3,500 impressions/split (matching the EB-NeRD Q1/Q2 run), not A1's usual
  20,000 — `head_clicks` lands at only 17–100 impressions, so its CI is wide; read it as
  directional, not precise.
- **`gbdt` here is Universe-A-trained, not retrained on retrieved candidates** — deliberate
  (matches `evaluate_reranker.py`'s existing convention), not an oversight.
- **`stage1_order` is a proxy, not the retriever's literal ranking** — `UnionRetriever.retrieve()`
  returns a deduplicated set with no score; semantic similarity stands in for "how stage 1 would
  have ordered its own candidates," same convention as `evaluate_reranker.py`.
- **Finding 2's fix lives only in `evaluate_twostage.py`.** `reports/rerank_{mind,ebnerd}.json`'s
  Universe-B numbers (from the earlier Q1/Q2 pass) still have the double-counting bug described
  above and should be treated as superseded by this report's numbers, not cross-cited.
- **EB-NeRD is `scale: small` here**, matching this branch's current config — not directly
  comparable to `reports/a2_q1_q2_implementation.md`'s `demo`-scale numbers.

## Files

| File | Status |
|---|---|
| `src/eval/evaluate_twostage.py` | new, **not committed** |
| `reports/eval_twostage_mind_test.json`, `reports/eval_twostage_ebnerd_test.json` | new, **not committed** |
| `reports/a2_q5_implementation.md` | new, **not committed** |

## Reproduce

```bash
python src/eval/evaluate_twostage.py --config config/mind.yaml --sample 3500
python src/eval/evaluate_twostage.py --config config/ebnerd.yaml --sample 3500
```

## Status

Per instruction, **nothing in this pass was committed or pushed** — all three files above are
untracked in the working tree on `q4-serving`. Not done: a `Makefile` target (`q5`), and
`evaluate_reranker.py`'s own Universe-B conditional-metric bug (Finding 2) was found but not
fixed, since it lives in a file this pass wasn't asked to touch. Leaderboard screenshots/zip
generation are Parth's responsibility, per instruction, and are not part of this report.
