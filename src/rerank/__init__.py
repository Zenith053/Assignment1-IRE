"""Assignment 2: behavioural features and re-rankers on top of A1's retrieval pipeline.

Nothing in here reimplements A1 - BM25, semantic scoring, popularity and the
per-impression evaluation metrics are all reused from `src.retrieval` and
`src.eval`. This package adds only what A1 did not have: click-history/session/
article behavioural features (`timeline.py`, `features.py`), a stage-1 union
retriever (`retriever.py`), the two evaluation universes (`candidates.py`), and
the two re-rankers trained over the feature frame (`gbdt.py`, `mlp.py`).
"""
