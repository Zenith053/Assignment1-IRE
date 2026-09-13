"""The two evaluation universes, behind one shape.

The leaderboards and A1's harness score re-ranking the impression's own
`inview_ids` (~12 candidates, editorially chosen). Q2.1 asks for something
different: retrieve top-K from the whole catalogue with A1's candidate
generator, then re-rank. These are different evaluation universes with
different ceilings - A1 measured circulating-pool recall@50 of just 0.079
(MIND) / 0.037 (EB-NeRD), so a retrieved candidate list misses most clicks
before a re-ranker ever sees it, while an inview list already contains every
click by construction. `CandidateSet` is written so the feature builder, the
re-rankers and the eval code depend on this shape and nothing else, and moving
between universes is a choice of constructor, not a second pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.retrieval.hybrid import flatten_candidates


@dataclass
class CandidateSet:
    """One flattened batch of impressions and their candidates.

    `offsets` is the flatten/regroup idiom already used by
    `hybrid.flatten_candidates`: impression i's candidates are
    `flat_ids[offsets[i]:offsets[i+1]]`. `flat_doc_rows` is -1 for a candidate
    id the article feature store has no row for (should not happen for
    `inview`, and is exactly the recall ceiling for `retrieved`).
    """
    impressions: pd.DataFrame        # the source rows, same order as offsets
    offsets: np.ndarray              # int64 (n_imp + 1,)
    flat_ids: list                   # article ids, len == offsets[-1]
    flat_doc_rows: np.ndarray        # int64, -1 for unknown
    labels_by_imp: list              # list[np.ndarray[int8]]
    universe: str                    # "inview" | "retrieved"

    @property
    def n_impressions(self) -> int:
        return len(self.offsets) - 1

    def ids_by_imp(self, i: int) -> list:
        return self.flat_ids[self.offsets[i]:self.offsets[i + 1]]


def _labels_by_imp(offsets: np.ndarray, flat_ids: list, clicked_sets: list[set]) -> list:
    out = []
    for i in range(len(offsets) - 1):
        lo, hi = offsets[i], offsets[i + 1]
        ids = flat_ids[lo:hi]
        out.append(np.fromiter((1 if a in clicked_sets[i] else 0 for a in ids),
                               dtype=np.int8, count=len(ids)))
    return out


def from_inview(impressions: pd.DataFrame, row_of: dict[str, int]) -> CandidateSet:
    """Universe A: the impression's own inview list - what the leaderboards score."""
    offsets, flat_ids, flat_doc_rows = flatten_candidates(impressions, row_of)
    clicked_sets = [set(c) for c in impressions["clicked_ids"]]
    labels = _labels_by_imp(offsets, flat_ids, clicked_sets)
    return CandidateSet(impressions, offsets, flat_ids, flat_doc_rows, labels, "inview")


def from_retrieval(impressions: pd.DataFrame, retrieved_ids_by_user: dict[str, list],
                   row_of: dict[str, int]) -> CandidateSet:
    """Universe B: stage-1's retrieved top-K, keyed per user by `UnionRetriever`.

    Every impression from the same user gets the same candidate list (stage 1
    depends on the user's history and the pool, not on which impression is
    being scored), matching how `retriever.UnionRetriever.retrieve` is called
    once per unique user.
    """
    lengths = np.array([len(retrieved_ids_by_user.get(u, [])) for u in impressions["user_id"]])
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    flat_ids = [a for u in impressions["user_id"] for a in retrieved_ids_by_user.get(u, [])]
    flat_doc_rows = np.array([row_of.get(a, -1) for a in flat_ids], dtype=np.int64)
    clicked_sets = [set(c) for c in impressions["clicked_ids"]]
    labels = _labels_by_imp(offsets, flat_ids, clicked_sets)
    return CandidateSet(impressions, offsets, flat_ids, flat_doc_rows, labels, "retrieved")
