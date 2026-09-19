"""Stage 1 of Q2: union A1's three candidate generators into one top-K list per user.

A1 measured BM25 and semantic winning on different datasets at retrieval time -
circulating-pool recall@50 is semantic 0.0790 / bm25 0.0474 on MIND but bm25
0.0367 / semantic 0.0239 on EB-NeRD (`reports/design_note.md` Sec.4) - so
neither arm is safe to drop, and their misses are not the same misses. This
takes the union rather than picking a winner.

The semantic arm here is mean-pooled, not A1's better top-5-pooled ranking
score: FAISS needs one query vector per user, and top-5 pooling produces a
score *function* of the candidate, not a point in embedding space, so it has
nothing to hand an index. That is a real ceiling on this stage, not an
oversight - it is exactly why Universe A (re-ranking the inview list, where
top-5 pooling is used as a feature) and Universe B (re-ranking this retrieved
list) are reported separately rather than pretending they measure one thing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.retrieval.bm25 import BM25Index, build_queries
from src.retrieval.pool import build_pool
from src.retrieval.popularity import PopularityRanker
from src.retrieval.semantic import build_user_vectors


class UnionRetriever:
    """BM25 top-`k_bm25` + FAISS top-`k_semantic` + popularity top-`k_pop`, deduped, capped at K."""

    def __init__(self, cfg, index: BM25Index, embeddings: np.ndarray, articles: pd.DataFrame,
                impressions: pd.DataFrame, row_of: dict[str, int],
                pool_name: str = "circulating", k_bm25: int = 80, k_semantic: int = 100,
                k_pop: int = 20):
        import faiss  # local import: only retrieval needs it, mirrors semantic.py

        # One OpenMP thread: faiss, torch and lightgbm each link a different libomp in
        # this environment, and multi-threaded faiss search segfaults once torch is loaded
        # (reproduced in evaluate_reranker.py after MLP training; see also Q4 scale_10x.py).
        faiss.omp_set_num_threads(1)

        self.index = index
        self.embeddings = embeddings
        self.articles = articles
        self.row_of = row_of
        self.k_bm25, self.k_semantic, self.k_pop = k_bm25, k_semantic, k_pop
        self.pool_name = pool_name

        pool_idx, unavailable = build_pool(pool_name, articles, impressions,
                                           cfg.can("has_published_time"))
        if unavailable:
            # `fresh` genuinely unavailable on a dataset without published_time;
            # fall back to `all` rather than silently returning nothing.
            pool_idx, _ = build_pool("all", articles, impressions, False)
        self.pool_idx = pool_idx

        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss_index.add(np.ascontiguousarray(embeddings[pool_idx]))

        self.popularity = PopularityRanker(articles)
        self._pop_allowed = set(articles["article_id"].to_numpy()[pool_idx])

    def retrieve(self, profiles: pd.DataFrame, articles: pd.DataFrame, k: int = 200
                ) -> dict[str, list[str]]:
        """One retrieved id list per user in `profiles`, deduplicated, capped at k."""
        article_ids = np.asarray(articles["article_id"])

        user_ids, token_lists = build_queries(profiles, articles)
        bm_hits = self.index.retrieve(self.index.query_matrix(token_lists),
                                      k=self.k_bm25, pool=self.pool_idx)

        _, user_vectors = build_user_vectors(profiles, self.row_of, self.embeddings, False, 5.0)
        _, local = self.faiss_index.search(np.ascontiguousarray(user_vectors), self.k_semantic)
        sem_hits = self.pool_idx[local]  # pool-local index -> catalogue row

        pop_ids = self.popularity.top_k(self.k_pop, allowed=self._pop_allowed)

        out: dict[str, list[str]] = {}
        for i, user_id in enumerate(user_ids):
            union = np.concatenate([
                article_ids[bm_hits[i]], article_ids[sem_hits[i]], pop_ids,
            ])
            # dict.fromkeys dedups while keeping first-seen order (bm25 first,
            # then semantic, then popularity) - order does not affect the
            # re-ranker, which will score every one of these independently.
            out[user_id] = list(dict.fromkeys(union.tolist()))[:k]
        return out
