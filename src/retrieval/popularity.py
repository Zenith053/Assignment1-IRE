"""Popularity baseline.

Not optional. News clicking is dominated by recency and popularity, so a
content-based retriever that cannot beat "show the most-clicked articles" is
not earning its complexity. Every reported metric is read against this floor.

Popularity is counted on the **train** split only (see feature_store.py) and
applied unchanged to val and test, so it carries no future information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PopularityRanker:
    """Scores an article by its train-split click count."""

    def __init__(self, articles: pd.DataFrame):
        self.scores = dict(zip(articles["article_id"], articles["train_clicks"]))
        # Rank order is fixed, so top-K is the same list for every user.
        ordered = articles.sort_values(["train_clicks", "article_id"],
                                       ascending=[False, True])
        self.ranked_ids = ordered["article_id"].to_numpy()
        self.ranked_counts = ordered["train_clicks"].to_numpy()

    def top_k(self, k: int, allowed: set[str] | None = None) -> np.ndarray:
        """Most-clicked articles, optionally restricted to a candidate pool."""
        if allowed is None:
            return self.ranked_ids[:k]
        mask = np.fromiter((a in allowed for a in self.ranked_ids),
                           dtype=bool, count=len(self.ranked_ids))
        return self.ranked_ids[mask][:k]

    def score_articles(self, article_ids) -> np.ndarray:
        """Per-article scores for ranking within an impression."""
        return np.fromiter((self.scores.get(a, 0) for a in article_ids),
                           dtype=np.float32, count=len(article_ids))
