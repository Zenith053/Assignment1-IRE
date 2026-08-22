"""Candidate pool selection, shared by the lexical and semantic retrievers.

Recall@K depends enormously on what the retriever is allowed to return, so the
pool is an explicit, reported choice rather than an implicit one:

    all          every article in the catalogue. Honest but pessimistic - most
                 of the catalogue is old news that no serving system would
                 consider. This is the headline number.
    circulating  articles that appeared in at least one impression during the
                 split. Closest to a real serving pool. NOTE: derived from the
                 evaluation split itself, so it is an optimistic bound, not a
                 deployable filter. Always reported alongside `all`.
    fresh        articles published before the split ends. Needs
                 `has_published_time`, so EB-NeRD only; MIND reports N/A.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

POOLS = ("all", "circulating", "fresh")


def build_pool(name: str, articles: pd.DataFrame, impressions: pd.DataFrame,
               has_published_time: bool) -> tuple[np.ndarray, str | None]:
    """Return row indices of the allowed articles, or a reason it is unavailable."""
    if name == "all":
        return np.arange(len(articles)), None

    if name == "circulating":
        seen = {a for row in impressions["inview_ids"] for a in row}
        mask = articles["article_id"].isin(seen).to_numpy()
        return np.flatnonzero(mask), None

    if name == "fresh":
        if not has_published_time:
            return np.array([], dtype=int), "dataset has no published_time"
        cutoff = impressions["timestamp"].max()
        published = pd.to_datetime(articles["published_time"], errors="coerce")
        # Anything published after the window closed cannot have been shown.
        mask = (published <= cutoff).fillna(False).to_numpy()
        return np.flatnonzero(mask), None

    raise ValueError(f"unknown pool {name!r}; expected one of {POOLS}")
