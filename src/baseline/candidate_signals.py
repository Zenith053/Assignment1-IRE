"""Q3 step 3: the per-candidate numbers the improved NRMS adds - popularity and freshness.

NRMS reads only titles, so it cannot tell a 2-hour-old article everyone is
clicking from a 5-day-old one nobody is. These signals give it exactly that,
and nothing else:

  popularity  log1p(clicks in the trailing 1h / 24h / 168h before the impression)
  freshness   log1p(hours since the article was published)      [EB-NeRD only]

Leakage: the click timeline is every click in train/val/test (the same event
set `RollingPopularity` uses), but each count only includes clicks with
timestamp strictly before the impression's own - the impression's own click is
never counted. `counts_before` below is a vectorised equivalent of
`src/rerank/timeline.RollingPopularity.counts_before`, and
`tests/test_candidate_signals.py` checks the two agree on real data, so it
inherits that class's counterfactual causality test rather than asserting
causality a second time by a different route.

Freshness: 0.03% of EB-NeRD train candidates carry a publish time up to 0.7h
after the impression (clock skew between systems, never more than ~9h in
test); age is clamped to 0 rather than letting a negative age hint at the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.config import Config
from src.common.io import read_table

POPULARITY_WINDOWS_HOURS = (1.0, 24.0, 168.0)


class ClickTimeline:
    """All click events as (article row, second) pairs sorted on one int64 key.

    Key = article_row * 2^32 + seconds_since_first_event, so every article's
    clicks are one contiguous, time-sorted run, and a windowed count for any
    number of (article, t) queries is two `searchsorted` calls in total.
    """

    SHIFT = np.int64(2 ** 32)

    def __init__(self, article_rows: np.ndarray, timestamps: np.ndarray):
        ts = np.asarray(timestamps).astype("datetime64[s]")
        if len(ts) and not np.array_equal(ts.astype(np.asarray(timestamps).dtype),
                                          np.asarray(timestamps)):
            raise ValueError("timestamps must be whole seconds for the packed key")
        self.origin = ts.min() if len(ts) else np.datetime64(0, "s")
        seconds = (ts - self.origin).astype(np.int64)
        if len(seconds) and seconds.max() >= self.SHIFT:
            raise ValueError("timeline spans more than 2^32 seconds")
        self.keys = np.sort(np.asarray(article_rows, dtype=np.int64) * self.SHIFT + seconds)

    @classmethod
    def from_impressions(cls, impressions_by_split: dict[str, pd.DataFrame],
                         row_of: dict[str, int]) -> "ClickTimeline":
        rows, stamps = [], []
        for imps in impressions_by_split.values():
            for ts, clicked in zip(imps["timestamp"].to_numpy(), imps["clicked_ids"]):
                for a in clicked:
                    if a in row_of:
                        rows.append(row_of[a])
                        stamps.append(ts)
        return cls(np.asarray(rows, dtype=np.int64), np.asarray(stamps, dtype="datetime64[us]"))

    def _query_seconds(self, t: np.ndarray) -> np.ndarray:
        # Ceil to whole seconds: an event at second s is "before t" iff s < t,
        # which for whole-second events is s < ceil(t). Keys are whole seconds.
        t_us = np.asarray(t).astype("datetime64[us]")
        return ((t_us - self.origin.astype("datetime64[us]")).astype(np.int64) + 999_999) // 1_000_000

    def counts_before(self, article_rows: np.ndarray, t: np.ndarray, window_hours: float) -> np.ndarray:
        """Clicks on each article in [t - window, t): strictly before t, same bounds as RollingPopularity."""
        rows = np.asarray(article_rows, dtype=np.int64)
        hi_s = self._query_seconds(t)
        lo_s = self._query_seconds(np.asarray(t).astype("datetime64[us]")
                                   - np.timedelta64(int(round(window_hours * 3600)), "s"))
        # Queries before the origin clamp to 0 seconds; nothing precedes the first event anyway.
        hi = np.searchsorted(self.keys, rows * self.SHIFT + np.maximum(hi_s, 0), side="left")
        lo = np.searchsorted(self.keys, rows * self.SHIFT + np.maximum(lo_s, 0), side="left")
        return (hi - lo).astype(np.int32)


def signal_names(popularity: bool, freshness: bool) -> list[str]:
    names = [f"log1p_clicks_{int(w)}h" for w in POPULARITY_WINDOWS_HOURS] if popularity else []
    return names + (["log1p_age_hours"] if freshness else [])


def build_timeline(cfg: Config, row_of: dict[str, int]) -> ClickTimeline:
    """Every click in all three splits, never sampled - a sampled timeline would undercount."""
    return ClickTimeline.from_impressions(
        {s: read_table(cfg.processed / s / "impressions.parquet", "impressions")
         for s in ("train", "val", "test")},
        row_of,
    )


def candidate_signals(cfg: Config, tensors, article_ids: np.ndarray, timeline: ClickTimeline | None,
                      popularity: bool, freshness: bool) -> np.ndarray:
    """(C, F) float32 aligned with `tensors.cand_rows`."""
    if freshness and not cfg.can("has_published_time"):
        raise ValueError(f"{cfg.dataset}: freshness requested but has_published_time is false")

    imp_of_cand = np.repeat(np.arange(tensors.n_impressions), np.diff(tensors.offsets))
    t = tensors.impressions["timestamp"].to_numpy().astype("datetime64[us]")[imp_of_cand]
    cols = []
    if popularity:
        for w in POPULARITY_WINDOWS_HOURS:
            cols.append(np.log1p(timeline.counts_before(tensors.cand_rows, t, w)))
    if freshness:
        articles = read_table(cfg.processed / "articles.parquet", "articles")
        pub_of = dict(zip(articles["article_id"].astype(str), articles["published_time"]))
        pub = pd.to_datetime(pd.Series([pub_of.get(a) for a in article_ids])).to_numpy()
        age_h = (t - pub[tensors.cand_rows].astype("datetime64[us]")) / np.timedelta64(1, "h")
        age_h = np.where(np.isnan(age_h), 0.0, np.maximum(age_h, 0.0))
        cols.append(np.log1p(age_h))
    if not cols:
        return np.zeros((len(tensors.cand_rows), 0), dtype=np.float32)
    return np.stack(cols, axis=1).astype(np.float32)
