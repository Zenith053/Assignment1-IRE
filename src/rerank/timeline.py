"""Strictly-causal aggregates over click events: rolling popularity and session context.

A1's `train_clicks` (feature_store.py) is a single scalar fit once on the train
split and frozen - deliberately, so it never sees an evaluation-split click. The
aggregates here are a different, complementary kind of feature: a trailing
count "as of this impression's own timestamp", recomputed per query rather than
fit once. That is safe for the same reason a live production counter is safe -
it only ever looks backward from `t` - but the mechanism is different enough
(a per-article event timeline, queried by binary search) that it needs its own
leakage test rather than inheriting `train_clicks`'s. See
`tests/test_no_leakage.py::test_rolling_popularity_is_strictly_causal`, which is
a counterfactual: truncating the timeline to "everything before t" must give
the exact same answer as searching the full timeline for "before t", or the
window is not actually trailing.

Session features are EB-NeRD-only (`cfg.can("has_session_id")`; MIND's raw
schema has no session concept at all) and read three raw, EB-NeRD-specific
columns that A1's `clean.py` drops: `session_id`, the context `article_id` (the
article the user was reading when the impression fired), and the per-click
`read_time_fixed`/`scroll_percentage_fixed` dwell-time pair in `history.parquet`.
Reading raw parquet here rather than extending the processed schema avoids a
full data rebuild (`io.validate_table`'s exact-column contract would otherwise
force re-cleaning both datasets); it follows the precedent already set by
`harness.load_leaky_popularity`, which does the same thing for the Q9 ablation.

One distinction matters for anti-gaming and is worth stating plainly: the
current impression's own `read_time` is the *outcome* of the click being
predicted and is correctly quarantined (`config/ebnerd.yaml`'s
`serving_time_unavailable`); a user's *past* dwell time on articles they
already clicked, before this impression ever fired, is an ordinary behavioural
feature and is what `HistoricalDwellTime` below computes - it is never the
current article's read time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.config import Config

# Trailing windows reported by default; short enough to matter for a breaking
# news spike, long enough to smooth over a quiet news day.
DEFAULT_WINDOWS_HOURS = (24.0, 168.0)


class RollingPopularity:
    """Per-article click counts in a trailing window ending strictly before `t`.

    Built once over every click event this pipeline has record of (all three
    splits - the timeline itself is not evaluation data, only a per-query
    trailing count of it is a feature, and that count only ever reads
    backward). Each article's click timestamps are kept as a single sorted
    array so a windowed count is two `searchsorted` calls, not a scan.
    """

    def __init__(self, article_ids: np.ndarray, timestamps: np.ndarray):
        order = np.argsort(timestamps, kind="mergesort")  # stable: ties keep event order
        ts_sorted = np.asarray(timestamps)[order]
        aid_sorted = np.asarray(article_ids, dtype=object)[order]
        # Grouping a globally time-sorted array preserves per-group order, so
        # every article's array comes out already ascending - no per-group sort.
        frame = pd.DataFrame({"article_id": aid_sorted, "ts": ts_sorted})
        self._times: dict[str, np.ndarray] = {
            aid: grp["ts"].to_numpy() for aid, grp in frame.groupby("article_id", sort=False)
        }

    @classmethod
    def from_impressions(cls, impressions_by_split: dict[str, pd.DataFrame]) -> "RollingPopularity":
        """Build from every clicked (article, timestamp) pair across the given splits.

        A click's timestamp is its impression's timestamp - both datasets carry
        a real one there (MIND's history clicks are timestamp-free, but
        impressions never are; split.py could not do a temporal split otherwise).
        """
        article_ids, timestamps = [], []
        for impressions in impressions_by_split.values():
            for ts, clicked in zip(impressions["timestamp"], impressions["clicked_ids"]):
                article_ids.extend(clicked)
                timestamps.extend([ts] * len(clicked))
        return cls(np.asarray(article_ids, dtype=object), np.asarray(timestamps))

    def counts_before(self, article_ids, t, window_hours: float) -> np.ndarray:
        """Clicks on each article in (t - window_hours, t), i.e. strictly before `t`.

        `side="left"` on the upper bound is the entire leakage guarantee: it
        counts events with timestamp < t, never <= t, so an event at exactly
        the query time (the impression's own click, if any) is excluded.
        """
        lower = t - np.timedelta64(int(round(window_hours * 3600)), "s")
        out = np.zeros(len(article_ids), dtype=np.int32)
        for i, aid in enumerate(article_ids):
            arr = self._times.get(aid)
            if arr is None or len(arr) == 0:
                continue
            hi = np.searchsorted(arr, t, side="left")
            lo = np.searchsorted(arr, lower, side="left")
            out[i] = hi - lo
        return out

    def counts_multi_window(self, article_ids, t, windows_hours=DEFAULT_WINDOWS_HOURS) -> dict[float, np.ndarray]:
        return {w: self.counts_before(article_ids, t, w) for w in windows_hours}


class SessionContext:
    """Within-session features and historical dwell time, EB-NeRD only.

    `available` is False on any dataset that does not declare `has_session_id`
    (MIND) - callers must check it and emit NaN rather than guess a value, the
    same `N/A`-not-faked convention A1 uses for `published_time`/`fresh` pools.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.available = cfg.can("has_session_id")
        # keyed by (source_split, source_impression_id) -> dict of session fields
        self._by_impression: dict[tuple[str, int], dict] = {}
        # keyed by (user_id, split_name) -> dict of historical dwell-time means
        self._dwell_by_user_split: dict[tuple[str, str], dict] = {}
        if self.available:
            self._build_session_index()

    def _build_session_index(self) -> None:
        """Sort each raw split's behaviors by (session, time) and derive causal counters.

        `session_clicks_so_far` and `session_impression_index` only count
        impressions that are strictly earlier in the same session - the ordering
        A1's `position` column already relies on for history, applied here to
        one session's impressions instead of one user's whole history.
        """
        for split in ("train", "val"):
            key = f"{split}_behaviors"
            if key not in self.cfg.raw or not self.cfg.raw[key].exists():
                continue
            beh = pd.read_parquet(
                self.cfg.raw[key], engine="pyarrow",
                columns=["impression_id", "session_id", "article_id",
                        "impression_time", "article_ids_clicked"],
            )
            beh["session_id"] = beh["session_id"].astype(str)
            beh["impression_time"] = pd.to_datetime(beh["impression_time"])
            beh["n_clicks"] = beh["article_ids_clicked"].map(
                lambda v: 0 if v is None else len(v)
            )
            beh = beh.sort_values(["session_id", "impression_time", "impression_id"])

            grouped = beh.groupby("session_id", sort=False)
            session_start = grouped["impression_time"].transform("min")
            # cumcount/cumsum over the sorted frame = "how many came before me
            # in this session"; shift(1) so the current impression's own click
            # is excluded, matching the searchsorted<t convention above.
            impression_index = grouped.cumcount()
            clicks_so_far = grouped["n_clicks"].cumsum() - beh["n_clicks"]
            seconds_since_start = (beh["impression_time"] - session_start).dt.total_seconds()

            for imp_id, ctx_article, idx, clicks, secs in zip(
                beh["impression_id"], beh["article_id"], impression_index,
                clicks_so_far, seconds_since_start,
            ):
                self._by_impression[(split, int(imp_id))] = {
                    "session_impression_index": int(idx),
                    "session_clicks_so_far": int(clicks),
                    "seconds_since_session_start": float(secs),
                    "context_article_id": None if pd.isna(ctx_article) else str(ctx_article),
                }

        for split in ("train", "val"):
            key = f"{split}_history"
            if key not in self.cfg.raw or not self.cfg.raw[key].exists():
                continue
            hist = pd.read_parquet(
                self.cfg.raw[key], engine="pyarrow",
                columns=["user_id", "read_time_fixed", "scroll_percentage_fixed"],
            )
            for user_id, read_times, scrolls in zip(
                hist["user_id"].astype(str), hist["read_time_fixed"], hist["scroll_percentage_fixed"]
            ):
                rt = np.asarray(read_times, dtype=np.float64) if read_times is not None else np.array([])
                sc = np.asarray(scrolls, dtype=np.float64) if scrolls is not None else np.array([])
                rt, sc = rt[~np.isnan(rt)], sc[~np.isnan(sc)]
                self._dwell_by_user_split[(user_id, split)] = {
                    "mean_hist_read_time": float(rt.mean()) if len(rt) else np.nan,
                    "mean_hist_scroll_pct": float(sc.mean()) if len(sc) else np.nan,
                }

    def session_features(self, source_split: str, source_impression_id: int) -> dict:
        """Session-relative features for one impression, or all-NaN if unavailable."""
        if not self.available:
            return {"session_impression_index": np.nan, "session_clicks_so_far": np.nan,
                   "seconds_since_session_start": np.nan, "context_article_id": None}
        found = self._by_impression.get((source_split, int(source_impression_id)))
        if found is None:
            return {"session_impression_index": np.nan, "session_clicks_so_far": np.nan,
                   "seconds_since_session_start": np.nan, "context_article_id": None}
        return found

    def dwell_features(self, user_id: str, history_snapshot: str) -> dict:
        """Mean past read-time/scroll-percentage for a user, from the snapshot
        `split.py` already certified as ending before the split in question -
        the same snapshot `feature_store.build_user_profiles` uses, just with
        the two dwell-time columns `clean.py` otherwise drops."""
        if not self.available:
            return {"mean_hist_read_time": np.nan, "mean_hist_scroll_pct": np.nan}
        return self._dwell_by_user_split.get(
            (user_id, history_snapshot),
            {"mean_hist_read_time": np.nan, "mean_hist_scroll_pct": np.nan},
        )
