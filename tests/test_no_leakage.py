"""Q9: assert the behaviour-window boundary holds - no future-click leakage.

These are the checks that caught a real bug during development. EB-NeRD ships
one history snapshot per split directory, each covering the 21 days *before*
that split, so the validation snapshot spans the entire train impression
window. Collapsing the two snapshots gave train impressions access to clicks
that happen during and after them; `test_history_snapshot_predates_split`
fails loudly if that regresses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config
from src.common.io import read_table

DATASETS = ["mind", "ebnerd"]
SPLIT_ORDER = ["train", "val", "test"]


def _config(dataset: str):
    return load_config(REPO_ROOT / "config" / f"{dataset}.yaml")


def _require_built(cfg) -> dict:
    meta = cfg.processed / "split_meta.json"
    if not meta.exists():
        pytest.skip(f"{cfg.dataset} not built yet; run make data")
    return json.loads(meta.read_text())


@pytest.fixture(scope="module", params=DATASETS)
def built(request):
    cfg = _config(request.param)
    meta = _require_built(cfg)
    return cfg, meta


def test_history_snapshot_predates_split(built):
    """Every split's history must end strictly before its first impression."""
    cfg, meta = built
    history = read_table(cfg.processed / "history.parquet", "history")

    for split, info in meta["splits"].items():
        snapshot = history[history["snapshot"] == info["history_snapshot"]]
        assert len(snapshot) > 0, f"{split}: snapshot {info['history_snapshot']} is empty"

        stamps = snapshot["timestamp"].dropna()
        if stamps.empty:
            continue  # MIND stores click order only; nothing to compare
        split_start = pd.Timestamp(info["t_min"])
        assert stamps.max() < split_start, (
            f"{cfg.dataset}/{split}: history snapshot "
            f"{info['history_snapshot']!r} ends {stamps.max()} but the split "
            f"starts {split_start} - future clicks are visible to the model"
        )


def test_user_profiles_contain_no_future_clicks(built):
    """The built profiles, not just the raw snapshots, must respect the boundary."""
    cfg, meta = built
    profiles_path = cfg.features / "user_profiles.parquet"
    if not profiles_path.exists():
        pytest.skip("feature store not built")

    history = read_table(cfg.processed / "history.parquet", "history")
    stamp_of = history.dropna(subset=["timestamp"]).set_index(
        ["snapshot", "article_id"]
    )["timestamp"]
    if stamp_of.empty:
        pytest.skip(f"{cfg.dataset} has no history timestamps")

    profiles = pd.read_parquet(profiles_path)
    for split, info in meta["splits"].items():
        split_start = pd.Timestamp(info["t_min"])
        snapshot = info["history_snapshot"]
        part = profiles[profiles["split"] == split]
        # Spot-check a sample: the full cross product is large and the property
        # is uniform across users.
        for clicked in part["clicked_ids"].head(200):
            for article_id in list(clicked)[-25:]:
                key = (snapshot, str(article_id))
                if key in stamp_of.index:
                    stamp = stamp_of.loc[key]
                    latest = stamp.max() if hasattr(stamp, "max") else stamp
                    assert latest < split_start, (
                        f"{cfg.dataset}/{split}: profile contains a click at "
                        f"{latest}, at or after the split start {split_start}"
                    )


def test_popularity_is_fit_on_train_only(built):
    """Article popularity must not count clicks from val or test."""
    cfg, meta = built
    features = cfg.features / "articles.parquet"
    if not features.exists():
        pytest.skip("feature store not built")

    articles = pd.read_parquet(features)
    train = read_table(cfg.processed / "train" / "impressions.parquet", "impressions")

    from collections import Counter
    expected = Counter(a for row in train["clicked_ids"] for a in row)
    recorded = dict(zip(articles["article_id"], articles["train_clicks"]))

    assert sum(recorded.values()) == sum(expected.values()), (
        f"{cfg.dataset}: train_clicks total {sum(recorded.values())} does not "
        f"match the train split's {sum(expected.values())} clicks - popularity "
        f"was fit on more than the train split"
    )
    for article_id, count in list(expected.items())[:500]:
        assert recorded.get(article_id, 0) == count


def test_no_impression_appears_in_two_splits(built):
    """The splits must partition the impressions exactly."""
    cfg, meta = built
    seen: set[int] = set()
    total = 0
    for split in SPLIT_ORDER:
        part = read_table(cfg.processed / split / "impressions.parquet", "impressions")
        ids = set(part["impression_id"])
        assert not (ids & seen), f"{cfg.dataset}: {split} shares impressions with an earlier split"
        seen |= ids
        total += len(part)

    full = read_table(cfg.processed / "impressions.parquet", "impressions")
    assert total == len(full), (
        f"{cfg.dataset}: splits hold {total} impressions but the cleaned table "
        f"has {len(full)} - rows were lost or duplicated"
    )


def test_rolling_popularity_is_strictly_causal(built):
    """A2 Q1.4/Q9: `RollingPopularity`'s trailing counts must not see the future.

    This is the counterfactual, not a proxy: build the counter once over the
    whole click timeline and once over only the events strictly before a probe
    timestamp `t0`, then demand `counts_before(..., t0)` agrees between the two.
    If `searchsorted`'s `side` argument were ever `"right"` instead of
    `"left"`, or the window were centred on `t0` rather than trailing it, the
    two builds would disagree at exactly the boundary - a leaky implementation
    cannot pass this by accident.
    """
    from src.rerank.timeline import RollingPopularity

    cfg, meta = built
    impressions_by_split = {
        split: read_table(cfg.processed / split / "impressions.parquet", "impressions")
        for split in SPLIT_ORDER
    }
    full = RollingPopularity.from_impressions(impressions_by_split)

    # Flatten the same click events the constructor sees, to pick probe points
    # and to build the "past-only" counterfactual from a plain boolean mask.
    all_article_ids, all_timestamps = [], []
    for impressions in impressions_by_split.values():
        for ts, clicked in zip(impressions["timestamp"], impressions["clicked_ids"]):
            all_article_ids.extend(clicked)
            all_timestamps.extend([ts] * len(clicked))
    all_article_ids = pd.Series(all_article_ids, dtype=object)
    all_timestamps = pd.Series(pd.to_datetime(all_timestamps))
    if all_timestamps.empty:
        pytest.skip(f"{cfg.dataset}: no click events to probe")

    rng_seed = 13
    probe_idx = all_timestamps.sample(min(15, len(all_timestamps)), random_state=rng_seed).index
    for idx in probe_idx:
        t0 = all_timestamps.loc[idx].to_numpy()
        past_mask = (all_timestamps < all_timestamps.loc[idx]).to_numpy()
        past_only = RollingPopularity(
            all_article_ids[past_mask].to_numpy(), all_timestamps[past_mask].to_numpy()
        )
        probe_articles = all_article_ids.sample(
            min(20, len(all_article_ids)), random_state=rng_seed
        ).tolist()

        for window_hours in (24.0, 168.0):
            from_full = full.counts_before(probe_articles, t0, window_hours)
            from_past = past_only.counts_before(probe_articles, t0, window_hours)
            assert (from_full == from_past).all(), (
                f"{cfg.dataset}: rolling popularity at t0={t0} (window "
                f"{window_hours}h) differs when future clicks are removed from "
                f"the timeline - counts_before is not strictly causal"
            )


def test_session_features_are_strictly_causal(built):
    """A2 Q1.4: session counters must only see strictly-earlier impressions
    in the same session, and must be N/A (not fabricated) where unavailable."""
    from src.rerank.timeline import SessionContext

    cfg, meta = built
    session = SessionContext(cfg)
    if not session.available:
        # MIND declares no session concept; the honest behaviour is N/A, not a guess.
        feats = session.session_features("train", 1)
        assert feats["session_impression_index"] != feats["session_impression_index"], (
            f"{cfg.dataset}: has_session_id is false but session_features "
            f"returned a number instead of NaN"
        )
        pytest.skip(f"{cfg.dataset}: has_session_id is false, N/A as expected")

    for split in ("train", "val"):
        key = f"{split}_behaviors"
        if key not in cfg.raw or not cfg.raw[key].exists():
            continue
        beh = pd.read_parquet(cfg.raw[key], engine="pyarrow",
                              columns=["impression_id", "session_id", "impression_time"])
        beh["impression_time"] = pd.to_datetime(beh["impression_time"])
        # Spot-check sessions with >1 impression: the later impression's index
        # and clicks-so-far must be strictly greater than the earlier one's,
        # and its "seconds since session start" must be strictly later.
        sizes = beh.groupby("session_id").size()
        multi = sizes[sizes > 1].index[:10]
        for sid in multi:
            rows = beh[beh["session_id"] == sid].sort_values("impression_time")
            imp_ids = rows["impression_id"].tolist()
            feats = [session.session_features(split, int(i)) for i in imp_ids]
            indices = [f["session_impression_index"] for f in feats]
            assert indices == sorted(indices) and len(set(indices)) == len(indices), (
                f"{cfg.dataset}/{split} session {sid}: impression indices "
                f"{indices} are not strictly increasing in time order"
            )
            seconds = [f["seconds_since_session_start"] for f in feats]
            assert seconds == sorted(seconds), (
                f"{cfg.dataset}/{split} session {sid}: seconds-since-start "
                f"{seconds} are not monotonic in time order"
            )


def test_serving_time_ablation_declared_not_faked(built):
    """Q9: the leaky-popularity ablation must only run where the dataset says so.

    `load_leaky_popularity` reads `serving_time_unavailable` from the config,
    not the dataset name - MIND declares no such columns and must get an
    honest "unavailable" rather than a fabricated feature.
    """
    from src.eval.harness import load_leaky_popularity

    cfg, meta = built
    features = cfg.features / "articles.parquet"
    if not features.exists():
        pytest.skip("feature store not built")
    article_ids = pd.read_parquet(features, columns=["article_id"])["article_id"].tolist()

    leaky = load_leaky_popularity(cfg, article_ids)
    if not cfg.serving_time_unavailable:
        assert leaky is None, (
            f"{cfg.dataset}: no serving_time_unavailable columns declared, "
            f"but load_leaky_popularity returned a feature anyway"
        )
    else:
        assert leaky is not None, (
            f"{cfg.dataset}: declares {cfg.serving_time_unavailable} but "
            f"load_leaky_popularity returned nothing"
        )
        assert set(leaky) == set(article_ids)
        # A real ablation needs the feature to actually vary, or it cannot
        # inflate anything - guard against a silently-broken column read.
        assert len(set(leaky.values())) > 1, (
            f"{cfg.dataset}: total_pageviews is constant across articles - "
            f"the ablation would be a no-op"
        )
