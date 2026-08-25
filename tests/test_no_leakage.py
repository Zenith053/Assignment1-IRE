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
