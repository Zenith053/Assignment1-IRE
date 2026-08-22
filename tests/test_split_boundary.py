"""Temporal split boundaries must be ordered, disjoint and non-random."""

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


@pytest.fixture(scope="module", params=DATASETS)
def built(request):
    cfg = load_config(REPO_ROOT / "config" / f"{request.param}.yaml")
    meta_path = cfg.processed / "split_meta.json"
    if not meta_path.exists():
        pytest.skip(f"{request.param} not built yet; run make data")
    return cfg, json.loads(meta_path.read_text())


def test_windows_are_ordered_and_disjoint(built):
    """train must end before val begins, and val before test."""
    cfg, meta = built
    for earlier, later in zip(SPLIT_ORDER, SPLIT_ORDER[1:]):
        end = pd.Timestamp(meta["splits"][earlier]["t_max"])
        start = pd.Timestamp(meta["splits"][later]["t_min"])
        assert end < start, (
            f"{cfg.dataset}: {earlier} ends {end} but {later} starts {start}"
        )


def test_split_is_temporal_not_random(built):
    """Every impression in a split must lie inside that split's window.

    A random split would interleave timestamps across splits, so this fails
    immediately if someone swaps in a shuffle.
    """
    cfg, meta = built
    for split in SPLIT_ORDER:
        part = read_table(cfg.processed / split / "impressions.parquet", "impressions")
        lo = pd.Timestamp(meta["splits"][split]["t_min"])
        hi = pd.Timestamp(meta["splits"][split]["t_max"])
        assert part["timestamp"].min() >= lo
        assert part["timestamp"].max() <= hi


def test_test_split_comes_from_the_held_out_file(built):
    """The test split must be exactly the held-out source file."""
    cfg, meta = built
    held_out = cfg.split["held_out_source"]
    test = read_table(cfg.processed / "test" / "impressions.parquet", "impressions")
    assert set(test["source_split"].unique()) == {held_out}

    for split in ("train", "val"):
        part = read_table(cfg.processed / split / "impressions.parquet", "impressions")
        assert held_out not in set(part["source_split"].unique()), (
            f"{cfg.dataset}: {split} contains rows from the held-out file"
        )


def test_clicked_is_subset_of_inview(built):
    """A clicked article that was never shown means the label join is broken."""
    cfg, _ = built
    for split in SPLIT_ORDER:
        part = read_table(cfg.processed / split / "impressions.parquet", "impressions")
        for inview, clicked in zip(part["inview_ids"].head(5000),
                                   part["clicked_ids"].head(5000)):
            assert set(clicked) <= set(inview)
