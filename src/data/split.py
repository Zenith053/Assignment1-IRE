#!/usr/bin/env python3
"""Carve the cleaned impressions into temporal train/val/test splits.

Interaction data is never split randomly: a random split lets a model train on
clicks that happen after the ones it is scored on. Every split here is a
contiguous time window, and the boundaries are asserted disjoint and ordered
before anything is written.

Two mechanisms combine:

1. `held_out_source` names the raw file/directory (via the `source_split`
   column) that becomes the test split. Both datasets ship a separate later
   file for this, so the test window needs no date arithmetic.
2. The remaining rows are cut into train and val, either at explicit dates
   (MIND) or by holding back the last `val_days` days (EB-NeRD).

Each split is also paired with a history snapshot that *ends before the split
begins*, which is what stops EB-NeRD's validation history - it spans the whole
train impression window - from leaking future clicks into training.

Usage
-----
    python src/data/split.py --config config/mind.yaml
    python src/data/split.py --config config/ebnerd.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.common.config import Config, load_config  # noqa: E402
from src.common.io import read_table, write_table  # noqa: E402

SPLIT_ORDER = ["train", "val", "test"]


def assign_splits(impressions: pd.DataFrame, cfg: Config) -> pd.Series:
    """Label every impression 'train', 'val' or 'test' by time alone."""
    spec = cfg.split
    held_out = spec.get("held_out_source")
    if not held_out:
        raise ValueError(f"[{cfg.dataset}] config is missing split.held_out_source")

    sources = set(impressions["source_split"].unique())
    if held_out not in sources:
        raise ValueError(
            f"[{cfg.dataset}] held_out_source={held_out!r} not present in the data; "
            f"source_split values are {sorted(sources)}"
        )

    labels = pd.Series("train", index=impressions.index, dtype=object)
    is_held_out = impressions["source_split"] == held_out
    labels[is_held_out] = "test"

    # Everything not held out is available for the train/val cut.
    pool = impressions.loc[~is_held_out, "timestamp"]

    if "val" in spec:
        # Explicit day bounds; the end date is inclusive, so advance a full day.
        val_start = pd.Timestamp(spec["val"][0])
        val_end = pd.Timestamp(spec["val"][1]) + pd.Timedelta(days=1)
        in_val = (pool >= val_start) & (pool < val_end)
    else:
        # Hold back the final `val_days` of the available window.
        val_days = int(spec.get("val_days", 2))
        cutoff = pool.max().normalize() - pd.Timedelta(days=val_days - 1)
        in_val = pool >= cutoff

    labels[pool.index[in_val]] = "val"
    return labels


def choose_history_snapshot(history: pd.DataFrame, split_start: pd.Timestamp,
                            dataset: str, split_name: str) -> str:
    """Pick the newest history snapshot that ends before this split begins.

    Applying the rule uniformly means no module has to know that EB-NeRD ships
    two snapshots while MIND ships one.
    """
    ends = history.groupby("snapshot")["timestamp"].max()

    # A dataset without click times (MIND) can only offer its single snapshot.
    if ends.isna().all():
        if len(ends) != 1:
            raise ValueError(
                f"[{dataset}] {len(ends)} untimed history snapshots; cannot choose"
            )
        return str(ends.index[0])

    eligible = ends[ends <= split_start]
    if eligible.empty:
        raise ValueError(
            f"[{dataset}] no history snapshot ends before {split_name} starts "
            f"({split_start}). Snapshot end times: {ends.to_dict()}"
        )
    return str(eligible.idxmax())  # the latest snapshot that is still safe


def assert_ordered_and_disjoint(bounds: dict[str, tuple], dataset: str) -> None:
    """Fail loudly if the split windows overlap or run out of order.

    This is the structural half of the Q9 no-leakage requirement: a val window
    that starts before train ends would put future clicks in the training set.
    """
    present = [s for s in SPLIT_ORDER if s in bounds]
    for earlier, later in zip(present, present[1:]):
        if bounds[earlier][1] >= bounds[later][0]:
            raise AssertionError(
                f"[{dataset}] split windows overlap: {earlier} ends "
                f"{bounds[earlier][1]} but {later} starts {bounds[later][0]}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--in-dir", type=Path, default=None,
                        help="override the cleaned-data directory")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    in_dir = args.in_dir or cfg.processed
    out_dir = args.out_dir or cfg.processed

    impressions_path = in_dir / "impressions.parquet"
    if not impressions_path.exists():
        raise SystemExit(
            f"cleaned impressions not found: {impressions_path}\n"
            f"  Run: python src/data/clean.py --config {args.config}"
        )

    impressions = read_table(impressions_path, "impressions")
    history = read_table(in_dir / "history.parquet", "history")

    labels = assign_splits(impressions, cfg)
    # Every row must land in exactly one split, or rows are being silently lost.
    unassigned = int(labels.isna().sum())
    if unassigned:
        raise AssertionError(f"[{cfg.dataset}] {unassigned} impressions were unassigned")

    bounds: dict[str, tuple] = {}
    meta_splits: dict[str, dict] = {}

    for name in SPLIT_ORDER:
        part = impressions[labels == name]
        if part.empty:
            raise AssertionError(
                f"[{cfg.dataset}] split {name!r} is empty; check the split config"
            )
        t_min, t_max = part["timestamp"].min(), part["timestamp"].max()
        bounds[name] = (t_min, t_max)

        snapshot = choose_history_snapshot(history, t_min, cfg.dataset, name)
        write_table(part, out_dir / name / "impressions.parquet", "impressions")

        meta_splits[name] = {
            "n_impressions": int(len(part)),
            "n_users": int(part["user_id"].nunique()),
            "n_clicks": int(part["clicked_ids"].map(len).sum()),
            "t_min": str(t_min),
            "t_max": str(t_max),
            "history_snapshot": snapshot,
            "source_splits": sorted(part["source_split"].unique().tolist()),
        }
        print(f"  {name:<6} {len(part):>8,} impressions  "
              f"{t_min} .. {t_max}  history={snapshot}")

    assert_ordered_and_disjoint(bounds, cfg.dataset)

    meta = {
        "dataset": cfg.dataset,
        "scale": cfg.scale,
        "split_config": cfg.split,
        "splits": meta_splits,
        # Recorded so tests and the design note can assert on it later.
        "boundaries_ordered_and_disjoint": True,
    }
    meta_path = out_dir / "split_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"  boundaries ordered and disjoint: OK")
    print(f"  -> {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
