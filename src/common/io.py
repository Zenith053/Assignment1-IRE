"""Unified schema definitions and the only place that touches parquet/TSV.

The three tables below are the contract between the dataset adapters in
`src/data/clean.py` and everything downstream. Once data is in this shape, no
module needs to know which dataset it came from.

Storage forms (see PLAN.md):
  articles     wide   - one row per article
  impressions  nested - one row per impression, candidate ids as list columns
  history      long   - one row per click event
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #

ARTICLE_COLUMNS = [
    "article_id",      # str, cast at the adapter boundary
    "title",
    "abstract",
    "body",            # empty string when the dataset has no body text
    "category",
    "subcategory",
    "entities",        # list[str]
    "published_time",  # datetime64[ns], NaT when unavailable
]

IMPRESSION_COLUMNS = [
    # MIND numbers impressions from 1 in *both* the train and dev files, so all
    # 73,152 dev ids collide with train ids. clean.py assigns a unique id here and
    # preserves the raw one below, which the Codabench submission must echo back.
    "impression_id",         # int64, unique within the dataset
    "source_impression_id",  # int64, the id as it appears in the raw file
    "user_id",               # str
    "timestamp",             # datetime64[ns]
    "inview_ids",            # list[str], the candidate set shown
    "clicked_ids",           # list[str], may hold several ids
    # Which raw file/directory the row came from. Only the adapter knows this,
    # and split.py needs it to carve off the held-out file without hardcoding
    # dataset names or re-reading raw inputs.
    "source_split",          # str: 'train' | 'dev' (MIND) | 'val' (EB-NeRD)
]

HISTORY_COLUMNS = [
    "user_id",         # str
    "article_id",      # str
    "timestamp",       # datetime64[ns], NaT for MIND
    "position",        # int32, 0-based order within the user's history
    # EB-NeRD ships one history snapshot per split directory, each covering the
    # 21 days *before* that split. The validation snapshot therefore spans the
    # entire train impression window, so collapsing the two would let train
    # impressions see future clicks. Keep both; split.py picks one per split.
    "snapshot",        # str: 'all' (MIND) | 'train' | 'val' (EB-NeRD)
]

TABLE_COLUMNS = {
    "articles": ARTICLE_COLUMNS,
    "impressions": IMPRESSION_COLUMNS,
    "history": HISTORY_COLUMNS,
}

# Columns holding Python lists; they need care on the parquet round trip.
LIST_COLUMNS = {"entities", "inview_ids", "clicked_ids"}


# --------------------------------------------------------------------------- #
# coercion helpers
# --------------------------------------------------------------------------- #

def to_str_list(value) -> list[str]:
    """Coerce any nested-cell representation to a plain list of strings.

    Parquet round trips hand back numpy arrays, pandas hands back lists, and
    missing values arrive as None or NaN. Normalising here keeps every caller
    from re-implementing the same three checks.
    """
    if value is None:
        return []
    # np.ndarray is truthy-ambiguous, so check it before any boolean test.
    if isinstance(value, np.ndarray):
        return [str(v) for v in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    # A scalar NaN is the "missing" marker for object columns.
    if isinstance(value, float) and np.isnan(value):
        return []
    return [str(value)]


def to_str_series(series: pd.Series) -> pd.Series:
    """Cast an id column to string, without turning missing values into 'nan'."""
    return series.astype("string").fillna("").astype(str)


# --------------------------------------------------------------------------- #
# read / write
# --------------------------------------------------------------------------- #

def write_table(df: pd.DataFrame, path: Path, name: str) -> Path:
    """Validate against the schema, then write one table as parquet."""
    validate_table(df, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    # pyarrow maps object-dtype list cells onto arrow list types.
    df.to_parquet(path, engine="pyarrow", index=False)
    return path


def read_table(path: Path, name: str | None = None) -> pd.DataFrame:
    """Read a table written by `write_table`, restoring list columns to lists."""
    df = pd.read_parquet(path, engine="pyarrow")
    # Arrow gives back numpy arrays for list types; normalise to plain lists.
    for col in df.columns:
        if col in LIST_COLUMNS:
            df[col] = df[col].map(to_str_list)
    if name:
        validate_table(df, name)
    return df


def validate_table(df: pd.DataFrame, name: str) -> None:
    """Assert a frame carries exactly the schema columns, in order."""
    expected = TABLE_COLUMNS.get(name)
    if expected is None:
        raise KeyError(f"unknown table {name!r}; known: {sorted(TABLE_COLUMNS)}")
    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]
    if missing or extra:
        raise ValueError(
            f"table {name!r} does not match the schema.\n"
            f"  missing: {missing}\n"
            f"  unexpected: {extra}"
        )


# --------------------------------------------------------------------------- #
# derived views
# --------------------------------------------------------------------------- #

def explode_impressions(impressions: pd.DataFrame) -> pd.DataFrame:
    """Materialise the long view: one row per (impression, candidate, label).

    Derived on demand for feature joins, popularity counts and leakage checks;
    never stored, because the nested form is the evaluation unit.
    """
    validate_table(impressions, "impressions")

    # Repeat each impression's scalar fields once per candidate, vectorised.
    counts = impressions["inview_ids"].map(len).to_numpy()
    long = pd.DataFrame({
        "impression_id": np.repeat(impressions["impression_id"].to_numpy(), counts),
        "user_id": np.repeat(impressions["user_id"].to_numpy(), counts),
        "timestamp": np.repeat(impressions["timestamp"].to_numpy(), counts),
        "article_id": np.concatenate(
            [np.asarray(v, dtype=object) for v in impressions["inview_ids"] if len(v)]
        ) if counts.sum() else np.array([], dtype=object),
    })

    # Label by membership in the clicked set of the same impression.
    clicked_pairs = {
        (imp, art)
        for imp, arts in zip(impressions["impression_id"], impressions["clicked_ids"])
        for art in arts
    }
    long["label"] = [
        1 if (imp, art) in clicked_pairs else 0
        for imp, art in zip(long["impression_id"], long["article_id"])
    ]
    return long


def summarise(df: pd.DataFrame, name: str) -> str:
    """One-line description used by the pipeline's stdout report."""
    return f"{name:<12} {len(df):>9,} rows  x {len(df.columns)} cols"
