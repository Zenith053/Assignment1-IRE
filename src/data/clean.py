#!/usr/bin/env python3
"""Parse both raw datasets into the one unified schema.

This module holds the *only* dataset-specific code in the pipeline. Each adapter
(`load_mind`, `load_ebnerd`) reads its native format and returns the same three
frames defined in `src/common/io.py`; everything downstream is dataset-agnostic
and must never branch on the dataset name.

Usage
-----
    python src/data/clean.py --config config/mind.yaml
    python src/data/clean.py --config config/ebnerd.yaml
    python src/data/clean.py --config config/ebnerd.yaml --limit 1000   # quick smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.common.config import Config, load_config  # noqa: E402
from src.common.io import (  # noqa: E402
    ARTICLE_COLUMNS,
    HISTORY_COLUMNS,
    IMPRESSION_COLUMNS,
    summarise,
    to_str_list,
    write_table,
)

# --------------------------------------------------------------------------- #
# MIND adapter
# --------------------------------------------------------------------------- #

# MIND ships headerless TSVs; these are the documented column orders.
MIND_NEWS_COLUMNS = [
    "article_id", "category", "subcategory", "title",
    "abstract", "url", "title_entities", "abstract_entities",
]
MIND_BEHAVIOR_COLUMNS = ["impression_id", "user_id", "time", "history", "impressions"]
MIND_TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"  # e.g. 11/11/2019 9:05:58 AM


def _read_mind_tsv(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read a headerless MIND TSV without letting pandas reinterpret the text.

    QUOTE_NONE matters: 7,233 MIND articles contain a double-quote inside the
    title or abstract, and the default QUOTE_MINIMAL silently swallows fields.
    keep_default_na=False stops an empty abstract becoming NaN and a literal
    "NA" headline becoming a missing value.
    """
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=columns,
        dtype=str,
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
        na_values=[],
        encoding="utf-8",
    )


def _mind_entities(cell: str) -> list[str]:
    """Pull the human-readable labels out of a MIND entity JSON blob."""
    if not cell or cell in ("[]", "null"):
        return []
    try:
        parsed = json.loads(cell)
    except json.JSONDecodeError:
        return []  # a handful of rows carry malformed JSON; drop rather than crash
    return [e["Label"] for e in parsed if isinstance(e, dict) and e.get("Label")]


def _parse_mind_impressions(cell: str) -> tuple[list[str], list[str]]:
    """Split `N123-1 N456-0` into (all candidates, clicked candidates)."""
    inview: list[str] = []
    clicked: list[str] = []
    for token in cell.split():
        # rsplit on the last dash: article ids themselves never contain one.
        article_id, _, label = token.rpartition("-")
        if not article_id:  # malformed token, keep it as a candidate only
            inview.append(token)
            continue
        inview.append(article_id)
        if label == "1":
            clicked.append(article_id)
    return inview, clicked


def load_mind(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Adapter: MIND-small TSVs -> (articles, impressions, history)."""
    cfg.require("train_news", "train_behaviors", "dev_news", "dev_behaviors")

    # --- articles: union of the train and dev catalogues (65,238 unique) ---
    news = pd.concat(
        [
            _read_mind_tsv(cfg.raw["train_news"], MIND_NEWS_COLUMNS),
            _read_mind_tsv(cfg.raw["dev_news"], MIND_NEWS_COLUMNS),
        ],
        ignore_index=True,
    ).drop_duplicates(subset="article_id", keep="first")

    entities = (
        news["title_entities"].map(_mind_entities)
        + news["abstract_entities"].map(_mind_entities)
    )
    articles = pd.DataFrame({
        "article_id": news["article_id"].astype(str),
        "title": news["title"],
        "abstract": news["abstract"],
        "body": "",                          # MIND-small ships a url, not article text
        "category": news["category"],
        "subcategory": news["subcategory"],
        # dict.fromkeys de-duplicates while preserving first-seen order.
        "entities": entities.map(lambda xs: list(dict.fromkeys(xs))),
        "published_time": pd.NaT,            # unavailable -> capability flag is False
    })
    articles["published_time"] = articles["published_time"].astype("datetime64[ns]")

    # --- impressions: both files, distinguished later by their timestamps ---
    behaviors = pd.concat(
        [
            _read_mind_tsv(cfg.raw["train_behaviors"], MIND_BEHAVIOR_COLUMNS),
            _read_mind_tsv(cfg.raw["dev_behaviors"], MIND_BEHAVIOR_COLUMNS),
        ],
        ignore_index=True,
    )

    parsed = behaviors["impressions"].map(_parse_mind_impressions)
    impressions = pd.DataFrame({
        # Positional index, because the raw ids collide between the two files.
        "impression_id": np.arange(len(behaviors), dtype="int64"),
        "source_impression_id": behaviors["impression_id"].astype("int64"),
        "user_id": behaviors["user_id"].astype(str),
        "timestamp": pd.to_datetime(behaviors["time"], format=MIND_TIME_FORMAT),
        "inview_ids": parsed.map(lambda pair: pair[0]),
        "clicked_ids": parsed.map(lambda pair: pair[1]),
    })

    # --- history: one fixed snapshot per user, verified identical across rows ---
    per_user = (
        behaviors[["user_id", "history"]]
        .drop_duplicates(subset="user_id", keep="first")
        .reset_index(drop=True)
    )
    history = _explode_ordered_history(
        user_ids=per_user["user_id"].astype(str),
        # 3,238 impressions belong to users with no history at all.
        id_lists=per_user["history"].map(lambda s: s.split() if s else []),
        timestamps=None,  # MIND records order only, never click times
    )
    return articles, impressions, history


# --------------------------------------------------------------------------- #
# EB-NeRD adapter
# --------------------------------------------------------------------------- #

def load_ebnerd(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Adapter: EB-NeRD parquet -> (articles, impressions, history)."""
    cfg.require(
        "articles", "train_behaviors", "train_history", "val_behaviors", "val_history"
    )

    raw_articles = pd.read_parquet(cfg.raw["articles"], engine="pyarrow")
    articles = pd.DataFrame({
        "article_id": raw_articles["article_id"].astype(str),  # int64 in the file
        "title": raw_articles["title"].fillna("").astype(str),
        "abstract": raw_articles["subtitle"].fillna("").astype(str),  # renamed
        "body": raw_articles["body"].fillna("").astype(str),
        # category_str is the readable label; `category` is an opaque integer.
        "category": raw_articles["category_str"].fillna("").astype(str),
        "subcategory": raw_articles["subcategory"].map(
            lambda v: "|".join(to_str_list(v))  # a list of ids -> one stable string
        ),
        "entities": raw_articles["ner_clusters"].map(to_str_list),
        "published_time": pd.to_datetime(raw_articles["published_time"], errors="coerce"),
    })

    # Both directories share a layout, so read them the same way and concatenate.
    behaviour_frames = []
    history_frames = []
    for split in ("train", "val"):
        behaviour_frames.append(
            pd.read_parquet(cfg.raw[f"{split}_behaviors"], engine="pyarrow")
        )
        history_frames.append(
            pd.read_parquet(cfg.raw[f"{split}_history"], engine="pyarrow")
        )
    behaviors = pd.concat(behaviour_frames, ignore_index=True)

    impressions = pd.DataFrame({
        "impression_id": np.arange(len(behaviors), dtype="int64"),
        "source_impression_id": behaviors["impression_id"].astype("int64"),
        "user_id": behaviors["user_id"].astype(str),
        "timestamp": pd.to_datetime(behaviors["impression_time"]),
        "inview_ids": behaviors["article_ids_inview"].map(to_str_list),
        "clicked_ids": behaviors["article_ids_clicked"].map(to_str_list),
    })

    # The validation history is a *different* snapshot; keeping the later one for a
    # user that appears in both avoids pairing test impressions with stale history.
    hist = pd.concat(history_frames, ignore_index=True).drop_duplicates(
        subset="user_id", keep="last"
    )
    history = _explode_ordered_history(
        user_ids=hist["user_id"].astype(str),
        id_lists=hist["article_id_fixed"].map(to_str_list),
        timestamps=hist["impression_time_fixed"],  # parallel list, same length
    )
    return articles, impressions, history


# --------------------------------------------------------------------------- #
# shared history construction
# --------------------------------------------------------------------------- #

def _explode_ordered_history(user_ids, id_lists, timestamps=None) -> pd.DataFrame:
    """Flatten per-user click lists into the long history table.

    Both datasets store history as a per-user list; MIND has order only while
    EB-NeRD carries a parallel list of timestamps. Producing `position` for both
    means recency features work even where timestamps are absent.
    """
    id_lists = list(id_lists)
    counts = np.fromiter((len(v) for v in id_lists), dtype="int64", count=len(id_lists))

    if counts.sum() == 0:
        return pd.DataFrame({c: [] for c in HISTORY_COLUMNS}).astype(
            {"user_id": str, "article_id": str, "position": "int32"}
        )

    flat_ids = np.concatenate([np.asarray(v, dtype=object) for v in id_lists if len(v)])
    # Per-user 0-based rank: a global arange minus each group's starting offset.
    starts = np.repeat(np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
    positions = np.arange(len(flat_ids), dtype="int64") - starts

    if timestamps is not None:
        stamps = np.concatenate(
            [np.asarray(v) for v in timestamps if v is not None and len(v)]
        )
        stamp_col = pd.to_datetime(pd.Series(stamps), errors="coerce")
        if len(stamp_col) != len(flat_ids):
            raise ValueError(
                f"history timestamps ({len(stamp_col)}) and article ids "
                f"({len(flat_ids)}) are not parallel"
            )
    else:
        stamp_col = pd.Series(pd.NaT, index=range(len(flat_ids)), dtype="datetime64[ns]")

    return pd.DataFrame({
        "user_id": np.repeat(np.asarray(list(user_ids), dtype=object), counts),
        "article_id": flat_ids.astype(str),
        "timestamp": stamp_col.to_numpy(),
        "position": positions.astype("int32"),
    })


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

# The adapter registry is the one place dataset names appear downstream of config.
ADAPTERS = {"mind": load_mind, "ebnerd": load_ebnerd}


def check_referential_integrity(
    articles: pd.DataFrame, impressions: pd.DataFrame, history: pd.DataFrame
) -> dict[str, float]:
    """Measure how much of the behavioural data resolves to a known article.

    A near-zero coverage here almost always means an id dtype mismatch rather
    than genuinely missing articles, so it is worth reporting on every build.
    """
    known = set(articles["article_id"])
    inview = {a for row in impressions["inview_ids"] for a in row}
    clicked = {a for row in impressions["clicked_ids"] for a in row}
    hist_ids = set(history["article_id"])

    def covered(ids: set[str]) -> float:
        return 100.0 * len(ids & known) / len(ids) if ids else 100.0

    return {
        "inview_coverage_pct": covered(inview),
        "clicked_coverage_pct": covered(clicked),
        "history_coverage_pct": covered(hist_ids),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True,
                        help="dataset config, e.g. config/mind.yaml")
    parser.add_argument("--out", type=Path, default=None,
                        help="override the output directory (default: data/processed/<dataset>)")
    parser.add_argument("--limit", type=int, default=None,
                        help="keep only the first N impressions (smoke testing)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    adapter = ADAPTERS.get(cfg.dataset)
    if adapter is None:
        raise SystemExit(f"no adapter for dataset {cfg.dataset!r}; known: {sorted(ADAPTERS)}")

    scale = f" (scale={cfg.scale})" if cfg.scale else ""
    print(f"[{cfg.dataset}]{scale} reading raw files")
    articles, impressions, history = adapter(cfg)

    if args.limit:
        impressions = impressions.head(args.limit)

    # Column order is part of the schema contract, so pin it before writing.
    articles = articles[ARTICLE_COLUMNS]
    impressions = impressions[IMPRESSION_COLUMNS]
    history = history[HISTORY_COLUMNS]

    out_dir = args.out or cfg.processed
    for name, df in (("articles", articles), ("impressions", impressions),
                     ("history", history)):
        write_table(df, out_dir / f"{name}.parquet", name)
        print("  " + summarise(df, name))

    coverage = check_referential_integrity(articles, impressions, history)
    print(f"  users          {impressions['user_id'].nunique():>9,}")
    print(f"  window         {impressions['timestamp'].min()} .. {impressions['timestamp'].max()}")
    for key, value in coverage.items():
        print(f"  {key:<14} {value:>9.2f}%")
    if coverage["clicked_coverage_pct"] < 50:
        print("  WARNING: most clicked ids are missing from the article table - "
              "check for an id dtype mismatch")

    print(f"  -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
