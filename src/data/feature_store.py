#!/usr/bin/env python3
"""Build the reusable feature store for articles and users.

Everything derived from behaviour - popularity, head/tail bands, cold/warm
flags - is fit on the **train split only** and then reused unchanged by val and
test. Fitting popularity on the evaluation split is the easiest way to
manufacture a leaderboard score that does not survive contact with serving.

Outputs (data/feature_store/<dataset>/):
    articles.parquet       article text, tokens, train popularity, head/tail band
    user_profiles.parquet  per (user, split) click history and recency
    stats.json             catalog size, thresholds, popularity concentration

Usage
-----
    python src/data/feature_store.py --config config/mind.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.common.config import Config, load_config  # noqa: E402
from src.common.io import read_table  # noqa: E402
from src.common.text import tokenize  # noqa: E402

COLD_START_MAX_CLICKS = 5   # users with fewer history clicks are "cold"
HEAD_FRACTION = 0.20        # top 20% of articles by train clicks are "head"
LOW_HISTORY_QUANTILE = 0.25 # dataset-relative cold band, always non-empty


def build_article_features(articles: pd.DataFrame, cfg: Config,
                           train_clicks: Counter) -> pd.DataFrame:
    """Tokenise article text and attach train-split popularity bands."""
    fields = cfg.text.get("index_fields", ["title", "abstract"])
    # One indexable string per article, from the configured fields only.
    text = articles[fields[0]].fillna("").astype(str)
    for field in fields[1:]:
        text = text + " " + articles[field].fillna("").astype(str)

    tokens = text.map(lambda t: tokenize(t, cfg.language))
    counts = articles["article_id"].map(lambda a: train_clicks.get(a, 0)).astype("int64")

    out = pd.DataFrame({
        "article_id": articles["article_id"],
        "title": articles["title"],
        "abstract": articles["abstract"],
        "category": articles["category"],
        "published_time": articles["published_time"],
        "tokens": tokens,
        "n_tokens": tokens.map(len).astype("int32"),
        "train_clicks": counts,
    })

    # Rank 0 = most clicked. Ties broken by id for a deterministic ordering.
    out = out.sort_values(["train_clicks", "article_id"], ascending=[False, True])
    out["popularity_rank"] = np.arange(len(out), dtype="int64")
    # Head/tail is defined over the *clicked* catalogue; never-clicked articles
    # are all tail, otherwise the band would be mostly zeros.
    n_clicked = int((out["train_clicks"] > 0).sum())
    head_cutoff = int(n_clicked * HEAD_FRACTION)
    out["is_head"] = (out["popularity_rank"] < head_cutoff) & (out["train_clicks"] > 0)
    return out.sort_values("article_id").reset_index(drop=True)


def build_user_profiles(history: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Roll the long history up into one ordered click list per (user, split).

    Each split reads the snapshot that split.py certified as ending before the
    split begins, so a profile can never contain a future click.
    """
    frames = []
    for split, info in meta["splits"].items():
        snapshot = info["history_snapshot"]
        part = history[history["snapshot"] == snapshot]
        # `position` is the within-user ordering, so sorting by it restores
        # chronological order even where timestamps are absent (MIND).
        part = part.sort_values(["user_id", "position"])
        grouped = part.groupby("user_id", sort=False)

        profile = pd.DataFrame({
            "user_id": grouped["article_id"].apply(list).index,
            "clicked_ids": grouped["article_id"].apply(list).to_numpy(),
            "last_click_time": grouped["timestamp"].max().to_numpy(),
        })
        profile["split"] = split
        profile["n_clicks"] = profile["clicked_ids"].map(len).astype("int32")
        profile["is_cold"] = profile["n_clicks"] < COLD_START_MAX_CLICKS
        # EB-NeRD's shortest history is 5 clicks, so the absolute threshold above
        # selects nobody there. A within-dataset quartile keeps the cold/warm
        # slice non-empty on both datasets; the harness reports both bands.
        cutoff = profile["n_clicks"].quantile(LOW_HISTORY_QUANTILE)
        profile["is_low_history"] = profile["n_clicks"] <= cutoff
        frames.append(profile)

    return pd.concat(frames, ignore_index=True)[
        ["user_id", "split", "clicked_ids", "n_clicks", "last_click_time",
         "is_cold", "is_low_history"]
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    proc = cfg.processed
    out_dir = args.out_dir or cfg.features
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = proc / "split_meta.json"
    if not meta_path.exists():
        raise SystemExit(
            f"split metadata not found: {meta_path}\n"
            f"  Run: python src/data/split.py --config {args.config}"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    articles = read_table(proc / "articles.parquet", "articles")
    history = read_table(proc / "history.parquet", "history")
    train = read_table(proc / "train" / "impressions.parquet", "impressions")

    # Popularity comes from the TRAIN split alone and is reused by val/test.
    train_clicks = Counter(a for row in train["clicked_ids"] for a in row)

    article_features = build_article_features(articles, cfg, train_clicks)
    profiles = build_user_profiles(history, meta)

    article_features.to_parquet(out_dir / "articles.parquet", index=False)
    profiles.to_parquet(out_dir / "user_profiles.parquet", index=False)

    empty_tokens = int((article_features["n_tokens"] == 0).sum())
    clicked_catalog = int((article_features["train_clicks"] > 0).sum())
    top1pct = max(1, len(article_features) // 100)
    concentration = float(
        article_features.nlargest(top1pct, "train_clicks")["train_clicks"].sum()
        / max(1, article_features["train_clicks"].sum())
    )

    stats = {
        "dataset": cfg.dataset,
        "scale": cfg.scale,
        "n_articles": int(len(article_features)),
        "n_articles_clicked_in_train": clicked_catalog,
        "mean_tokens_per_article": float(article_features["n_tokens"].mean()),
        "articles_with_no_tokens": empty_tokens,
        "head_fraction": HEAD_FRACTION,
        "cold_start_max_clicks": COLD_START_MAX_CLICKS,
        "click_share_of_top_1pct_articles": concentration,
        "profiles": {
            split: {
                "n_users": int((profiles["split"] == split).sum()),
                "cold_users": int(profiles.loc[profiles["split"] == split, "is_cold"].sum()),
                "low_history_users": int(
                    profiles.loc[profiles["split"] == split, "is_low_history"].sum()
                ),
                "mean_history_len": float(
                    profiles.loc[profiles["split"] == split, "n_clicks"].mean()
                ),
            }
            for split in meta["splits"]
        },
        "popularity_fit_on": "train",
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print(f"[{cfg.dataset}] feature store")
    print(f"  articles       {len(article_features):>8,}  "
          f"mean {stats['mean_tokens_per_article']:.1f} tokens, "
          f"{empty_tokens} with none")
    print(f"  clicked in train {clicked_catalog:>6,}  "
          f"top 1% take {concentration:.1%} of clicks")
    for split, info in stats["profiles"].items():
        print(f"  {split:<6} profiles {info['n_users']:>7,}  "
              f"cold {info['cold_users']:>6,}  low-history {info['low_history_users']:>6,}  "
              f"mean history {info['mean_history_len']:.1f}")
    print(f"  -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
