#!/usr/bin/env python3
"""Generate Codabench prediction files (Q5).

Both leaderboards use the same line format:

    <impression_id> [rank_1,rank_2,...,rank_n]

where rank_i is the 1-based predicted rank of the i-th article in the
impression's inview list, **in the original inview order**. The ranks must be a
permutation of 1..n, so `validate_submission` checks that locally rather than
discovering it on the leaderboard.

Note the id: submissions must echo `source_impression_id` (the id as it appears
in the raw file), not the internal unique `impression_id` this pipeline
assigns to work around MIND's colliding train/dev numbering.

Usage
-----
    python src/submission/generate_predictions.py --config config/mind.yaml
    python src/submission/generate_predictions.py --config config/ebnerd.yaml --split val
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config  # noqa: E402
from src.common.io import read_table  # noqa: E402
from src.retrieval.bm25 import BM25Index, build_queries  # noqa: E402
from src.retrieval.semantic import (  # noqa: E402
    build_user_vectors, encode_articles, l2_normalize, load_provided_embeddings,
)


def ranks_from_scores(scores: np.ndarray) -> list[int]:
    """Convert scores to 1-based ranks in the original candidate order.

    Highest score gets rank 1. Ties broken by position, which keeps the output
    a strict permutation as the format requires.
    """
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.int32)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks.tolist()


def validate_submission(rows: list[tuple[int, list[int]]],
                        expected: int) -> list[str]:
    """Check row count and that every rank list is a clean permutation."""
    problems = []
    if len(rows) != expected:
        problems.append(f"expected {expected} rows, wrote {len(rows)}")
    seen = set()
    for impression_id, ranks in rows:
        if impression_id in seen:
            problems.append(f"duplicate impression_id {impression_id}")
        seen.add(impression_id)
        if sorted(ranks) != list(range(1, len(ranks) + 1)):
            problems.append(
                f"impression {impression_id}: ranks are not a permutation of "
                f"1..{len(ranks)}"
            )
            if len(problems) > 5:
                break
    return problems


def score_impressions(cfg, impressions: pd.DataFrame, method: str,
                      last_n: int) -> list[np.ndarray]:
    """Score each impression's inview list with the chosen retriever."""
    articles = pd.read_parquet(cfg.features / "articles.parquet")
    articles["tokens"] = articles["tokens"].map(list)
    article_ids = articles["article_id"].tolist()
    row_of = {a: i for i, a in enumerate(article_ids)}

    profiles = pd.read_parquet(cfg.features / "user_profiles.parquet")
    profiles = profiles[profiles["split"] == impressions.attrs["split"]].copy()
    profiles["clicked_ids"] = profiles["clicked_ids"].map(list)
    profiles = profiles[profiles["user_id"].isin(set(impressions["user_id"]))]

    lengths = impressions["inview_ids"].map(len).to_numpy()
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    flat_ids = [a for row in impressions["inview_ids"] for a in row]
    flat_doc = np.array([row_of.get(a, -1) for a in flat_ids])

    if method == "bm25":
        index = BM25Index(article_ids, articles["tokens"].tolist())
        user_ids, token_lists = build_queries(profiles, articles, last_n)
        user_row = {u: i for i, u in enumerate(user_ids)}
        queries = index.query_matrix(token_lists)
        flat_user = np.repeat([user_row.get(u, -1) for u in impressions["user_id"]],
                              lengths)
        valid = (flat_doc >= 0) & (flat_user >= 0)
        flat = np.zeros(len(flat_ids), dtype=np.float32)
        flat[valid] = index.score_pairs(queries, flat_user[valid], flat_doc[valid])
    elif method == "semantic":
        if cfg.can("has_provided_embeddings"):
            raw = load_provided_embeddings(cfg, article_ids)
        else:
            raw = encode_articles(cfg, articles, batch_size=128)
        embeddings = l2_normalize(raw)
        user_ids, user_vectors = build_user_vectors(
            profiles, row_of, embeddings, last_n, False, 5.0
        )
        user_row = {u: i for i, u in enumerate(user_ids)}
        flat_user = np.repeat([user_row.get(u, -1) for u in impressions["user_id"]],
                              lengths)
        valid = (flat_doc >= 0) & (flat_user >= 0)
        flat = np.zeros(len(flat_ids), dtype=np.float32)
        flat[valid] = np.einsum("ij,ij->i", user_vectors[flat_user[valid]],
                                embeddings[flat_doc[valid]])
    elif method == "popularity":
        popularity = dict(zip(articles["article_id"], articles["train_clicks"]))
        flat = np.array([popularity.get(a, 0) for a in flat_ids], dtype=np.float32)
    else:
        raise ValueError(f"unknown method {method!r}")

    return [flat[offsets[i]:offsets[i + 1]] for i in range(len(impressions))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", default=None,
                        help="split to predict on (default: test for MIND, "
                             "val for EB-NeRD's validation-only dry run)")
    parser.add_argument("--method", default="semantic",
                        choices=["semantic", "bm25", "popularity"])
    parser.add_argument("--last-n", type=int, default=20)
    parser.add_argument("--validate-only", action="store_true",
                        help="check the format without writing the zip")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports" / "submissions")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    # EB-NeRD's scored leaderboard needs ebnerd_testset, which is not downloaded;
    # the validation split is used as a format dry run instead.
    split = args.split or ("val" if cfg.dataset == "ebnerd" else "test")

    impressions = read_table(cfg.processed / split / "impressions.parquet", "impressions")
    impressions.attrs["split"] = split
    print(f"[{cfg.dataset}/{split}] scoring {len(impressions):,} impressions "
          f"with {args.method}")

    scores = score_impressions(cfg, impressions, args.method, args.last_n)
    rows = [
        (int(source_id), ranks_from_scores(s))
        for source_id, s in zip(impressions["source_impression_id"], scores)
    ]

    problems = validate_submission(rows, len(impressions))
    if problems:
        for p in problems[:10]:
            print(f"  INVALID: {p}")
        raise SystemExit("submission failed local validation; nothing written")
    print(f"  format valid: {len(rows):,} rows, ranks are permutations")

    if args.validate_only:
        print("  --validate-only: nothing written")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    txt = args.out_dir / f"prediction_{cfg.dataset}_{split}.txt"
    with open(txt, "w", encoding="utf-8") as fh:
        for impression_id, ranks in rows:
            fh.write(f"{impression_id} [{','.join(map(str, ranks))}]\n")

    # Codabench expects prediction.txt inside a zip, at the archive root.
    archive = args.out_dir / f"submission_{cfg.dataset}_{split}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(txt, arcname="prediction.txt")

    print(f"  -> {txt}")
    print(f"  -> {archive}  ({archive.stat().st_size / 1e6:.1f} MB)")
    if cfg.dataset == "ebnerd":
        print("  NOTE: validation-split dry run. A scored RecSys 2024 submission "
              "needs ebnerd_testset (make ebnerd-testset).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
