#!/usr/bin/env python3
"""Score the EB-NeRD official test set and emit a RecSys 2024 submission.

`ebnerd_testset` is unlabelled and ships no train split, so like MINDlarge_test
it cannot go through `clean.py -> split.py -> feature_store.py`. This is the
inference-only path for Codabench competition 2469.

Format differences from the MIND submission, both taken from the official
`ebnerd-benchmark` helper `write_submission_file`:
  * the file inside the zip is `predictions.txt` (plural), not `prediction.txt`
  * ranks are 1-based with the highest score ranked 1, in original inview order
    - verified identical to `rank_predictions_by_score` except on ties, where
      this implementation keeps the original order instead of reversing it

Usage
-----
    python src/submission/predict_ebnerd_test.py --dir data/raw/ebnerd/ebnerd_testset
"""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.common.io import to_str_list  # noqa: E402
from src.retrieval.semantic import l2_normalize  # noqa: E402
from src.submission.generate_predictions import ranks_from_scores  # noqa: E402

DEFAULT_VECTORS = (
    REPO_ROOT / "data" / "raw" / "ebnerd" / "Ekstra_Bladet_word2vec"
    / "Ekstra_Bladet_word2vec" / "document_vector.parquet"
)


def find_test_dir(root: Path, prefer: str | None = None) -> Path:
    """Locate the directory holding behaviors.parquet.

    The testset archive may unpack nested, and the demo/small bundles name the
    equivalent directory `validation`, so search rather than hardcode.
    """
    hits = sorted(p.parent for p in root.rglob("behaviors.parquet"))
    if not hits:
        raise SystemExit(
            f"could not find behaviors.parquet under {root}\n"
            f"  contents: {[p.name for p in root.iterdir()] if root.is_dir() else 'missing'}"
        )
    if prefer:
        named = [h for h in hits if h.name == prefer]
        if not named:
            raise SystemExit(f"no '{prefer}' directory under {root}; found {[h.name for h in hits]}")
        return named[0]
    # Prefer a directory literally called `test` when the archive offers several.
    for hit in hits:
        if hit.name == "test":
            return hit
    return hits[0]


def find_articles(root: Path) -> Path:
    hits = list(root.rglob("articles.parquet"))
    if not hits:
        raise SystemExit(f"no articles.parquet under {root}")
    return hits[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dir", type=Path,
                        default=REPO_ROOT / "data" / "raw" / "ebnerd" / "ebnerd_testset")
    parser.add_argument("--vectors", type=Path, default=DEFAULT_VECTORS,
                        help="provided document_vector.parquet")
    parser.add_argument("--last-n", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=200_000,
                        help="impressions held in memory at once")
    parser.add_argument("--subdir", default=None,
                        help="force a specific split directory, e.g. validation")
    parser.add_argument("--name", default="ebnerd_testset")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "reports" / "submissions")
    args = parser.parse_args(argv)

    test_dir = find_test_dir(args.dir, args.subdir)
    articles_path = find_articles(args.dir)
    print(f"[{args.name}] reading {test_dir}")

    articles = pd.read_parquet(articles_path, columns=["article_id"], engine="pyarrow")
    article_ids = articles["article_id"].astype(str).to_numpy()
    row_of = {a: i for i, a in enumerate(article_ids)}
    print(f"  {len(article_ids):,} articles")

    # Provided Danish word2vec vectors, aligned to the article table's row order.
    vectors = pd.read_parquet(args.vectors, engine="pyarrow")
    id_col, vec_col = vectors.columns[0], vectors.columns[1]
    lookup = dict(zip(vectors[id_col].astype(str), vectors[vec_col]))
    dim = len(next(iter(lookup.values())))
    embeddings = np.zeros((len(article_ids), dim), dtype=np.float32)
    missing = 0
    for i, article_id in enumerate(article_ids):
        vec = lookup.get(article_id)
        if vec is None:
            missing += 1  # zero row; contributes nothing and can never be ranked up
            continue
        embeddings[i] = np.asarray(vec, dtype=np.float32)
    embeddings = l2_normalize(embeddings)
    print(f"  {len(article_ids) - missing:,} vectors matched (dim {dim}), "
          f"{missing:,} missing")

    # History is one row per user. 807k users x 300 dims is ~970 MB, so hold it
    # as one contiguous array with an index rather than a dict of small arrays.
    hist_file = pq.ParquetFile(test_dir / "history.parquet")
    n_users = hist_file.metadata.num_rows
    user_index: dict[str, int] = {}
    user_vectors = np.zeros((n_users, dim), dtype=np.float32)
    filled = 0
    for batch in hist_file.iter_batches(batch_size=50_000,
                                        columns=["user_id", "article_id_fixed"]):
        block = batch.to_pandas()
        for user_id, clicked in zip(block["user_id"].astype(str),
                                    block["article_id_fixed"]):
            rows = [row_of[a] for a in to_str_list(clicked)[-args.last_n:] if a in row_of]
            if rows:
                user_vectors[filled] = embeddings[rows].mean(axis=0)
            user_index[user_id] = filled
            filled += 1
    user_vectors = l2_normalize(user_vectors)
    print(f"  {filled:,} user histories indexed")

    beh_file = pq.ParquetFile(test_dir / "behaviors.parquet")
    n_impressions = beh_file.metadata.num_rows
    print(f"  {n_impressions:,} impressions ({beh_file.metadata.num_row_groups} row groups)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    txt = args.out_dir / f"predictions_{args.name}.txt"
    zero = np.zeros(dim, dtype=np.float32)
    n_rows = 0
    n_no_profile = 0
    started = time.time()

    with open(txt, "w", encoding="utf-8") as out:
        # Stream row groups: 13.5M impressions x ~11.7 candidates is ~158M rows,
        # which cannot be materialised as pandas object columns.
        for batch in beh_file.iter_batches(
            batch_size=args.chunk_size,
            columns=["impression_id", "user_id", "article_ids_inview"],
        ):
            block = batch.to_pandas()
            for impression_id, user_id, inview in zip(
                block["impression_id"], block["user_id"].astype(str),
                block["article_ids_inview"]
            ):
                candidates = to_str_list(inview)
                slot = user_index.get(user_id)
                profile = zero if slot is None else user_vectors[slot]
                if slot is None:
                    n_no_profile += 1
                rows = np.fromiter((row_of.get(a, -1) for a in candidates),
                                   dtype=np.int64, count=len(candidates))
                scores = np.zeros(len(candidates), dtype=np.float32)
                ok = rows >= 0
                if ok.any():
                    scores[ok] = embeddings[rows[ok]] @ profile
                ranks = ranks_from_scores(scores)
                out.write(f"{int(impression_id)} [{','.join(map(str, ranks))}]\n")
                n_rows += 1

            rate = n_rows / max(time.time() - started, 1e-6)
            eta = (n_impressions - n_rows) / max(rate, 1e-6)
            print(f"\r  scored {n_rows:,}/{n_impressions:,} "
                  f"({rate:,.0f}/s, eta {eta / 60:.1f}m)   ", end="", flush=True)
    print()

    if n_rows != n_impressions:
        raise SystemExit(f"wrote {n_rows} rows, expected {n_impressions}")
    print(f"  {n_no_profile:,} impressions had no user profile "
          f"({100 * n_no_profile / max(1, n_rows):.1f}%)")

    archive = args.out_dir / f"submission_{args.name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        # predictions.txt (plural) - the RecSys 2024 scorer's expected name.
        zf.write(txt, arcname="predictions.txt")

    print(f"  -> {txt}  ({txt.stat().st_size / 1e6:.1f} MB)")
    print(f"  -> {archive}  ({archive.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
