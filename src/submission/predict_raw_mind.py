#!/usr/bin/env python3
"""Score a raw MIND directory directly and emit a Codabench submission.

MINDlarge_test is unlabelled and ships no train split, so it cannot go through
`clean.py -> split.py -> feature_store.py`, which all assume labelled
interactions and a temporal partition. This is the inference-only path: it
reads one MIND directory (news.tsv + behaviors.tsv), builds article and user
representations from that directory alone, and writes predictions.

Row order is preserved exactly as it appears in behaviors.tsv, because the
official evaluate.py zips truth and prediction line by line and rejects any
impression-id mismatch.

Usage
-----
    python src/submission/predict_raw_mind.py --dir data/raw/mind/MINDlarge_test
    python src/submission/predict_raw_mind.py --dir data/raw/mind/MINDsmall_dev --method bm25
"""

from __future__ import annotations

import os

# Cap BLAS threading before numpy is imported. The top-k scorer issues millions
# of tiny matmuls, where OpenBLAS spends more time starting and synchronising
# threads than doing arithmetic - it saturated 8 of 12 cores and showed 12m of
# system time against 4m of user time. Single-threaded is both faster here and
# leaves the machine usable.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.common.text import tokenize  # noqa: E402
from src.data.clean import (  # noqa: E402
    MIND_BEHAVIOR_COLUMNS, MIND_NEWS_COLUMNS, _parse_mind_impressions, _read_mind_tsv,
)
from src.retrieval.bm25 import BM25Index  # noqa: E402
from src.retrieval.semantic import MIND_ENCODER, l2_normalize  # noqa: E402
from src.submission.generate_predictions import ranks_from_scores, validate_submission  # noqa: E402


def encode(texts: list[str], cache: Path, ids: np.ndarray, batch_size: int) -> np.ndarray:
    """Encode article text, reusing a cache keyed on the exact article id list."""
    ids_cache = cache.with_name(cache.stem + "_ids.npy")
    if cache.exists() and ids_cache.exists():
        cached = np.load(ids_cache, allow_pickle=True)
        if len(cached) == len(ids) and (cached == ids).all():
            print(f"  reusing cached embeddings {cache.name}")
            return np.load(cache)

    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  encoding {len(texts):,} articles with {MIND_ENCODER} on {device}")
    model = SentenceTransformer(MIND_ENCODER, device=device)
    started = time.time()
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                           convert_to_numpy=True).astype(np.float32)
    print(f"  encoded in {time.time() - started:.1f}s")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, vectors)
    np.save(ids_cache, ids)
    return vectors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dir", type=Path, required=True,
                        help="a MIND directory containing news.tsv and behaviors.tsv")
    parser.add_argument("--method", default="semantic", choices=["semantic", "bm25"])
    parser.add_argument("--last-n", type=int, default=50,
                        help="history clicks used per user")
    parser.add_argument("--pooling", default="topk", choices=["mean", "topk"],
                        help="topk scores a candidate by its mean similarity to the "
                             "k most similar history articles; mean pools the history "
                             "into one vector first. Measured on MIND val: topk k=5 "
                             "gives AUC 0.6449 vs 0.6305 for mean-pool.")
    parser.add_argument("--topk", type=int, default=5,
                        help="k for --pooling topk; 5 was the peak of a 1..20 sweep")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=100_000,
                        help="impressions held in memory at once")
    parser.add_argument("--score-batch", type=int, default=200_000,
                        help="candidate rows per embedding gather")
    parser.add_argument("--name", default=None, help="label used in output filenames")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "reports" / "submissions")
    args = parser.parse_args(argv)

    news_path = args.dir / "news.tsv"
    behaviors_path = args.dir / "behaviors.tsv"
    for path in (news_path, behaviors_path):
        if not path.exists():
            raise SystemExit(f"missing {path}")
    name = args.name or args.dir.name

    print(f"[{name}] reading raw files")
    news = _read_mind_tsv(news_path, MIND_NEWS_COLUMNS).drop_duplicates(
        subset="article_id", keep="first"
    )
    behaviors = _read_mind_tsv(behaviors_path, MIND_BEHAVIOR_COLUMNS)
    print(f"  {len(news):,} articles, {len(behaviors):,} impressions")

    article_ids = news["article_id"].to_numpy()
    row_of = {a: i for i, a in enumerate(article_ids)}

    # Article representations are built once; impressions are streamed, because
    # MINDlarge_test has 2.37M impressions x ~39.5 candidates = ~94M candidate
    # rows. Flattening those at once would need ~135 GB for the embedding gather.
    if args.method == "semantic":
        text = (news["title"].fillna("") + ". " + news["abstract"].fillna("")).tolist()
        # Key the cache on the source directory, not --name: the vectors depend
        # only on which articles were encoded, so two runs over the same data
        # with different labels must share one cache instead of re-encoding.
        cache = (REPO_ROOT / "data" / "feature_store"
                 / f"{args.dir.resolve().name}_embeddings.npy")
        embeddings = l2_normalize(encode(text, cache, article_ids, args.batch_size))
        index = None
        token_lookup = None
    else:
        tokens = [tokenize(f"{t} {a}", "en")
                  for t, a in zip(news["title"], news["abstract"])]
        index = BM25Index(list(article_ids), tokens)
        token_lookup = dict(zip(article_ids, tokens))
        embeddings = None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    txt = args.out_dir / f"prediction_{name}.txt"

    n_rows = 0
    n_missing = 0
    labelled = 0
    seen_ids: set[int] = set()
    started = time.time()

    with open(txt, "w", encoding="utf-8") as out:
        for chunk in range(0, len(behaviors), args.chunk_size):
            block = behaviors.iloc[chunk:chunk + args.chunk_size]
            parsed = block["impressions"].map(_parse_mind_impressions)
            inview = parsed.map(lambda pair: pair[0])
            labelled += int(parsed.map(lambda pair: len(pair[1]) > 0).sum())
            histories = block["history"].map(lambda s: s.split() if s else [])

            lengths = inview.map(len).to_numpy()
            offsets = np.concatenate([[0], np.cumsum(lengths)])
            flat_ids = [a for row in inview for a in row]
            flat_doc = np.array([row_of.get(a, -1) for a in flat_ids])
            flat_user = np.repeat(np.arange(len(block)), lengths)
            valid = flat_doc >= 0
            n_missing += int((~valid).sum())
            scores = np.zeros(len(flat_ids), dtype=np.float32)

            if args.method == "semantic" and args.pooling == "topk":
                # Score each candidate against the user's whole history and keep
                # the k best matches. Mean-pooling collapses a multi-interest
                # history into one vector and washes out exactly the niche
                # interest that explains the click.
                for i, clicked in enumerate(histories):
                    lo, hi = offsets[i], offsets[i + 1]
                    rows = [row_of[a] for a in clicked[-args.last_n:] if a in row_of]
                    if not rows:
                        continue
                    cand = flat_doc[lo:hi]
                    ok = cand >= 0
                    if not ok.any():
                        continue
                    sims = embeddings[cand[ok]] @ embeddings[rows].T
                    k = min(args.topk, sims.shape[1])
                    block_scores = np.sort(sims, axis=1)[:, -k:].mean(1)
                    seg = scores[lo:hi]
                    seg[ok] = block_scores
                    scores[lo:hi] = seg
            elif args.method == "semantic":
                dim = embeddings.shape[1]
                user_vectors = np.zeros((len(block), dim), dtype=np.float32)
                for i, clicked in enumerate(histories):
                    rows = [row_of[a] for a in clicked[-args.last_n:] if a in row_of]
                    if rows:
                        user_vectors[i] = embeddings[rows].mean(axis=0)
                user_vectors = l2_normalize(user_vectors)

                # Sub-batch the gather so peak memory stays bounded.
                idx = np.flatnonzero(valid)
                for start in range(0, len(idx), args.score_batch):
                    sel = idx[start:start + args.score_batch]
                    scores[sel] = np.einsum(
                        "ij,ij->i",
                        user_vectors[flat_user[sel]], embeddings[flat_doc[sel]],
                    )
            else:
                queries = index.query_matrix([
                    [tok for a in clicked[-args.last_n:] for tok in token_lookup.get(a, ())]
                    for clicked in histories
                ])
                scores[valid] = index.score_pairs(
                    queries, flat_user[valid], flat_doc[valid]
                )

            # Write immediately, preserving the raw file's row order.
            for i, impression_id in enumerate(block["impression_id"]):
                ranks = ranks_from_scores(scores[offsets[i]:offsets[i + 1]])
                impression_id = int(impression_id)
                if impression_id in seen_ids:
                    raise SystemExit(f"duplicate impression_id {impression_id}")
                seen_ids.add(impression_id)
                if sorted(ranks) != list(range(1, len(ranks) + 1)):
                    raise SystemExit(f"impression {impression_id}: ranks not a permutation")
                out.write(f"{impression_id} [{','.join(map(str, ranks))}]\n")
                n_rows += 1

            done = min(chunk + args.chunk_size, len(behaviors))
            rate = done / max(time.time() - started, 1e-6)
            print(f"\r  scored {done:,}/{len(behaviors):,} "
                  f"({rate:,.0f}/s, eta {(len(behaviors) - done) / max(rate, 1e-6):.0f}s)   ",
                  end="", flush=True)
    print()

    if n_rows != len(behaviors):
        raise SystemExit(f"wrote {n_rows} rows, expected {len(behaviors)}")
    print(f"  {labelled:,} impressions carried labels "
          f"({'unlabelled test set, as expected' if labelled == 0 else 'labelled'})")
    if n_missing:
        print(f"  {n_missing:,} candidates absent from news.tsv, scored 0")
    print(f"  format valid: {n_rows:,} rows, ranks are permutations")

    archive = args.out_dir / f"submission_{name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(txt, arcname="prediction.txt")  # evaluate.py opens res/prediction.txt

    print(f"  -> {txt}  ({txt.stat().st_size / 1e6:.1f} MB)")
    print(f"  -> {archive}  ({archive.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
