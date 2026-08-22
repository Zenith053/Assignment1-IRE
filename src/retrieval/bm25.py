#!/usr/bin/env python3
"""BM25 lexical retrieval over article title + abstract.

Implemented directly rather than via `rank_bm25`, which scores one document at a
time in Python and cannot handle 65k documents against tens of thousands of
queries in reasonable time.

The index is a genuine inverted index - `postings` maps a term to the documents
containing it - materialised as a sparse matrix so scoring a query is one
sparse matrix-vector product instead of a Python loop over postings lists.

Two scoring modes, both needed downstream:
    retrieve()     top-K over the whole corpus  -> recall@K (Q2.4)
    score_inview() score only an impression's candidates -> ranking metrics (Q4)

Usage
-----
    python src/retrieval/bm25.py --config config/mind.yaml --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.common.config import Config, load_config  # noqa: E402
from src.common.io import read_table  # noqa: E402
from src.common.text import tokenize  # noqa: E402
from src.retrieval.pool import POOLS, build_pool  # noqa: E402

RECALL_KS = (50, 100, 200)


class BM25Index:
    """An inverted index with precomputed BM25 document weights."""

    def __init__(self, doc_ids: list[str], token_lists, k1: float = 1.2,
                 b: float = 0.75):
        self.doc_ids = list(doc_ids)
        self.k1 = k1
        self.b = b
        self.doc_index = {d: i for i, d in enumerate(self.doc_ids)}
        self._build(token_lists)

    def _build(self, token_lists) -> None:
        """Build postings, then fold BM25's document half into a sparse matrix."""
        vocab: dict[str, int] = {}
        rows, cols, tfs = [], [], []
        doc_lengths = np.zeros(len(self.doc_ids), dtype=np.float32)

        for doc_i, tokens in enumerate(token_lists):
            doc_lengths[doc_i] = len(tokens)
            if not tokens:
                continue
            counts: dict[int, int] = {}
            for tok in tokens:
                term_i = vocab.get(tok)
                if term_i is None:
                    term_i = vocab[tok] = len(vocab)
                counts[term_i] = counts.get(term_i, 0) + 1
            rows.extend([doc_i] * len(counts))
            cols.extend(counts.keys())
            tfs.extend(counts.values())

        self.vocabulary = vocab
        n_docs, n_terms = len(self.doc_ids), len(vocab)
        tf = sparse.csr_matrix(
            (np.asarray(tfs, dtype=np.float32), (rows, cols)),
            shape=(n_docs, n_terms),
        )

        # Document frequency straight off the postings structure.
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        # Robertson IDF with the +1 guard, so a term in every document scores ~0
        # rather than going negative.
        self.idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        avgdl = float(doc_lengths.mean()) if n_docs else 0.0
        # Length normalisation is per-document, so it can be applied row-wise.
        norm = (self.k1 * (1.0 - self.b + self.b * doc_lengths / max(avgdl, 1e-9)))

        # weight(t,d) = idf(t) * tf*(k1+1) / (tf + norm(d)); build it in COO form.
        coo = tf.tocoo()
        numerator = coo.data * (self.k1 + 1.0)
        denominator = coo.data + norm[coo.row]
        weights = (numerator / denominator) * self.idf[coo.col]
        self.weights = sparse.csr_matrix(
            (weights.astype(np.float32), (coo.row, coo.col)), shape=(n_docs, n_terms)
        )
        # Terms-by-documents orientation makes query scoring a single product.
        self.weights_t = self.weights.T.tocsr()
        self.doc_lengths = doc_lengths
        self.avgdl = avgdl

    def query_matrix(self, token_lists) -> sparse.csr_matrix:
        """Turn tokenised queries into a sparse query-term-frequency matrix."""
        rows, cols, vals = [], [], []
        for q_i, tokens in enumerate(token_lists):
            counts: dict[int, int] = {}
            for tok in tokens:
                term_i = self.vocabulary.get(tok)
                if term_i is None:
                    continue  # out-of-vocabulary terms contribute nothing
                counts[term_i] = counts.get(term_i, 0) + 1
            rows.extend([q_i] * len(counts))
            cols.extend(counts.keys())
            vals.extend(counts.values())
        return sparse.csr_matrix(
            (np.asarray(vals, dtype=np.float32), (rows, cols)),
            shape=(len(token_lists), len(self.vocabulary)),
        )

    def retrieve(self, queries: sparse.csr_matrix, k: int,
                 batch_size: int = 256, pool: np.ndarray | None = None) -> np.ndarray:
        """Top-k document indices per query, densifying only one batch at a time.

        `pool` restricts scoring to a subset of documents; returned indices are
        still absolute row numbers so callers need no translation.
        """
        weights_t = self.weights_t if pool is None else self.weights[pool].T.tocsr()
        n_avail = weights_t.shape[1]
        k = min(k, n_avail)
        out = np.empty((queries.shape[0], k), dtype=np.int32)
        for start in range(0, queries.shape[0], batch_size):
            chunk = queries[start:start + batch_size]
            scores = np.asarray((chunk @ weights_t).todense())
            # argpartition finds the top k without a full sort of 65k columns.
            part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
            ordered = np.take_along_axis(
                part, np.argsort(-np.take_along_axis(scores, part, 1), axis=1), axis=1
            )
            # Map back to absolute row ids when a pool was applied.
            out[start:start + chunk.shape[0]] = ordered if pool is None else pool[ordered]
        return out

    def score_pairs(self, queries: sparse.csr_matrix, query_idx: np.ndarray,
                    doc_idx: np.ndarray, batch_size: int = 500_000) -> np.ndarray:
        """Score specific (query, document) pairs without a full score matrix."""
        out = np.empty(len(query_idx), dtype=np.float32)
        for start in range(0, len(query_idx), batch_size):
            end = start + batch_size
            q = queries[query_idx[start:end]]
            d = self.weights[doc_idx[start:end]]
            # Row-wise dot product of two aligned sparse matrices.
            out[start:end] = np.asarray(q.multiply(d).sum(axis=1)).ravel()
        return out


def build_queries(profiles: pd.DataFrame, articles: pd.DataFrame,
                  last_n: int) -> tuple[list[str], list[list[str]]]:
    """Concatenate the tokens of each user's most recent `last_n` clicks."""
    tokens_by_id = dict(zip(articles["article_id"], articles["tokens"]))
    user_ids, queries = [], []
    for user_id, clicked in zip(profiles["user_id"], profiles["clicked_ids"]):
        recent = list(clicked)[-last_n:]  # profiles are stored oldest-first
        merged: list[str] = []
        for article_id in recent:
            merged.extend(tokens_by_id.get(article_id, ()))
        user_ids.append(user_id)
        queries.append(merged)
    return user_ids, queries


def recall_at_k(retrieved: np.ndarray, doc_ids: list[str], impressions: pd.DataFrame,
                user_row: dict[str, int], ks=RECALL_KS) -> dict[str, float]:
    """Fraction of ground-truth clicks that appear in the user's top-K.

    Averaged over impressions, so a user with many impressions counts once per
    impression - matching how the ranking metrics are averaged.
    """
    doc_id_array = np.asarray(doc_ids)
    results = {}
    for k in ks:
        hits, total = 0, 0
        for user_id, clicked in zip(impressions["user_id"], impressions["clicked_ids"]):
            row = user_row.get(user_id)
            if row is None or not clicked:
                continue
            topk = set(doc_id_array[retrieved[row, :k]])
            hits += sum(1 for c in clicked if c in topk)
            total += len(clicked)
        results[f"recall@{k}"] = hits / total if total else 0.0
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--last-n", type=int, default=20,
                        help="how many recent clicks form the query")
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--sample", type=int, default=20000,
                        help="impressions to evaluate (0 = all)")
    parser.add_argument("--pools", nargs="+", default=["all", "circulating", "fresh"],
                        choices=list(POOLS),
                        help="candidate pools to report recall over")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    articles = pd.read_parquet(cfg.features / "articles.parquet")
    articles["tokens"] = articles["tokens"].map(list)
    profiles = pd.read_parquet(cfg.features / "user_profiles.parquet")
    profiles = profiles[profiles["split"] == args.split]
    profiles["clicked_ids"] = profiles["clicked_ids"].map(list)
    impressions = read_table(cfg.processed / args.split / "impressions.parquet",
                             "impressions")

    if args.sample and args.sample < len(impressions):
        impressions = impressions.sample(args.sample, random_state=13)

    print(f"[{cfg.dataset}/{args.split}] building BM25 index over "
          f"{len(articles):,} articles")
    index = BM25Index(articles["article_id"].tolist(), articles["tokens"].tolist(),
                      k1=args.k1, b=args.b)
    print(f"  vocabulary {len(index.vocabulary):,}  avg doc length {index.avgdl:.1f}")

    # Only users that actually appear in this split need a query.
    needed = set(impressions["user_id"])
    profiles = profiles[profiles["user_id"].isin(needed)]
    user_ids, token_lists = build_queries(profiles, articles, args.last_n)
    user_row = {u: i for i, u in enumerate(user_ids)}
    queries = index.query_matrix(token_lists)
    print(f"  {len(user_ids):,} user queries, mean {queries.getnnz(axis=1).mean():.1f} "
          f"distinct terms")

    by_pool: dict[str, dict] = {}
    for pool_name in args.pools:
        pool_idx, unavailable = build_pool(
            pool_name, articles, impressions, cfg.can("has_published_time")
        )
        if unavailable:
            by_pool[pool_name] = {"available": False, "reason": unavailable}
            print(f"  pool {pool_name:<12} N/A ({unavailable})")
            continue

        retrieved = index.retrieve(queries, k=max(RECALL_KS), pool=pool_idx)
        metrics = recall_at_k(retrieved, index.doc_ids, impressions, user_row)
        # A random retriever over the same pool: the only fair floor to compare to.
        baseline = {f"random@{k}": min(1.0, k / max(1, len(pool_idx))) for k in RECALL_KS}
        by_pool[pool_name] = {
            "available": True, "pool_size": int(len(pool_idx)), **metrics, **baseline
        }
        lift = metrics["recall@50"] / baseline["random@50"] if baseline["random@50"] else 0
        print(f"  pool {pool_name:<12} size {len(pool_idx):>7,}  " + "  ".join(
            f"r@{k}={metrics[f'recall@{k}']:.4f}" for k in RECALL_KS
        ) + f"  ({lift:.1f}x random)")

    result = {
        "dataset": cfg.dataset, "split": args.split, "method": "bm25",
        "params": {"k1": args.k1, "b": args.b, "last_n": args.last_n},
        "n_impressions_evaluated": int(len(impressions)),
        "n_articles": int(len(articles)),
        "vocabulary_size": int(len(index.vocabulary)),
        "pools": by_pool,
    }
    out = args.out or REPO_ROOT / "reports" / f"recall_bm25_{cfg.dataset}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
