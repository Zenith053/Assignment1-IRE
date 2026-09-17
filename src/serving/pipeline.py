#!/usr/bin/env python3
"""Q4 phase 1: the two-stage pipeline answering ONE request, timed per stage.

    request (user_id, t)
      stage 1  candidate generation         src/rerank/retriever.UnionRetriever logic
        profile     look up the user's click history
        bm25        top-80 by BM25 over the user's clicked words
        faiss       top-100 by inner product with the mean-pooled A1 embedding
        popularity  top-20 by train clicks
        merge       deduplicated union, capped at 200
      stage 2  re-ranking                   src/baseline/nrms.NRMS (Q3 improved model)
        features    NRMS rows, last-20 history, trailing click counts (+ article age)
        nrms        user encoder over precomputed article vectors, score, gate/sum
        sort        top-10
    response: 10 article ids

`load()` does everything that is paid once at start-up and is never timed:
indexes, models, and every article's NRMS vector (a news service encodes an
article once, when it is published - never per request).

Two stage-1 modes, identical output (tested):
  as_is    calls the existing batch-evaluation functions per request. They rebuild
           state that never changes between requests: a token dict over every
           article (`build_queries`), the pool-sliced BM25 matrix
           (`BM25Index.retrieve(pool=...)`), and a Python scan of all articles to
           filter popularity to the pool (`PopularityRanker.top_k(allowed=...)`).
  serving  computes those three once in `load()`.
Stage 2 is identical in both modes.

Usage
-----
    python src/serving/pipeline.py --config config/mind.yaml            # demo: one request, printed
    python src/serving/pipeline.py --config config/mind.yaml --check    # correctness vs offline scores
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.serving import load_serving_config, pin_threads  # noqa: E402  (sets BLAS thread env first)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.baseline import candidate_signals as sig  # noqa: E402
from src.baseline import nrms  # noqa: E402
from src.baseline.news_data import PAD, last_n_rows, load_news_tokens  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.common.io import read_table  # noqa: E402
from src.retrieval.bm25 import build_queries  # noqa: E402
from src.retrieval.semantic import build_user_vectors  # noqa: E402

MODES = ("as_is", "serving")
STAGES = ("profile", "bm25", "faiss", "popularity", "merge", "features", "nrms", "sort")


@dataclass
class Response:
    article_ids: list[str]                     # top-n, best first
    candidate_ids: list[str]                   # what stage 2 scored
    scores: np.ndarray                         # stage-2 score per candidate
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return sum(self.timings_ms.values())


class ServingPipeline:
    def __init__(self, dataset: str, mode: str = "serving", split: str = "test"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.dataset, self.mode, self.split = dataset, mode, split
        self.conf = load_serving_config()
        self.load_seconds: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # start-up (not timed per request)
    # ------------------------------------------------------------------ #

    def _stopwatch(self, name: str, t0: float) -> float:
        now = time.perf_counter()
        self.load_seconds[name] = round(now - t0, 2)
        return now

    def load(self, impressions: pd.DataFrame | None = None) -> "ServingPipeline":
        from src.rerank.features import build_context
        from src.rerank.retriever import UnionRetriever

        pin_threads(self.conf["hardware"]["threads"])
        torch.set_grad_enabled(False)
        s1, spec = self.conf["pipeline"]["stage1"], self.conf["models"][self.dataset]
        self.cfg = cfg = load_config(REPO_ROOT / "config" / f"{self.dataset}.yaml")
        t = time.perf_counter()

        # --- stage 1: A1/Q2 indexes -------------------------------------------
        self.ctx = ctx = build_context(cfg)
        self.impressions = impressions if impressions is not None else read_table(
            cfg.processed / self.split / "impressions.parquet", "impressions")
        # Pool = articles circulating in this split, exactly as Q2's Universe B.
        self.retriever = UnionRetriever(cfg, ctx.bm25, ctx.embeddings, ctx.articles, self.impressions,
                                        ctx.row_of, pool_name=s1["pool"], k_bm25=s1["k_bm25"],
                                        k_semantic=s1["k_semantic"], k_pop=s1["k_popularity"])
        self.k_total = s1["k_total"]
        self.a1_article_ids = np.asarray(ctx.articles["article_id"])
        profiles = ctx.profiles_all[ctx.profiles_all["split"] == self.split]
        self.clicks_of = dict(zip(profiles["user_id"], profiles["clicked_ids"]))   # the "profile store"
        t = self._stopwatch("stage1_indexes", t)

        if self.mode == "serving":
            r = self.retriever
            self.tokens_by_id = dict(zip(ctx.articles["article_id"], ctx.articles["tokens"]))
            self.bm25_pooled = copy.copy(ctx.bm25)
            self.bm25_pooled.weights_t = ctx.bm25.weights[r.pool_idx].T.tocsr()
            self.pop_ids = r.popularity.top_k(r.k_pop, allowed=r._pop_allowed)
            t = self._stopwatch("stage1_precompute", t)

        # --- stage 2: Q3 model -------------------------------------------------
        self.news = load_news_tokens(cfg)
        self.nrms_row_of = self.news.row_of
        self.signal_names = sig.signal_names(spec["popularity"], spec["freshness"])
        self.model = nrms.NRMS(self.news.tokens, nrms.load_word_embeddings(cfg, self.news),
                               n_signals=len(self.signal_names), gate=spec["gate"]).eval()
        ckpt = torch.load(REPO_ROOT / spec["checkpoint"], map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"], strict=True)
        t = self._stopwatch("stage2_model", t)

        self.news_vecs = nrms.encode_all_news(self.model, torch.device("cpu"))   # (N+1, 400)
        t = self._stopwatch("stage2_encode_all_articles", t)

        self.popularity, self.freshness = spec["popularity"], spec["freshness"]
        self.timeline = sig.build_timeline(cfg, self.nrms_row_of) if self.popularity else None
        if self.freshness:
            arts = read_table(cfg.processed / "articles.parquet", "articles")
            pub_of = dict(zip(arts["article_id"].astype(str), arts["published_time"]))
            pub = pd.to_datetime(pd.Series([pub_of.get(a) for a in self.news.article_ids]))
            # Hours since epoch per NRMS row, NaN where unknown (matches candidate_signals).
            self.pub_hours = pub.to_numpy().astype("datetime64[us]").astype("float64") / 3.6e9
            self.pub_hours[pub.isna().to_numpy()] = np.nan
        self._stopwatch("stage2_signals", t)
        return self

    # ------------------------------------------------------------------ #
    # one request
    # ------------------------------------------------------------------ #

    def retrieve(self, user_id: str, timings: dict) -> list[str]:
        """Stage 1: up to k_total candidate ids for one user."""
        r = self.retriever
        t0 = time.perf_counter()
        clicked = self.clicks_of.get(user_id, [])
        profile = pd.DataFrame({"user_id": [user_id], "clicked_ids": [clicked]})
        t1 = time.perf_counter()

        if self.mode == "as_is":
            _, token_lists = build_queries(profile, self.ctx.articles)
            bm_hits = self.ctx.bm25.retrieve(self.ctx.bm25.query_matrix(token_lists),
                                             k=r.k_bm25, pool=r.pool_idx)[0]
        else:
            tokens = [tok for a in clicked for tok in self.tokens_by_id.get(a, ())]
            local = self.bm25_pooled.retrieve(self.bm25_pooled.query_matrix([tokens]), k=r.k_bm25)[0]
            bm_hits = r.pool_idx[local]
        t2 = time.perf_counter()

        _, user_vec = build_user_vectors(profile, r.row_of, r.embeddings, False, 5.0)
        _, local = r.faiss_index.search(np.ascontiguousarray(user_vec), r.k_semantic)
        sem_hits = r.pool_idx[local[0]]
        t3 = time.perf_counter()

        pop_ids = (r.popularity.top_k(r.k_pop, allowed=r._pop_allowed)
                   if self.mode == "as_is" else self.pop_ids)
        t4 = time.perf_counter()

        union = np.concatenate([self.a1_article_ids[bm_hits], self.a1_article_ids[sem_hits], pop_ids])
        ids = list(dict.fromkeys(union.tolist()))[: self.k_total]
        t5 = time.perf_counter()

        timings.update(profile=1e3 * (t1 - t0), bm25=1e3 * (t2 - t1), faiss=1e3 * (t3 - t2),
                       popularity=1e3 * (t4 - t3), merge=1e3 * (t5 - t4))
        return ids

    def rerank(self, user_id: str, t: np.datetime64, candidate_ids: list[str],
               timings: dict) -> np.ndarray:
        """Stage 2: one score per candidate id, at request time `t`."""
        t0 = time.perf_counter()
        rows = np.fromiter((self.nrms_row_of.get(a, PAD) for a in candidate_ids),
                           dtype=np.int64, count=len(candidate_ids))
        hist = torch.from_numpy(last_n_rows(self.clicks_of.get(user_id), self.nrms_row_of).astype(np.int64))
        feats = None
        if self.signal_names:
            tq = np.full(len(rows), np.datetime64(t, "us"))
            cols = []
            if self.popularity:
                for w in sig.POPULARITY_WINDOWS_HOURS:
                    cols.append(np.log1p(self.timeline.counts_before(rows, tq, w)))
            if self.freshness:
                t_hours = np.datetime64(t, "us").astype("float64") / 3.6e9
                age = t_hours - self.pub_hours[rows]
                cols.append(np.log1p(np.where(np.isnan(age), 0.0, np.maximum(age, 0.0))))
            feats = torch.from_numpy(np.stack(cols, axis=1).astype(np.float32))[None]
        t1 = time.perf_counter()

        cand = torch.from_numpy(rows)
        user = self.model.user_encoder(self.news_vecs[hist][None], (hist != PAD)[None])     # (1, 400)
        content = (self.news_vecs[cand] @ user[0])[None]                                    # (1, C)
        scores = self.model.combine(content, user, feats)[0].numpy()
        t2 = time.perf_counter()

        timings.update(features=1e3 * (t1 - t0), nrms=1e3 * (t2 - t1))
        return scores

    def handle(self, user_id: str, t, candidate_ids: list[str] | None = None) -> Response:
        """Answer one request. `candidate_ids` skips stage 1 (used by the correctness check)."""
        timings: dict[str, float] = {}
        ids = candidate_ids if candidate_ids is not None else self.retrieve(user_id, timings)
        scores = self.rerank(user_id, np.datetime64(t, "us"), ids, timings)
        t0 = time.perf_counter()
        top_n = self.conf["pipeline"]["stage2"]["top_n"]
        order = np.argsort(-scores, kind="stable")[:top_n]
        top = [ids[i] for i in order]
        timings["sort"] = 1e3 * (time.perf_counter() - t0)
        return Response(article_ids=top, candidate_ids=ids, scores=scores, timings_ms=timings)


# --------------------------------------------------------------------------- #
# correctness: serving path vs saved offline scores
# --------------------------------------------------------------------------- #

def check_against_offline(pipe: ServingPipeline, n: int, seed: int = 13) -> dict:
    """Re-score `n` test impressions' inview lists through `handle` and compare with
    the scores `train_nrms.py` saved for the same checkpoint (batch path, trained device)."""
    spec = pipe.conf["models"][pipe.dataset]
    z = np.load(REPO_ROOT / spec["test_scores"])
    imps = pipe.impressions
    if not np.array_equal(imps["impression_id"].to_numpy(), z["impression_id"]):
        raise ValueError("test impressions are not in the order the offline scores were saved in")
    ids_by_row = pipe.news.article_ids

    picks = np.random.default_rng(seed).choice(len(imps), size=min(n, len(imps)), replace=False)
    max_abs, n_cands, top1_agree = 0.0, 0, 0
    for i in picks:
        lo, hi = z["offsets"][i], z["offsets"][i + 1]
        cand_ids = [ids_by_row[r] for r in z["cand_rows"][lo:hi]]
        row = imps.iloc[i]
        got = pipe.handle(row["user_id"], row["timestamp"], candidate_ids=cand_ids).scores
        want = z["scores"][lo:hi]
        max_abs = max(max_abs, float(np.abs(got - want).max()))
        n_cands += hi - lo
        top1_agree += int(np.argmax(got) == np.argmax(want))
    tol = pipe.conf["correctness"]["tolerance"]
    return {"dataset": pipe.dataset, "n_impressions": len(picks), "n_candidates": int(n_cands),
            "max_abs_diff": max_abs, "tolerance": tol, "passed": max_abs <= tol,
            "top1_agreement": top1_agree / len(picks)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, default="serving")
    parser.add_argument("--check", action="store_true", help="compare with saved offline scores")
    args = parser.parse_args(argv)

    dataset = load_config(args.config).dataset
    pipe = ServingPipeline(dataset, mode=args.mode).load()
    print(f"[{dataset}] loaded in {sum(pipe.load_seconds.values()):.1f}s: {pipe.load_seconds}")

    if args.check:
        res = check_against_offline(pipe, pipe.conf["correctness"]["n_impressions"])
        print(json.dumps(res, indent=2))
        out = REPO_ROOT / "reports" / f"q4_correctness_{dataset}.json"
        out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        print(f"-> {out}")
        return 0 if res["passed"] else 1

    row = pipe.impressions.iloc[0]
    for _ in range(3):                      # warm up, then show one request
        resp = pipe.handle(row["user_id"], row["timestamp"])
    titles = dict(zip(pipe.ctx.articles["article_id"], pipe.ctx.articles["title"]))
    print(f"\nrequest: user {row['user_id']} at {row['timestamp']}  "
          f"({len(pipe.clicks_of.get(row['user_id'], []))} clicks in history)")
    print(f"stage 1 -> {len(resp.candidate_ids)} candidates; stage 2 -> top {len(resp.article_ids)}:")
    for rank, a in enumerate(resp.article_ids, 1):
        print(f"  {rank:>2}. {a:<8} {str(titles.get(a, ''))[:80]}")
    print("timings (ms): " + ", ".join(f"{k} {v:.2f}" for k, v in resp.timings_ms.items())
          + f"  | total {resp.total_ms:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
