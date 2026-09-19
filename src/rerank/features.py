"""Q1: the behavioural feature table, and the generic base-scorer over a `CandidateSet`.

Every feature below is serving-legal: nothing here reads a column declared in
`cfg.serving_time_unavailable`, nothing reads the *current* impression's own
outcome (its own read_time/scroll_percentage), and nothing reads a click that
has not happened yet by the impression's own timestamp. That last guarantee
has two different mechanisms behind it, and it matters to know which:

  - click-history features (hist_len, hist_decay_mass, ...) inherit their
    safety from `split.py`'s history-snapshot pairing, exactly as A1's BM25
    and semantic scorers already do - a profile simply cannot contain a future
    click.
  - `rolling_clicks_*` and the session counters in `timeline.py` are a new
    mechanism (a global event timeline queried by `searchsorted(..., t, side=
    "left")`) and carry their own counterfactual test in
    `tests/test_no_leakage.py::test_rolling_popularity_is_strictly_causal`.

A feature that is architecturally unavailable on a dataset (session features on
MIND, freshness/dwell-time on any dataset without the declared capability) is
left NaN, never a fabricated zero - a GBDT splits on NaN natively, and it is
the same `N/A`-not-faked convention A1 uses throughout.

Deliberately excluded: the `hybrid` score. A1's hybrid combiner is fit on
`val`; the re-ranker also trains on `train`/tunes on `val`, so folding a
val-fit score in as a training feature would leak validation information into
what the model is allowed to learn from. `bm25` and `semantic` are included
separately and a re-ranker can recover any linear (or nonlinear) combination
of them on its own.

Deliberately deferred to Q9 (not implemented here, and not requested with
Q1/Q2): a "leaky" feature arm built from `cfg.serving_time_unavailable`
columns for comparison purposes. Nothing in `FEATURE_NAMES` touches those
columns, which is the Q9 requirement on the *default* frame; the explicit
with/without ablation is separate work.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common.config import Config, load_config
from src.common.io import read_table
from src.eval.metrics import category_entropy
from src.retrieval.bm25 import BM25Index, build_queries
from src.retrieval.semantic import (
    build_user_history_rows, encode_articles, l2_normalize,
    load_provided_embeddings, score_topk_similarity,
)
from src.rerank.candidates import CandidateSet
from src.rerank.timeline import RollingPopularity, SessionContext

FEATURE_NAMES = [
    # interaction (A1's base scorers + in-impression ranks)
    "bm25", "semantic", "bm25_rank_in_imp", "semantic_rank_in_imp", "popularity_rank_in_imp",
    # click-history (Q1.1)
    "hist_len", "hist_len_log1p", "hist_n_distinct_categories", "hist_category_entropy",
    "hist_decay_mass", "hours_since_last_click", "mean_hist_read_time", "mean_hist_scroll_pct",
    # session (Q1.2)
    "candidate_position", "candidate_position_norm", "inview_size", "hour_of_day",
    "day_of_week", "session_impression_index", "session_clicks_so_far",
    "seconds_since_session_start",
    # article (Q1.3)
    "train_clicks_log1p", "popularity_rank", "is_head", "rolling_clicks_24h",
    "rolling_clicks_168h", "freshness_hours", "n_tokens",
    # category match with user history (Q1.3)
    "category_match", "category_match_rate",
]

CATEGORICAL_FEATURES = ["hour_of_day", "day_of_week"]  # passed to LightGBM as categorical


@dataclass
class FeatureContext:
    """Everything a feature needs, loaded once per dataset."""
    cfg: Config
    articles: pd.DataFrame          # feature-store articles, one row per article
    profiles_all: pd.DataFrame      # feature-store user_profiles, one row per (user, split)
    embeddings: np.ndarray          # L2-normalised, (n_articles, dim)
    row_of: dict[str, int]
    bm25: BM25Index
    popularity: dict[str, int]
    popularity_rank_of: dict[str, int]
    is_head_of: dict[str, bool]
    category_of: dict[str, str]
    published_time_of: dict[str, "pd.Timestamp | None"]
    n_tokens_of: dict[str, int]
    rolling: RollingPopularity
    session: SessionContext


def build_context(cfg: Config, encode_batch_size: int = 128) -> FeatureContext:
    """Load the feature store, embeddings and every derived index once for a dataset.

    Split-agnostic: articles and embeddings are dataset-wide, so this loads
    once and `build_frame` is called per split against the same context - the
    same "build once, score many splits" shape as `eval.harness.main`.
    """
    articles = pd.read_parquet(cfg.features / "articles.parquet")
    articles["tokens"] = articles["tokens"].map(list)
    profiles_all = pd.read_parquet(cfg.features / "user_profiles.parquet")
    profiles_all["clicked_ids"] = profiles_all["clicked_ids"].map(list)

    article_ids = articles["article_id"].tolist()
    row_of = {a: i for i, a in enumerate(article_ids)}

    raw_vectors = (load_provided_embeddings(cfg, article_ids)
                  if cfg.can("has_provided_embeddings")
                  else encode_articles(cfg, articles, batch_size=encode_batch_size))
    embeddings = l2_normalize(raw_vectors)

    bm25 = BM25Index(article_ids, articles["tokens"].tolist())

    rolling = RollingPopularity.from_impressions({
        split: read_table(cfg.processed / split / "impressions.parquet", "impressions")
        for split in ("train", "val", "test")
    })
    session = SessionContext(cfg)

    return FeatureContext(
        cfg=cfg, articles=articles, profiles_all=profiles_all, embeddings=embeddings,
        row_of=row_of, bm25=bm25,
        popularity=dict(zip(articles["article_id"], articles["train_clicks"])),
        popularity_rank_of=dict(zip(articles["article_id"], articles["popularity_rank"])),
        is_head_of=dict(zip(articles["article_id"], articles["is_head"])),
        category_of=dict(zip(articles["article_id"], articles["category"])),
        published_time_of=dict(zip(articles["article_id"], articles["published_time"])),
        n_tokens_of=dict(zip(articles["article_id"], articles["n_tokens"])),
        rolling=rolling, session=session,
    )


def profiles_for(ctx: FeatureContext, cand: CandidateSet, split_name: str) -> pd.DataFrame:
    """The (user, split) rows this candidate set's users need, pinned to one split.

    A user can appear in more than one split with a genuinely different click
    history (EB-NeRD stages a fresh snapshot per split); `split_name` is what
    keeps `score_base_scorers` and `build_frame` from silently picking whichever
    duplicate row happens to iterate last, mirroring `hybrid.score_split`.
    """
    needed = set(cand.impressions["user_id"])
    return ctx.profiles_all[
        (ctx.profiles_all["split"] == split_name) & (ctx.profiles_all["user_id"].isin(needed))
    ]


def score_base_scorers(ctx: FeatureContext, cand: CandidateSet, profiles: pd.DataFrame,
                       topk: int = 5) -> dict[str, np.ndarray]:
    """random/popularity/bm25/semantic per flat candidate, for either universe.

    Generalises `hybrid.score_split`'s scoring (same primitives: `build_queries`,
    `BM25Index.score_pairs`, `build_user_history_rows`, `score_topk_similarity`)
    from "candidates = this impression's inview list" to "candidates = whatever
    `CandidateSet` holds" - the retrieved-top-K universe needs base scores over
    its own candidates too, and `score_split` is hardcoded to `inview_ids`.
    """
    user_ids, token_lists = build_queries(profiles, ctx.articles)
    user_row = {u: i for i, u in enumerate(user_ids)}
    query_matrix = ctx.bm25.query_matrix(token_lists)

    lengths = np.diff(cand.offsets)
    flat_user_rows = np.repeat(
        [user_row.get(u, -1) for u in cand.impressions["user_id"]], lengths
    )
    valid = (cand.flat_doc_rows >= 0) & (flat_user_rows >= 0)

    n = len(cand.flat_ids)
    scores: dict[str, np.ndarray] = {
        "random": np.random.default_rng(13).random(n).astype(np.float32),
        "popularity": np.array([ctx.popularity.get(a, 0) for a in cand.flat_ids], dtype=np.float32),
    }

    bm = np.zeros(n, dtype=np.float32)
    bm[valid] = ctx.bm25.score_pairs(query_matrix, flat_user_rows[valid], cand.flat_doc_rows[valid])
    scores["bm25"] = bm

    sem = np.zeros(n, dtype=np.float32)
    hist_user_ids, hist_rows_list = build_user_history_rows(profiles, ctx.row_of)
    hist_by_user = dict(zip(hist_user_ids, hist_rows_list))
    for i in range(cand.n_impressions):
        lo, hi = cand.offsets[i], cand.offsets[i + 1]
        hist_rows = hist_by_user.get(cand.impressions["user_id"].iat[i])
        if hist_rows is None or len(hist_rows) == 0:
            continue
        doc_rows = cand.flat_doc_rows[lo:hi]
        ok = doc_rows >= 0
        if not ok.any():
            continue
        seg = sem[lo:hi]
        seg[ok] = score_topk_similarity(ctx.embeddings, doc_rows[ok], hist_rows, topk)
        sem[lo:hi] = seg
    scores["semantic"] = sem
    return scores


def build_frame(ctx: FeatureContext, cand: CandidateSet, profiles: pd.DataFrame,
                history_snapshot: str, base_scores: dict[str, np.ndarray],
                half_life: float = 5.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """-> X float32 (n_candidates, len(FEATURE_NAMES)), y int8, group int32 (n_imp,), names.

    `group` is `np.diff(cand.offsets)` in candidate order - exactly what
    `lgb.Dataset(X, label=y, group=group)` requires, guaranteed by
    `CandidateSet`'s flatten/regroup construction. Never returns a DataFrame:
    object-dtype list columns are what make a large flattened frame cost far
    more memory than the numpy equivalent, and it sidesteps LightGBM's pandas
    interop entirely.
    """
    n = len(cand.flat_ids)
    col = {name: i for i, name in enumerate(FEATURE_NAMES)}
    X = np.full((n, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    y = np.concatenate(cand.labels_by_imp).astype(np.int8) if cand.labels_by_imp else np.zeros(0, dtype=np.int8)
    group = np.diff(cand.offsets).astype(np.int32)
    profile_by_user = {r.user_id: r for r in profiles.itertuples()}

    for i in range(cand.n_impressions):
        lo, hi = cand.offsets[i], cand.offsets[i + 1]
        ids = cand.flat_ids[lo:hi]
        inview_size = hi - lo
        if inview_size == 0:
            continue
        imp_row = cand.impressions.iloc[i]
        user_id = imp_row["user_id"]
        t = pd.Timestamp(imp_row["timestamp"])

        profile = profile_by_user.get(user_id)
        clicked_hist = list(profile.clicked_ids) if profile is not None else []
        hist_len = len(clicked_hist)
        hist_categories = [ctx.category_of.get(a, "") for a in clicked_hist]
        hist_cat_counts = pd.Series(hist_categories).value_counts() if hist_len else None
        hist_cat_set = set(hist_categories)

        if hist_len:
            age = np.arange(hist_len - 1, -1, -1, dtype=np.float32)  # most recent = 0
            decay_mass = float(np.power(0.5, age / half_life).sum())
        else:
            decay_mass = 0.0

        hours_since_last = np.nan
        if (ctx.cfg.can("has_history_timestamps") and profile is not None
                and pd.notna(profile.last_click_time)):
            hours_since_last = (t - profile.last_click_time) / np.timedelta64(1, "h")

        dwell = ctx.session.dwell_features(user_id, history_snapshot)
        sess = ctx.session.session_features(imp_row["source_split"], imp_row["source_impression_id"])

        rolling24 = ctx.rolling.counts_before(ids, t, 24.0)
        rolling168 = ctx.rolling.counts_before(ids, t, 168.0)

        bm_seg = base_scores["bm25"][lo:hi]
        sem_seg = base_scores["semantic"][lo:hi]
        pop_seg = base_scores["popularity"][lo:hi]
        # rank 1 = best; argsort-of-argsort is the standard "rank within group" trick.
        # Rank, not the raw score, is what goes into X for all three base scorers:
        # A1 already measured that blending raw popularity (or raw position) into a
        # semantic score makes ranking *worse* on MIND (design_note.md Sec.2), and a
        # raw unbounded count gives a tree far more candidate split points than a
        # bounded cosine similarity - a well-known split-cardinality bias that
        # reproduced exactly that finding here (train_clicks dominated GBDT gain and
        # AUC dropped) before this was changed to rank-only.
        bm_rank = (-bm_seg).argsort(kind="mergesort").argsort(kind="mergesort") + 1
        sem_rank = (-sem_seg).argsort(kind="mergesort").argsort(kind="mergesort") + 1
        pop_rank = (-pop_seg).argsort(kind="mergesort").argsort(kind="mergesort") + 1

        for j, a in enumerate(ids):
            row = lo + j
            X[row, col["bm25"]] = bm_seg[j]
            X[row, col["semantic"]] = sem_seg[j]
            X[row, col["bm25_rank_in_imp"]] = bm_rank[j]
            X[row, col["semantic_rank_in_imp"]] = sem_rank[j]
            X[row, col["popularity_rank_in_imp"]] = pop_rank[j]

            X[row, col["hist_len"]] = hist_len
            X[row, col["hist_len_log1p"]] = np.log1p(hist_len)
            X[row, col["hist_n_distinct_categories"]] = len(hist_cat_set)
            X[row, col["hist_category_entropy"]] = category_entropy(hist_categories)
            X[row, col["hist_decay_mass"]] = decay_mass
            X[row, col["hours_since_last_click"]] = hours_since_last
            X[row, col["mean_hist_read_time"]] = dwell["mean_hist_read_time"]
            X[row, col["mean_hist_scroll_pct"]] = dwell["mean_hist_scroll_pct"]

            X[row, col["candidate_position"]] = j
            X[row, col["candidate_position_norm"]] = j / max(1, inview_size - 1)
            X[row, col["inview_size"]] = inview_size
            X[row, col["hour_of_day"]] = t.hour
            X[row, col["day_of_week"]] = t.dayofweek
            X[row, col["session_impression_index"]] = sess["session_impression_index"]
            X[row, col["session_clicks_so_far"]] = sess["session_clicks_so_far"]
            X[row, col["seconds_since_session_start"]] = sess["seconds_since_session_start"]

            X[row, col["train_clicks_log1p"]] = np.log1p(ctx.popularity.get(a, 0))
            X[row, col["popularity_rank"]] = ctx.popularity_rank_of.get(a, len(ctx.articles))
            X[row, col["is_head"]] = float(ctx.is_head_of.get(a, False))
            X[row, col["rolling_clicks_24h"]] = rolling24[j]
            X[row, col["rolling_clicks_168h"]] = rolling168[j]
            pub = ctx.published_time_of.get(a)
            if ctx.cfg.can("has_published_time") and pub is not None and pd.notna(pub):
                X[row, col["freshness_hours"]] = (t - pub) / np.timedelta64(1, "h")
            X[row, col["n_tokens"]] = ctx.n_tokens_of.get(a, 0)

            cand_cat = ctx.category_of.get(a, "")
            X[row, col["category_match"]] = float(bool(cand_cat) and cand_cat in hist_cat_set)
            X[row, col["category_match_rate"]] = (
                float(hist_cat_counts.get(cand_cat, 0)) / hist_len
                if hist_len and hist_cat_counts is not None else 0.0
            )

    return X, y, group, FEATURE_NAMES
