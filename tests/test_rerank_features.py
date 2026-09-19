"""Unit tests for the A2 feature/candidate pipeline, on synthetic data.

No built dataset required (unlike test_no_leakage.py) - a handful of hand-built
articles, users and impressions exercise the real code path end to end, so
these run the same way `test_metrics.py`'s pure-unit tests do.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.config import Config
from src.eval.metrics import paired_bootstrap_ci
from src.retrieval.bm25 import BM25Index
from src.retrieval.semantic import l2_normalize
from src.rerank.candidates import from_inview, from_retrieval
from src.rerank.features import FEATURE_NAMES, FeatureContext, build_frame, profiles_for, score_base_scorers
from src.rerank.timeline import RollingPopularity, SessionContext

ARTICLE_IDS = ["a1", "a2", "a3", "a4", "a5", "a6"]
CATEGORIES = ["news", "news", "sport", "sport", "tech", "tech"]
TOKENS = [
    ["market", "stocks", "rise"], ["market", "bonds", "fall"],
    ["football", "goal", "win"], ["football", "match", "draw"],
    ["ai", "model", "release"], ["ai", "chip", "launch"],
]


def _make_articles() -> pd.DataFrame:
    return pd.DataFrame({
        "article_id": ARTICLE_IDS,
        "title": [f"title {a}" for a in ARTICLE_IDS],
        "abstract": [f"abstract {a}" for a in ARTICLE_IDS],
        "category": CATEGORIES,
        "published_time": pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"]
        ),
        "tokens": TOKENS,
        "n_tokens": [len(t) for t in TOKENS],
        "train_clicks": [10, 2, 8, 1, 5, 0],
        "popularity_rank": [0, 4, 1, 5, 2, 3],
        "is_head": [True, False, True, False, False, False],
    })


def _make_profiles() -> pd.DataFrame:
    # Two users, both scored on split "test"; u1 has history, u2 is cold.
    return pd.DataFrame({
        "user_id": ["u1", "u2"],
        "split": ["test", "test"],
        "clicked_ids": [["a1", "a3"], []],
        "n_clicks": [2, 0],
        "last_click_time": pd.to_datetime(["2024-01-10 00:00:00", pd.NaT]),
        "is_cold": [False, True],
        "is_low_history": [False, True],
    })


def _make_impressions() -> pd.DataFrame:
    return pd.DataFrame({
        "impression_id": [1, 2],
        "source_impression_id": [101, 102],
        "user_id": ["u1", "u2"],
        "timestamp": pd.to_datetime(["2024-01-10 12:00:00", "2024-01-10 13:00:00"]),
        "inview_ids": [["a2", "a4", "a5"], ["a1", "a6"]],
        "clicked_ids": [["a4"], []],
        "source_split": ["test", "test"],
    })


def _make_context(has_published_time: bool, has_history_timestamps: bool,
                  has_session_id: bool = False) -> FeatureContext:
    articles = _make_articles()
    row_of = {a: i for i, a in enumerate(articles["article_id"])}
    rng = np.random.default_rng(0)
    embeddings = l2_normalize(rng.random((len(articles), 8)).astype(np.float32))
    bm25 = BM25Index(articles["article_id"].tolist(), articles["tokens"].tolist())

    cfg = Config(
        dataset="synthetic", language="en", scale=None, raw={},
        capabilities={"has_published_time": has_published_time,
                     "has_history_timestamps": has_history_timestamps,
                     "has_session_id": has_session_id},
        text={}, split={},
    )
    rolling = RollingPopularity.from_impressions({"test": _make_impressions()})
    session = SessionContext(cfg)  # has_session_id=False here -> no raw file access

    return FeatureContext(
        cfg=cfg, articles=articles, profiles_all=_make_profiles(), embeddings=embeddings,
        row_of=row_of, bm25=bm25,
        popularity=dict(zip(articles["article_id"], articles["train_clicks"])),
        popularity_rank_of=dict(zip(articles["article_id"], articles["popularity_rank"])),
        is_head_of=dict(zip(articles["article_id"], articles["is_head"])),
        category_of=dict(zip(articles["article_id"], articles["category"])),
        published_time_of=dict(zip(articles["article_id"], articles["published_time"])),
        n_tokens_of=dict(zip(articles["article_id"], articles["n_tokens"])),
        rolling=rolling, session=session,
    )


def test_candidate_set_from_inview_shapes():
    impressions = _make_impressions()
    ctx = _make_context(True, True)
    cand = from_inview(impressions, ctx.row_of)

    assert cand.universe == "inview"
    assert cand.n_impressions == 2
    assert len(cand.flat_ids) == 5  # 3 + 2 candidates
    assert cand.ids_by_imp(0) == ["a2", "a4", "a5"]
    # a4 is the only click, and it is the second candidate of impression 0.
    assert cand.labels_by_imp[0].tolist() == [0, 1, 0]
    assert cand.labels_by_imp[1].tolist() == [0, 0]
    assert (cand.flat_doc_rows >= 0).all(), "every synthetic id has a row"


def test_candidate_set_from_retrieval_missing_ids_get_minus_one():
    impressions = _make_impressions()
    ctx = _make_context(True, True)
    retrieved = {"u1": ["a1", "a6", "unknown_id"], "u2": ["a3"]}
    cand = from_retrieval(impressions, retrieved, ctx.row_of)

    assert cand.universe == "retrieved"
    assert cand.n_impressions == 2
    assert cand.flat_doc_rows.tolist() == [
        ctx.row_of["a1"], ctx.row_of["a6"], -1, ctx.row_of["a3"],
    ]
    # u1's retrieved list has no click (clicked_ids=["a4"]) -> all zero.
    assert cand.labels_by_imp[0].sum() == 0
    # u2's retrieved list is empty of clicks too (clicked_ids=[]).
    assert cand.labels_by_imp[1].sum() == 0


def test_build_frame_shapes_and_group_matches_flatten():
    ctx = _make_context(has_published_time=True, has_history_timestamps=True)
    impressions = _make_impressions()
    cand = from_inview(impressions, ctx.row_of)
    profiles = profiles_for(ctx, cand, "test")
    base = score_base_scorers(ctx, cand, profiles, topk=5)
    X, y, group, names = build_frame(ctx, cand, profiles, "test", base)

    assert names == FEATURE_NAMES
    assert X.shape == (5, len(FEATURE_NAMES))
    assert y.shape == (5,)
    assert y.tolist() == [0, 1, 0, 0, 0]
    assert group.tolist() == [3, 2]
    assert group.sum() == X.shape[0]


def test_build_frame_category_match_is_correct():
    """u1's history is [a1, a3] (categories news, sport); scoring [a2,a4,a5]
    (news, sport, tech) - a2 and a4 should match, a5 should not."""
    ctx = _make_context(has_published_time=True, has_history_timestamps=True)
    impressions = _make_impressions()
    cand = from_inview(impressions, ctx.row_of)
    profiles = profiles_for(ctx, cand, "test")
    base = score_base_scorers(ctx, cand, profiles, topk=5)
    X, y, group, names = build_frame(ctx, cand, profiles, "test", base)

    col = names.index("category_match")
    # impression 0 (u1) candidates are a2 (news), a4 (sport), a5 (tech).
    assert X[0:3, col].tolist() == [1.0, 1.0, 0.0]
    # impression 1 (u2, cold, empty history) candidates are a1, a6 - no match possible.
    assert X[3:5, col].tolist() == [0.0, 0.0]


def test_build_frame_session_features_are_nan_when_unavailable():
    ctx = _make_context(has_published_time=True, has_history_timestamps=True, has_session_id=False)
    impressions = _make_impressions()
    cand = from_inview(impressions, ctx.row_of)
    profiles = profiles_for(ctx, cand, "test")
    base = score_base_scorers(ctx, cand, profiles, topk=5)
    X, y, group, names = build_frame(ctx, cand, profiles, "test", base)

    for name in ("session_impression_index", "session_clicks_so_far",
                "seconds_since_session_start", "mean_hist_read_time", "mean_hist_scroll_pct"):
        col = names.index(name)
        assert np.isnan(X[:, col]).all(), f"{name} should be NaN when has_session_id=False"


def test_build_frame_freshness_is_nan_when_capability_absent():
    ctx = _make_context(has_published_time=False, has_history_timestamps=False)
    impressions = _make_impressions()
    cand = from_inview(impressions, ctx.row_of)
    profiles = profiles_for(ctx, cand, "test")
    base = score_base_scorers(ctx, cand, profiles, topk=5)
    X, y, group, names = build_frame(ctx, cand, profiles, "test", base)

    assert np.isnan(X[:, names.index("freshness_hours")]).all()
    assert np.isnan(X[:, names.index("hours_since_last_click")]).all()
    # But an always-available feature must still be real, not NaN.
    assert not np.isnan(X[:, names.index("hist_len")]).any()


def test_paired_bootstrap_ci_identical_arrays_include_zero():
    values = [0.7, 0.6, 0.8, 0.5, 0.9] * 10
    res = paired_bootstrap_ci(values, values, n_boot=500, seed=1)
    assert res["mean_diff"] == pytest.approx(0.0)
    assert not res["excludes_zero"]
    assert res["n_paired"] == len(values)


def test_paired_bootstrap_ci_detects_a_real_gain():
    rng = np.random.default_rng(0)
    before = rng.normal(0.5, 0.05, size=500)
    after = before + 0.10  # a real, consistent per-impression gain
    res = paired_bootstrap_ci(list(after), list(before), n_boot=2000, seed=1)
    assert res["mean_diff"] == pytest.approx(0.10, abs=0.01)
    assert res["excludes_zero"]
    assert res["ci_low"] > 0


def test_paired_bootstrap_ci_skips_undefined_entries():
    a = [0.9, None, 0.8]
    b = [0.5, 0.5, None]
    res = paired_bootstrap_ci(a, b, n_boot=100, seed=1)
    assert res["n_paired"] == 1  # only index 0 has both defined
    assert res["mean_diff"] == pytest.approx(0.4)
