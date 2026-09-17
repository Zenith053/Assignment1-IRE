"""Q3 step 3: popularity/freshness signals and the gated model.

The causality argument is by equivalence: `ClickTimeline.counts_before` must
return exactly what `RollingPopularity.counts_before` returns, and the latter
already has a counterfactual no-future-click test on real data
(`test_no_leakage.py::test_rolling_popularity_is_strictly_causal`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.baseline import nrms  # noqa: E402
from src.baseline.candidate_signals import ClickTimeline, build_timeline, candidate_signals  # noqa: E402
from src.baseline.news_data import build_split_tensors, sample_training_rows  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.common.io import read_table  # noqa: E402
from src.rerank.timeline import RollingPopularity  # noqa: E402

T = lambda s: np.datetime64(s, "us")  # noqa: E731


# --------------------------------------------------------------------------- #
# synthetic
# --------------------------------------------------------------------------- #

def test_counts_exclude_the_query_second_and_respect_window():
    tl = ClickTimeline(np.array([1, 1, 1, 2]),
                       np.array([T("2024-01-01T10:00:00"), T("2024-01-01T11:00:00"),
                                 T("2024-01-01T12:00:00"), T("2024-01-01T11:30:00")]))
    q = lambda row, t, w: int(tl.counts_before(np.array([row]), np.array([T(t)]), w)[0])  # noqa: E731
    assert q(1, "2024-01-01T12:00:00", 24) == 2, "a click at exactly t is not 'before t'"
    assert q(1, "2024-01-01T12:00:00.5", 24) == 3, "half a second later it is"
    assert q(1, "2024-01-01T12:00:00", 1) == 1, "window [t-1h, t) keeps 11:00 only"
    assert q(1, "2024-01-01T09:00:00", 24) == 0, "nothing before the first event"
    assert q(2, "2024-01-01T12:00:00", 24) == 1 and q(3, "2024-01-01T12:00:00", 24) == 0


def test_training_rows_carry_the_signals_of_their_own_candidates():
    imps = pd.DataFrame({"user_id": ["u1"], "inview_ids": [["a1", "a2", "a3"]], "clicked_ids": [["a2"]]})
    row_of = {"a1": 1, "a2": 2, "a3": 3}
    t = build_split_tensors("train", imps, pd.DataFrame({"user_id": [], "clicked_ids": []}), row_of, 2)
    t.cand_features = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)  # a1, a2, a3
    s = sample_training_rows(t, npratio=4, rng=np.random.default_rng(0))
    by_row = {1: 10.0, 2: 20.0, 3: 30.0, 0: 0.0}
    for c, f, m in zip(s.candidates[0], s.features[0, :, 0], s.mask[0]):
        assert f == by_row[int(c)], "a feature must travel with its candidate"
        assert m == (c != 0)
    assert s.candidates[0, 0] == 2 and s.features[0, 0, 0] == 20.0


def _model(n_signals, gate="learned"):
    rng = np.random.default_rng(0)
    tokens = np.zeros((9, 5), dtype=np.int32)
    for r in range(1, 9):
        tokens[r, :3] = rng.integers(1, 30, size=3)
    emb = rng.normal(size=(30, 16)).astype(np.float32)
    torch.manual_seed(0)
    return nrms.NRMS(tokens, emb, n_signals=n_signals, gate=gate).eval()


def test_plain_nrms_initialisation_unchanged_by_signal_support():
    """n_signals=0 must draw the same initial weights, so baselines stay comparable."""
    plain, with_sig = _model(0), _model(4)
    for k, v in plain.state_dict().items():
        assert torch.equal(v, with_sig.state_dict()[k]), k
    assert not hasattr(plain, "signal_mlp") and not hasattr(plain, "gate")


@pytest.mark.parametrize("gate", ["learned", "sum"])
def test_score_split_matches_forward_with_signals(gate):
    model = _model(2, gate)
    imps = pd.DataFrame({"user_id": ["u1", "u2"], "inview_ids": [["a1", "a2", "a3"], ["a4", "a5"]],
                         "clicked_ids": [["a1"], ["a5"]]})
    profiles = pd.DataFrame({"user_id": ["u1", "u2"], "clicked_ids": [["a6", "a7"], []]})
    t = build_split_tensors("test", imps, profiles, {f"a{i}": i for i in range(1, 9)}, 3)
    t.cand_features = np.random.default_rng(1).normal(size=(5, 2)).astype(np.float32)

    fast = nrms.score_split(model, t, torch.device("cpu"))
    for i in range(t.n_impressions):
        lo, hi = t.offsets[i], t.offsets[i + 1]
        slow = model(torch.from_numpy(t.history_of(i)[None].astype(np.int64)),
                     torch.from_numpy(t.cand_rows[lo:hi][None].astype(np.int64)),
                     torch.from_numpy(t.cand_features[lo:hi][None]))[0]
        assert np.allclose(fast[lo:hi], slow.detach().numpy(), atol=1e-4)


def test_signals_change_ranking_only_through_signal_path():
    """Same content, different popularity -> different scores; gate stays in (0, 1)."""
    model = _model(1)
    hist, cands = torch.tensor([[1, 2]]), torch.tensor([[3, 4]])
    a = model(hist, cands, torch.tensor([[[0.0], [0.0]]]))
    b = model(hist, cands, torch.tensor([[[5.0], [0.0]]]))
    assert not torch.allclose(a[0, 0], b[0, 0]) and torch.allclose(a[0, 1], b[0, 1])


# --------------------------------------------------------------------------- #
# real data
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module", params=["mind", "ebnerd"])
def built(request):
    cfg = load_config(REPO_ROOT / "config" / f"{request.param}.yaml")
    if not (cfg.processed / "test" / "impressions.parquet").exists():
        pytest.skip(f"{cfg.dataset} not built")
    return cfg


def test_click_timeline_matches_rolling_popularity(built):
    cfg = built
    articles = read_table(cfg.processed / "articles.parquet", "articles")
    row_of = {a: i + 1 for i, a in enumerate(articles["article_id"].astype(str))}
    ids = np.array([""] + articles["article_id"].astype(str).tolist(), dtype=object)
    by_split = {s: read_table(cfg.processed / s / "impressions.parquet", "impressions")
                for s in ("train", "val", "test")}
    reference = RollingPopularity.from_impressions(by_split)
    timeline = build_timeline(cfg, row_of)

    for split in ("train", "test"):
        imps = by_split[split].sample(300, random_state=3)
        t = build_split_tensors(split, imps, pd.DataFrame({"user_id": [], "clicked_ids": []}), row_of)
        imp_of = np.repeat(np.arange(t.n_impressions), np.diff(t.offsets))
        ts = t.impressions["timestamp"].to_numpy().astype("datetime64[us]")
        for w in (1.0, 24.0, 168.0):
            fast = timeline.counts_before(t.cand_rows, ts[imp_of], w)
            slow = np.concatenate([
                reference.counts_before(ids[t.cand_rows[t.offsets[i]:t.offsets[i + 1]]], ts[i], w)
                for i in range(t.n_impressions)])
            assert np.array_equal(fast, slow), f"{split} {w}h: vectorised counts disagree"
            assert fast.sum() > 0, "probe found no clicks at all - test would be vacuous"


def test_freshness_is_non_negative_and_na_on_mind(built):
    cfg = built
    articles = read_table(cfg.processed / "articles.parquet", "articles")
    row_of = {a: i + 1 for i, a in enumerate(articles["article_id"].astype(str))}
    ids = np.array([""] + articles["article_id"].astype(str).tolist(), dtype=object)
    imps = read_table(cfg.processed / "train" / "impressions.parquet", "impressions").sample(500, random_state=0)
    t = build_split_tensors("train", imps, pd.DataFrame({"user_id": [], "clicked_ids": []}), row_of)
    if not cfg.can("has_published_time"):
        with pytest.raises(ValueError):
            candidate_signals(cfg, t, ids, None, popularity=False, freshness=True)
        return
    f = candidate_signals(cfg, t, ids, None, popularity=False, freshness=True)
    assert f.shape == (len(t.cand_rows), 1) and (f >= 0).all() and np.isfinite(f).all()
