"""Q3 step 1: the NRMS data arrays.

Synthetic tests pin down the indexing contract (padding, ordering, negative
sampling). The real-data tests check the one property that matters for Q9:
the history NRMS reads for a split is exactly that split's leakage-safe
profile, and on EB-NeRD every history click predates every impression.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.baseline.news_data import (  # noqa: E402
    PAD, build_split_tensors, last_n_rows, sample_training_rows, tokenize_titles,
)
from src.common.config import load_config  # noqa: E402
from src.common.io import read_table  # noqa: E402


class FakeTokenizer:
    """Whitespace tokenizer with a stable id per word, HuggingFace call shape."""
    pad_token_id = 1

    def __init__(self):
        self.ids: dict[str, int] = {}

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[self.ids.setdefault(w, 100 + len(self.ids)) for w in t.split()]
                              for t in texts]}


ROW_OF = {"a1": 1, "a2": 2, "a3": 3, "a4": 4, "a5": 5, "a6": 6}


def _impressions(rows):
    return pd.DataFrame(rows, columns=["user_id", "inview_ids", "clicked_ids"])


# --------------------------------------------------------------------------- #
# synthetic
# --------------------------------------------------------------------------- #

def test_tokenize_pads_truncates_and_remaps():
    news = tokenize_titles(["a1", "a2"], ["stocks rise today", "stocks fall"],
                           FakeTokenizer(), title_len=2)
    assert news.tokens.shape == (3, 2)
    assert (news.tokens[PAD] == 0).all(), "row 0 must be the padding article"
    # truncated to 2 tokens; shared word "stocks" gets the same compact id
    assert news.tokens[1, 0] == news.tokens[2, 0] != 0
    assert news.tokens[2, 1] != 0 and news.lengths[1] == 3
    # compact ids index into vocab, and vocab maps back to tokenizer ids
    assert news.vocab[0] == FakeTokenizer.pad_token_id
    assert set(news.vocab[news.tokens[1:].ravel()]) <= set(range(100, 200))
    # "today" was truncated away, so it must not occupy a vocab slot
    assert len(news.vocab) == 1 + 3
    assert news.row_of == {"a1": 1, "a2": 2}


def test_last_n_rows_keeps_most_recent_left_padded():
    assert last_n_rows(["a1", "a2", "a3"], ROW_OF, n=2).tolist() == [2, 3]
    assert last_n_rows(["a1"], ROW_OF, n=3).tolist() == [0, 0, 1]
    assert last_n_rows([], ROW_OF, n=2).tolist() == [0, 0]
    assert last_n_rows(None, ROW_OF, n=2).tolist() == [0, 0]
    # unknown ids are dropped before truncation, not counted as history
    assert last_n_rows(["a1", "zzz", "a2"], ROW_OF, n=2).tolist() == [1, 2]


def test_build_split_tensors_offsets_labels_and_histories():
    imps = _impressions([
        ("u1", ["a1", "a2", "a3"], ["a2"]),
        ("u2", ["a4", "a5"], ["a4", "a5"]),
        ("u1", ["a6", "zzz"], ["a6"]),
    ])
    profiles = pd.DataFrame({"user_id": ["u1"], "split": ["train"],
                             "clicked_ids": [["a3", "a4"]]})
    t = build_split_tensors("train", imps, profiles, ROW_OF, history_len=3)

    assert t.n_impressions == 3
    assert t.offsets.tolist() == [0, 3, 5, 6], "unknown candidate ids are dropped"
    rows, labels = t.candidates_of(0)
    assert rows.tolist() == [1, 2, 3] and labels.tolist() == [0, 1, 0]
    assert t.history_of(0).tolist() == [0, 3, 4]
    assert t.history_of(2).tolist() == t.history_of(0).tolist(), "same user, same history"
    assert t.history_of(1).tolist() == [0, 0, 0], "user without a profile gets padding"


def test_build_split_tensors_rejects_other_splits_profiles():
    imps = _impressions([("u1", ["a1", "a2"], ["a1"])])
    profiles = pd.DataFrame({"user_id": ["u1", "u1"], "split": ["train", "test"],
                             "clicked_ids": [["a3"], ["a4"]]})
    with pytest.raises(ValueError):
        build_split_tensors("train", imps, profiles, ROW_OF)


def test_sample_training_rows_positive_first_negatives_from_same_impression():
    imps = _impressions([
        ("u1", ["a1", "a2", "a3", "a4", "a5", "a6"], ["a1", "a2"]),  # 2 clicks, 4 non-clicks
        ("u2", ["a3", "a4"], ["a3"]),                                # 1 non-click -> padded
        ("u3", ["a5"], ["a5"]),                                      # no non-click -> skipped
    ])
    t = build_split_tensors("train", imps, pd.DataFrame({"user_id": [], "clicked_ids": []}),
                            ROW_OF, history_len=2)
    s = sample_training_rows(t, npratio=3, rng=np.random.default_rng(7))

    assert s.candidates.shape == (3, 4) and s.mask.shape == (3, 4)
    assert s.n_skipped == 1
    assert s.candidates[:2, 0].tolist() == [1, 2], "column 0 is the clicked article"
    for r in range(2):
        negs = s.candidates[r, 1:]
        assert set(negs) <= {3, 4, 5, 6} and len(set(negs)) == 3, "distinct non-clicks only"
        assert s.mask[r].all()
    assert s.candidates[2].tolist() == [3, 4, 0, 0]
    assert s.mask[2].tolist() == [True, True, False, False]


def test_sample_training_rows_resamples_with_rng():
    imps = _impressions([("u1", ["a1", "a2", "a3", "a4", "a5", "a6"], ["a1"])] * 20)
    t = build_split_tensors("train", imps, pd.DataFrame({"user_id": [], "clicked_ids": []}),
                            ROW_OF, history_len=2)
    a = sample_training_rows(t, npratio=2, rng=np.random.default_rng(1)).candidates
    b = sample_training_rows(t, npratio=2, rng=np.random.default_rng(1)).candidates
    c = sample_training_rows(t, npratio=2, rng=np.random.default_rng(2)).candidates
    assert (a == b).all(), "same seed must reproduce"
    assert not (a == c).all(), "a new epoch seed must draw new negatives"


# --------------------------------------------------------------------------- #
# real data
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module", params=["mind", "ebnerd"])
def built(request):
    cfg = load_config(REPO_ROOT / "config" / f"{request.param}.yaml")
    if not (cfg.features / "user_profiles.parquet").exists():
        pytest.skip(f"{cfg.dataset} feature store not built; run make data")
    meta = json.loads((cfg.processed / "split_meta.json").read_text())
    return cfg, meta


def test_nrms_history_is_that_splits_profile(built):
    """The history row NRMS reads equals the tail of that split's own profile."""
    cfg, meta = built
    articles = read_table(cfg.processed / "articles.parquet", "articles")
    row_of = {a: i + 1 for i, a in enumerate(articles["article_id"].astype(str))}
    profiles = pd.read_parquet(cfg.features / "user_profiles.parquet")

    for split in meta["splits"]:
        imps = read_table(cfg.processed / split / "impressions.parquet", "impressions")
        imps = imps.sample(min(300, len(imps)), random_state=0)
        own = profiles[profiles["split"] == split]
        t = build_split_tensors(split, imps, own, row_of)

        clicks_of = dict(zip(own["user_id"].astype(str), own["clicked_ids"]))
        for i, user in enumerate(t.impressions["user_id"].astype(str)):
            expected = last_n_rows(clicks_of.get(user), row_of)
            assert (t.history_of(i) == expected).all(), f"{split}: history mismatch for {user}"


def test_nrms_history_clicks_predate_impressions(built):
    """EB-NeRD: every article in the NRMS history was clicked before the impression."""
    cfg, meta = built
    if not cfg.can("has_history_timestamps"):
        pytest.skip(f"{cfg.dataset}: has_history_timestamps is false (N/A)")
    articles = read_table(cfg.processed / "articles.parquet", "articles")
    row_of = {a: i + 1 for i, a in enumerate(articles["article_id"].astype(str))}
    profiles = pd.read_parquet(cfg.features / "user_profiles.parquet")
    history = read_table(cfg.processed / "history.parquet", "history")

    for split, info in meta["splits"].items():
        snap = history[history["snapshot"] == info["history_snapshot"]]
        last_seen = snap.groupby(["user_id", "article_id"])["timestamp"].max()
        imps = read_table(cfg.processed / split / "impressions.parquet", "impressions")
        imps = imps.sample(min(500, len(imps)), random_state=1)
        t = build_split_tensors(split, imps, profiles[profiles["split"] == split], row_of)

        for i, (user, stamp) in enumerate(zip(t.impressions["user_id"].astype(str),
                                              t.impressions["timestamp"])):
            for row in t.history_of(i)[t.history_of(i) != PAD]:
                clicked_at = last_seen.loc[(user, articles["article_id"].iloc[row - 1])]
                assert clicked_at < stamp, f"{split}: future click in history for {user}"
