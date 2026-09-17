"""Q4 phase 1: the one-request serving path computes the same thing as the offline code.

Latency is only meaningful for the system that was evaluated, so:
  - stage 1 in both modes returns exactly `UnionRetriever.retrieve`'s candidates
  - stage 2 reproduces the scores `train_nrms.py` saved for the same checkpoint
  - a response is well-formed and every stage is timed
Real data only; skipped when the dataset or checkpoints are not built. Each
dataset's pipeline loads once per module (MIND ~16 s, EB-NeRD ~9 s).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.serving import load_serving_config  # noqa: E402
from src.serving.pipeline import STAGES, ServingPipeline, check_against_offline  # noqa: E402


@pytest.fixture(scope="module", params=["mind", "ebnerd"])
def pipe(request):
    spec = load_serving_config()["models"][request.param]
    if not ((REPO_ROOT / spec["checkpoint"]).exists() and (REPO_ROOT / spec["test_scores"]).exists()):
        pytest.skip(f"{request.param}: Q3 checkpoint/test scores not available")
    return ServingPipeline(request.param, mode="serving").load()


def _sample_users(pipe, n=25):
    users = pipe.impressions["user_id"].drop_duplicates().sample(n, random_state=5).tolist()
    return users + ["__user_without_history__"]


def test_stage1_both_modes_match_union_retriever(pipe):
    for user in _sample_users(pipe):
        clicked = pipe.clicks_of.get(user, [])
        profile = pd.DataFrame({"user_id": [user], "clicked_ids": [clicked]})
        reference = pipe.retriever.retrieve(profile, pipe.ctx.articles, k=pipe.k_total)[user]

        pipe.mode = "serving"
        serving = pipe.retrieve(user, {})
        pipe.mode = "as_is"
        as_is = pipe.retrieve(user, {})
        pipe.mode = "serving"

        assert serving == reference, f"serving-mode candidates differ for {user}"
        assert as_is == reference, f"as-is candidates differ for {user}"
        assert 0 < len(serving) <= pipe.k_total


def test_stage2_reproduces_offline_scores(pipe):
    res = check_against_offline(pipe, n=60, seed=7)
    assert res["passed"], f"max |serving - offline| = {res['max_abs_diff']:.2e} > {res['tolerance']}"
    assert res["top1_agreement"] == 1.0


def test_response_is_well_formed_and_fully_timed(pipe):
    row = pipe.impressions.iloc[3]
    resp = pipe.handle(row["user_id"], row["timestamp"])
    top_n = pipe.conf["pipeline"]["stage2"]["top_n"]

    assert len(resp.article_ids) == min(top_n, len(resp.candidate_ids))
    assert set(resp.article_ids) <= set(resp.candidate_ids)
    assert len(resp.scores) == len(resp.candidate_ids) and np.isfinite(resp.scores).all()
    top_scores = [resp.scores[resp.candidate_ids.index(a)] for a in resp.article_ids]
    assert top_scores == sorted(top_scores, reverse=True), "top-n must be in score order"
    assert set(resp.timings_ms) == set(STAGES) and all(v >= 0 for v in resp.timings_ms.values())
