"""Q3 step 2: NRMS model behaviour, on random weights (no download needed).

These pin the properties the design relies on: padding is invisible, the fast
split scorer agrees with the training forward pass, and gradients reach the
word embeddings.
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

from src.baseline.news_data import build_split_tensors  # noqa: E402
from src.baseline.nrms import NRMS, MultiHeadSelfAttention, masked_softmax, score_split  # noqa: E402

N_ARTICLES, TITLE_LEN, VOCAB, EMB = 12, 6, 40, 16


def _model(seed: int = 0) -> NRMS:
    rng = np.random.default_rng(seed)
    tokens = np.zeros((N_ARTICLES + 1, TITLE_LEN), dtype=np.int32)
    for r in range(1, N_ARTICLES + 1):
        n = rng.integers(2, TITLE_LEN + 1)
        tokens[r, :n] = rng.integers(1, VOCAB, size=n)
    emb = rng.normal(size=(VOCAB, EMB)).astype(np.float32)
    emb[0] = 0
    torch.manual_seed(seed)
    return NRMS(tokens, emb).eval()


def test_masked_softmax_zero_on_mask_and_on_empty_rows():
    logits = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, True, False], [False, False, False]])
    w = masked_softmax(logits, mask)
    assert w[0, 2] == 0 and torch.isclose(w[0].sum(), torch.tensor(1.0))
    assert (w[1] == 0).all() and not torch.isnan(w).any()


def test_self_attention_ignores_padded_positions():
    torch.manual_seed(0)
    mhsa = MultiHeadSelfAttention(8, n_heads=2, head_dim=4)
    x = torch.randn(1, 4, 8)
    mask = torch.tensor([[True, True, False, False]])
    x2 = x.clone()
    x2[0, 2:] = torch.randn(2, 8) * 100          # change only the padded slots
    assert torch.allclose(mhsa(x, mask)[0, :2], mhsa(x2, mask)[0, :2], atol=1e-5)


def test_history_padding_does_not_change_scores():
    model = _model()
    cands = torch.tensor([[3, 4, 5]])
    short = model(torch.tensor([[0, 0, 0, 1, 2]]), cands)
    shorter_pad = model(torch.tensor([[0, 1, 2]]), cands)
    assert torch.allclose(short, shorter_pad, atol=1e-5)


def test_empty_history_gives_finite_zero_scores():
    model = _model()
    scores = model(torch.tensor([[0, 0, 0]]), torch.tensor([[1, 2]]))
    assert torch.isfinite(scores).all() and (scores == 0).all()


def test_score_split_matches_forward():
    model = _model()
    imps = pd.DataFrame({
        "user_id": ["u1", "u2", "u1"],
        "inview_ids": [["a1", "a2", "a3"], ["a4", "a5"], ["a6", "a7", "a8", "a9"]],
        "clicked_ids": [["a2"], ["a4"], ["a9"]],
    })
    profiles = pd.DataFrame({"user_id": ["u1", "u2"], "clicked_ids": [["a10", "a11"], ["a12"]]})
    row_of = {f"a{i}": i for i in range(1, N_ARTICLES + 1)}
    t = build_split_tensors("test", imps, profiles, row_of, history_len=4)

    fast = score_split(model, t, torch.device("cpu"))
    for i in range(t.n_impressions):
        rows, _ = t.candidates_of(i)
        slow = model(torch.from_numpy(t.history_of(i)[None].astype(np.int64)),
                     torch.from_numpy(rows[None].astype(np.int64)))[0]
        lo, hi = t.offsets[i], t.offsets[i + 1]
        assert np.allclose(fast[lo:hi], slow.detach().numpy(), atol=1e-4)


def test_gradients_reach_word_embeddings():
    model = _model().train()
    logits = model(torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6]]))
    torch.nn.functional.cross_entropy(logits, torch.tensor([0])).backward()
    grad = model.news_encoder.embedding.weight.grad
    assert grad is not None and grad.abs().sum() > 0
    assert (grad[0] == 0).all(), "padding embedding must stay fixed"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no Apple GPU")
def test_forward_backward_on_mps():
    model = _model().to("mps").train()
    logits = model(torch.tensor([[1, 2, 0]], device="mps"), torch.tensor([[4, 5]], device="mps"))
    torch.nn.functional.cross_entropy(logits, torch.tensor([0], device="mps")).backward()
    assert torch.isfinite(logits).all()
