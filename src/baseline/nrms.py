"""Q3 step 2: NRMS (Wu et al., EMNLP 2019) in PyTorch.

    title tokens --embedding--> word vectors --multi-head self-attention-->
    contextual word vectors --additive attention--> news vector          (NewsEncoder)

    last-20 news vectors --multi-head self-attention--> contextual click
    vectors --additive attention--> user vector                           (UserEncoder)

    click score = user vector . candidate news vector                     (NRMS.forward)

Sizes follow the ebnerd-benchmark NRMS hyperparameters: 20 heads x 20 dims =
400-d news/user vectors, 200-d additive-attention query, dropout 0.2 on word
embeddings and on the news encoder's self-attention output, trainable 768-d
xlm-roberta-base word embeddings.

One deliberate deviation from the reference Keras implementation: padding is
masked out of every softmax. The reference lets padding tokens and padded
history slots take attention weight; on MIND, where 59% of test users have
fewer than 20 clicks, that would mean most user vectors partly average over
empty slots. Masking changes no hyperparameter and is noted in the report.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.baseline.news_data import PAD, TOKENIZER_NAME, NewsTokens

N_HEADS = 20
HEAD_DIM = 20
ATTENTION_HIDDEN = 200
DROPOUT = 0.2
EMBED_CACHE = "nrms_word_embeddings.npz"


# --------------------------------------------------------------------------- #
# pretrained word embeddings
# --------------------------------------------------------------------------- #

def load_word_embeddings(cfg, news: NewsTokens) -> np.ndarray:
    """(V, 768) rows of xlm-roberta-base's input embedding matrix, in compact-vocab order.

    Only the embedding matrix is used (not the 12 transformer layers), exactly
    as the ebnerd-benchmark NRMS does: the pretrained model supplies a good
    starting point for each token's vector, and NRMS's own attention layers do
    the rest. Row 0 (padding) is zeroed.
    """
    cache = cfg.features / EMBED_CACHE
    if cache.exists():
        z = np.load(cache)
        if np.array_equal(z["vocab"], news.vocab):
            return z["weights"]

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    path = hf_hub_download(TOKENIZER_NAME, "model.safetensors")
    with safe_open(path, framework="np") as f:
        key = next(k for k in f.keys() if k.endswith("embeddings.word_embeddings.weight"))
        full = f.get_tensor(key)                       # (250002, 768)
    weights = full[news.vocab].astype(np.float32)
    weights[PAD] = 0.0
    del full
    np.savez(cache, vocab=news.vocab, weights=weights)
    return weights


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #

def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dim that gives masked positions exactly zero weight.

    A fully masked row (a padding article with no tokens, or a user with no
    history) returns all zeros instead of NaN, so its pooled vector is zero.
    """
    logits = logits.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(logits, dim=-1)
    return torch.where(mask.any(dim=-1, keepdim=True), weights, torch.zeros_like(weights))


class MultiHeadSelfAttention(nn.Module):
    """Every position becomes a weighted mix of all (unmasked) positions, n_heads times.

    Per head: Q = W_q x, K = W_k x, V = W_v x; weights = softmax(Q K^T / sqrt(d));
    output = weights V. Heads are concatenated (n_heads * head_dim wide).
    """

    def __init__(self, in_dim: int, n_heads: int = N_HEADS, head_dim: int = HEAD_DIM):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, head_dim
        out = n_heads * head_dim
        self.q = nn.Linear(in_dim, out, bias=False)
        self.k = nn.Linear(in_dim, out, bias=False)
        self.v = nn.Linear(in_dim, out, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape

        def split(t):  # (b, n, heads*dim) -> (b, heads, n, dim)
            return t.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)   # (b, heads, n, n)
        weights = masked_softmax(logits, mask[:, None, None, :])       # attend only to real positions
        return (weights @ v).transpose(1, 2).reshape(b, n, -1)


class AdditiveAttention(nn.Module):
    """Pool n vectors into one with learned importance weights.

    score_i = q . tanh(W h_i + b);  weights = softmax(score);  out = sum_i weights_i h_i
    """

    def __init__(self, in_dim: int, hidden: int = ATTENTION_HIDDEN):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.query = nn.Linear(hidden, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.query(torch.tanh(self.proj(x))).squeeze(-1)   # (b, n)
        weights = masked_softmax(scores, mask)
        return (weights.unsqueeze(-1) * x).sum(dim=1)


class NewsEncoder(nn.Module):
    """Title tokens (b, title_len) -> news vector (b, 400)."""

    def __init__(self, embeddings: np.ndarray):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(embeddings), freeze=False, padding_idx=PAD)
        self.dropout = nn.Dropout(DROPOUT)
        self.self_attention = MultiHeadSelfAttention(embeddings.shape[1])
        self.pool = AdditiveAttention(N_HEADS * HEAD_DIM)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens != PAD
        x = self.dropout(self.embedding(tokens))
        x = self.dropout(self.self_attention(x, mask))
        return self.pool(x, mask)


class UserEncoder(nn.Module):
    """History news vectors (b, history_len, 400) -> user vector (b, 400)."""

    def __init__(self):
        super().__init__()
        dim = N_HEADS * HEAD_DIM
        self.self_attention = MultiHeadSelfAttention(dim)
        self.pool = AdditiveAttention(dim)

    def forward(self, history_vecs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.pool(self.self_attention(history_vecs, mask), mask)


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #

SIGNAL_HIDDEN = 64
GATES = ("learned", "sum")


class NRMS(nn.Module):
    """Holds the title-token table so callers pass article rows, never text.

    With `n_signals=0` this is plain NRMS. With `n_signals>0` it is the Q3
    improvement (after PP-Rec, Qi et al. 2021): a second, content-free score
    from each candidate's popularity/freshness numbers, blended with the NRMS
    content score by a per-user gate:

        content = user . news                     (what the user is into)
        signal  = MLP(popularity, freshness)      (what is hot and new right now)
        g       = sigmoid(w . user + b)           (how much *this* user follows the crowd)
        score   = (1 - g) * content + g * signal  (gate="learned")
        score   = content + signal                (gate="sum": the no-gate ablation)
    """

    def __init__(self, title_tokens: np.ndarray, embeddings: np.ndarray,
                 n_signals: int = 0, gate: str = "learned"):
        super().__init__()
        if gate not in GATES:
            raise ValueError(f"gate must be one of {GATES}")
        self.register_buffer("title_tokens", torch.from_numpy(title_tokens.astype(np.int64)),
                             persistent=False)
        self.news_encoder = NewsEncoder(embeddings)
        self.user_encoder = UserEncoder()
        # Created only when used, after the encoders: plain NRMS draws exactly the
        # same initial weights as before, and its saved checkpoints still load.
        self.n_signals, self.gate_kind = n_signals, gate
        if n_signals:
            self.signal_mlp = nn.Sequential(
                nn.Linear(n_signals, SIGNAL_HIDDEN), nn.ReLU(), nn.Linear(SIGNAL_HIDDEN, 1))
            if gate == "learned":
                self.gate = nn.Linear(N_HEADS * HEAD_DIM, 1)

    def encode_news(self, rows: torch.Tensor) -> torch.Tensor:
        return self.news_encoder(self.title_tokens[rows])

    def combine(self, content: torch.Tensor, user: torch.Tensor,
                signals: torch.Tensor | None) -> torch.Tensor:
        """content (..., C) scores + signals (..., C, F) -> final scores; user (..., 400)."""
        if not self.n_signals:
            return content
        signal = self.signal_mlp(signals).squeeze(-1)
        if self.gate_kind == "sum":
            return content + signal
        g = torch.sigmoid(self.gate(user))          # (..., 1), broadcasts over candidates
        return (1 - g) * content + g * signal

    def forward(self, history: torch.Tensor, candidates: torch.Tensor,
                signals: torch.Tensor | None = None) -> torch.Tensor:
        """history (b, H) and candidates (b, C) article rows -> click logits (b, C).

        Each distinct article in the batch is encoded once: with b=32, H=20,
        C=5 that is at most 800 titles, usually far fewer after de-duplication.
        """
        b, h = history.shape
        rows, inverse = torch.unique(torch.cat([history.reshape(-1), candidates.reshape(-1)]),
                                     return_inverse=True)
        vecs = self.encode_news(rows)[inverse]
        hist_vecs = vecs[: b * h].view(b, h, -1)
        cand_vecs = vecs[b * h:].view(b, candidates.shape[1], -1)
        user = self.user_encoder(hist_vecs, history != PAD)
        content = (cand_vecs @ user.unsqueeze(-1)).squeeze(-1)
        return self.combine(content, user, signals)


# --------------------------------------------------------------------------- #
# inference over a whole split
# --------------------------------------------------------------------------- #

@torch.no_grad()
def encode_all_news(model: NRMS, device, batch_size: int = 2048) -> torch.Tensor:
    """Every article's vector once: (N+1, 400), row 0 = zeros (padding article)."""
    model.eval()
    n = model.title_tokens.shape[0]
    out = [model.encode_news(torch.arange(i, min(i + batch_size, n), device=device))
           for i in range(0, n, batch_size)]
    table = torch.cat(out)
    table[PAD] = 0.0
    return table


@torch.no_grad()
def score_split(model: NRMS, tensors, device, batch_size: int = 4096) -> np.ndarray:
    """Click scores for every candidate in a `SplitTensors`, aligned to `cand_rows`.

    Inference never re-encodes a title: news vectors are computed once, then
    each impression needs one user-encoder pass and a dot product per candidate.
    """
    model.eval()
    news = encode_all_news(model, device)
    user_hist = torch.from_numpy(tensors.user_history.astype(np.int64)).to(device)

    users = []
    for i in range(0, len(user_hist), batch_size):
        h = user_hist[i:i + batch_size]
        users.append(model.user_encoder(news[h], h != PAD))
    users = torch.cat(users)                                              # (U, 400)

    cand = torch.from_numpy(tensors.cand_rows.astype(np.int64)).to(device)
    imp_of_cand = np.repeat(np.arange(tensors.n_impressions), np.diff(tensors.offsets))
    user_of_cand = torch.from_numpy(tensors.user_index[imp_of_cand].astype(np.int64)).to(device)

    feats = (torch.from_numpy(tensors.cand_features).to(device)
             if model.n_signals else None)
    if model.n_signals and (feats is None or feats.shape[1] != model.n_signals):
        raise ValueError("model expects candidate signals the split tensors do not carry")

    scores, step = [], 65536
    for i in range(0, len(cand), step):
        u = users[user_of_cand[i:i + step]]
        content = (news[cand[i:i + step]] * u).sum(-1)
        # Each candidate is its own "list of one" for combine(): (n,) -> (n, 1).
        s = model.combine(content[:, None], u, feats[i:i + step, None] if feats is not None else None)
        scores.append(s[:, 0])
    return torch.cat(scores).float().cpu().numpy() if scores else np.zeros(0, np.float32)


def save(model: NRMS, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)
