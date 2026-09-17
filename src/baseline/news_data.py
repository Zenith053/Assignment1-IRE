#!/usr/bin/env python3
"""Q3 step 1: the integer arrays NRMS trains and scores on.

NRMS never sees a string. Everything it needs reduces to three lookups:

  article row  -> TITLE_LEN token ids          (`NewsTokens.tokens`)
  user         -> last HISTORY_LEN article rows (`SplitTensors.user_history`)
  impression   -> candidate article rows + click labels

Row 0 of the article table is a padding article (all-zero tokens) and token id
0 is the padding token, so "no article here" and "no word here" are both just
index 0 and the model can mask them uniformly.

Hyperparameters follow the ebnerd-benchmark NRMS configuration (title length
30, history length 20, 4 sampled negatives per click, xlm-roberta-base word
embeddings) so the baseline is a reproduction rather than a re-tune.

Leakage: histories come from `feature_store/<ds>/user_profiles.parquet`, which
`feature_store.build_user_profiles` builds per split from the history snapshot
that `split.py` certified as ending before that split starts. This module only
ever selects the row for the split being built; `tests/test_nrms_data.py`
checks that against real data.

Usage
-----
    python src/baseline/news_data.py --config config/mind.yaml     # build cache + print stats
    python src/baseline/news_data.py --config config/ebnerd.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.common.config import Config, load_config  # noqa: E402
from src.common.io import read_table, to_str_list  # noqa: E402

TOKENIZER_NAME = "FacebookAI/xlm-roberta-base"
TITLE_LEN = 30
HISTORY_LEN = 20
NPRATIO = 4
PAD = 0  # padding article row and padding token id

TOKEN_CACHE = "nrms_title_tokens.npz"


# --------------------------------------------------------------------------- #
# articles -> tokens
# --------------------------------------------------------------------------- #

@dataclass
class NewsTokens:
    """Tokenised titles over a compact vocabulary.

    xlm-roberta has 250k tokens but a news corpus uses a few tens of thousands,
    so ids are remapped: `vocab[compact_id]` is the tokenizer's id, which is
    what step 2 uses to slice only the needed rows out of the 250k x 768
    pretrained embedding matrix.
    """
    article_ids: np.ndarray   # object (N+1,), article_ids[0] = "" (padding article)
    tokens: np.ndarray        # int32 (N+1, title_len), compact ids, 0 = padding
    vocab: np.ndarray         # int64 (V,), tokenizer id per compact id; vocab[0] = tokenizer pad
    lengths: np.ndarray       # int32 (N+1,), untruncated token count (for stats only)

    @property
    def row_of(self) -> dict[str, int]:
        return {a: i for i, a in enumerate(self.article_ids) if i != PAD}


def tokenize_titles(article_ids: list[str], titles: list[str], tokenizer,
                    title_len: int = TITLE_LEN) -> NewsTokens:
    """Tokenise every title and remap to a compact vocabulary.

    `tokenizer` is any HuggingFace-style callable returning {"input_ids": [...]}.
    Special tokens (<s>, </s>) are left out: NRMS pools over words, and a
    constant token at every title's edge only adds a position it must learn to
    ignore.
    """
    encoded = tokenizer(list(titles), add_special_tokens=False)["input_ids"]
    lengths = np.array([0] + [len(ids) for ids in encoded], dtype=np.int32)

    used = sorted({t for ids in encoded for t in ids[:title_len]})
    pad_id = getattr(tokenizer, "pad_token_id", None)
    vocab = np.array([pad_id if pad_id is not None else -1] + used, dtype=np.int64)
    compact = {tok: i + 1 for i, tok in enumerate(used)}

    tokens = np.zeros((len(encoded) + 1, title_len), dtype=np.int32)
    for row, ids in enumerate(encoded, start=1):
        ids = ids[:title_len]
        tokens[row, :len(ids)] = [compact[t] for t in ids]

    return NewsTokens(
        article_ids=np.array([""] + list(article_ids), dtype=object),
        tokens=tokens, vocab=vocab, lengths=lengths,
    )


def _title_text(articles: pd.DataFrame) -> list[str]:
    return articles["title"].fillna("").astype(str).tolist()


def load_news_tokens(cfg: Config, title_len: int = TITLE_LEN,
                     rebuild: bool = False) -> NewsTokens:
    """Tokenise the processed article table once and cache it next to the feature store."""
    articles = read_table(cfg.processed / "articles.parquet", "articles")
    article_ids = articles["article_id"].astype(str).tolist()
    cache = cfg.features / TOKEN_CACHE
    # Cheap invalidation: a rebuilt article table or a changed setting changes one of these.
    key = f"{TOKENIZER_NAME}|{title_len}|{len(article_ids)}|{article_ids[0]}|{article_ids[-1]}"

    if cache.exists() and not rebuild:
        try:
            # No pickled objects in the cache, so it loads under any numpy version.
            z = np.load(cache, allow_pickle=False)
            if str(z["key"]) == key:
                return NewsTokens(article_ids=z["article_ids"].astype(object), tokens=z["tokens"],
                                  vocab=z["vocab"], lengths=z["lengths"])
        except ValueError:
            pass  # cache written by an older, pickled format -> rebuild

    from transformers import AutoTokenizer  # heavy import, only on a cache miss
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    news = tokenize_titles(article_ids, _title_text(articles), tokenizer, title_len)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, key=np.str_(key), article_ids=news.article_ids.astype(str), tokens=news.tokens,
             vocab=news.vocab, lengths=news.lengths)
    return news


# --------------------------------------------------------------------------- #
# users + impressions -> index arrays
# --------------------------------------------------------------------------- #

def last_n_rows(clicked_ids, row_of: dict[str, int], n: int = HISTORY_LEN) -> np.ndarray:
    """The user's most recent `n` clicks as article rows, left-padded with 0.

    Most recent click is last. Ids missing from the article table are dropped
    before truncating, so a user keeps `n` real clicks where they have them.
    """
    rows = [row_of[a] for a in to_str_list(clicked_ids) if a in row_of][-n:]
    out = np.zeros(n, dtype=np.int32)
    if rows:
        out[n - len(rows):] = rows
    return out


@dataclass
class SplitTensors:
    """One split's impressions, flattened the same way `CandidateSet` is.

    Impression i's candidates are `cand_rows[offsets[i]:offsets[i+1]]`, and its
    history is `user_history[user_index[i]]` (stored per user, not per
    impression, since a user's history is fixed within a split).
    """
    split: str
    impressions: pd.DataFrame   # source rows, same order as offsets
    user_index: np.ndarray      # int32 (M,)
    user_history: np.ndarray    # int32 (U, history_len)
    offsets: np.ndarray         # int64 (M+1,)
    cand_rows: np.ndarray       # int32 (C,)
    labels: np.ndarray          # int8 (C,)
    cand_features: np.ndarray | None = None   # float32 (C, F), Step 3 signals, attached by the caller

    @property
    def n_impressions(self) -> int:
        return len(self.offsets) - 1

    def history_of(self, i: int) -> np.ndarray:
        return self.user_history[self.user_index[i]]

    def candidates_of(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.offsets[i], self.offsets[i + 1]
        return self.cand_rows[lo:hi], self.labels[lo:hi]


def build_split_tensors(split: str, impressions: pd.DataFrame, profiles: pd.DataFrame,
                        row_of: dict[str, int], history_len: int = HISTORY_LEN) -> SplitTensors:
    """Index one split. `profiles` must already be restricted to this split's rows."""
    if "split" in profiles.columns and not (profiles["split"] == split).all():
        raise ValueError(f"profiles passed for split {split!r} contain rows from other splits")
    if profiles["user_id"].duplicated().any():
        raise ValueError(f"duplicate user rows in {split!r} profiles")

    impressions = impressions.reset_index(drop=True)
    users = impressions["user_id"].astype(str)
    uniq = pd.unique(users)
    user_pos = {u: i for i, u in enumerate(uniq)}
    clicks_of = dict(zip(profiles["user_id"].astype(str), profiles["clicked_ids"]))

    # A user with no profile row (MIND users with an empty history) gets all padding.
    user_history = np.stack([last_n_rows(clicks_of.get(u), row_of, history_len) for u in uniq]) \
        if len(uniq) else np.zeros((0, history_len), dtype=np.int32)

    offsets = np.zeros(len(impressions) + 1, dtype=np.int64)
    cand_rows, labels = [], []
    for i, (inview, clicked) in enumerate(zip(impressions["inview_ids"], impressions["clicked_ids"])):
        clicked = set(to_str_list(clicked))
        kept = [a for a in to_str_list(inview) if a in row_of]
        cand_rows.extend(row_of[a] for a in kept)
        labels.extend(1 if a in clicked else 0 for a in kept)
        offsets[i + 1] = offsets[i] + len(kept)

    return SplitTensors(
        split=split, impressions=impressions,
        user_index=np.fromiter((user_pos[u] for u in users), dtype=np.int32, count=len(users)),
        user_history=user_history, offsets=offsets,
        cand_rows=np.asarray(cand_rows, dtype=np.int32),
        labels=np.asarray(labels, dtype=np.int8),
    )


def load_split_impressions(cfg: Config, split: str, sample: int | None = None,
                           seed: int = 13) -> pd.DataFrame:
    """Same sampling as `evaluate_reranker.load_split`, so a sampled Q3 test set
    is the identical impressions Q2 scored and the two can be compared paired."""
    impressions = read_table(cfg.processed / split / "impressions.parquet", "impressions")
    if sample and sample < len(impressions):
        impressions = impressions.sample(sample, random_state=seed)
    return impressions.reset_index(drop=True)


def load_split_tensors(cfg: Config, split: str, news: NewsTokens, sample: int | None = None,
                       seed: int = 13, history_len: int = HISTORY_LEN) -> SplitTensors:
    impressions = load_split_impressions(cfg, split, sample, seed)
    profiles = pd.read_parquet(cfg.features / "user_profiles.parquet")
    profiles = profiles[profiles["split"] == split]
    return build_split_tensors(split, impressions, profiles, news.row_of, history_len)


# --------------------------------------------------------------------------- #
# training samples: 1 click + K sampled non-clicks
# --------------------------------------------------------------------------- #

@dataclass
class TrainSamples:
    """One row per click: the clicked article first, then K non-clicked from the same impression.

    Negatives are the other articles shown in *that* impression (what the user
    saw and skipped), not random catalogue articles - that is NRMS's training
    signal. An impression with fewer than K non-clicks pads with article 0 and
    marks the slot False in `mask`, so the model can exclude it from the
    softmax instead of treating a blank article as a real negative.
    """
    history: np.ndarray      # int32 (S, history_len)
    candidates: np.ndarray   # int32 (S, 1+K), column 0 is the click
    mask: np.ndarray         # bool  (S, 1+K)
    n_skipped: int           # clicks dropped because their impression had no non-click
    features: np.ndarray | None = None   # float32 (S, 1+K, F), Step 3 signals; None for plain NRMS


def sample_training_rows(t: SplitTensors, npratio: int = NPRATIO,
                         rng: np.random.Generator | None = None) -> TrainSamples:
    """Draw a fresh set of negatives. Call once per epoch with a per-epoch rng."""
    rng = rng if rng is not None else np.random.default_rng(0)
    # Sample flat candidate positions rather than article rows, so per-candidate
    # signals (Step 3) follow the same draw. `rng.choice` over a same-length
    # array consumes the rng identically, so the baseline's negatives are unchanged.
    hist, slots, masks = [], [], []
    skipped = 0
    for i in range(t.n_impressions):
        lo, hi = t.offsets[i], t.offsets[i + 1]
        flat = np.arange(lo, hi)
        labels = t.labels[lo:hi]
        pos, neg = flat[labels == 1], flat[labels == 0]
        if len(pos) == 0:
            continue
        if len(neg) == 0:
            skipped += len(pos)
            continue
        h = t.history_of(i)
        for p in pos:
            if len(neg) >= npratio:
                picked = rng.choice(neg, size=npratio, replace=False)
                m = np.ones(npratio + 1, dtype=bool)
            else:
                picked = np.concatenate([neg, np.full(npratio - len(neg), -1)])
                m = np.concatenate([np.ones(len(neg) + 1, dtype=bool),
                                    np.zeros(npratio - len(neg), dtype=bool)])
            hist.append(h)
            slots.append(np.concatenate([[p], picked]))
            masks.append(m)

    width = npratio + 1
    slots = np.stack(slots).astype(np.int64) if slots else np.zeros((0, width), np.int64)
    padded = slots < 0
    candidates = np.where(padded, PAD, t.cand_rows[np.maximum(slots, 0)]).astype(np.int32)
    features = None
    if t.cand_features is not None:
        features = t.cand_features[np.maximum(slots, 0)]
        features[padded] = 0.0
    return TrainSamples(
        history=np.stack(hist) if hist else np.zeros((0, t.user_history.shape[1]), np.int32),
        candidates=candidates,
        mask=np.stack(masks) if masks else np.zeros((0, width), bool),
        n_skipped=skipped,
        features=features,
    )


# --------------------------------------------------------------------------- #
# stats CLI
# --------------------------------------------------------------------------- #

def describe(cfg: Config, news: NewsTokens, tensors: dict[str, SplitTensors]) -> dict:
    lengths = news.lengths[1:]
    out = {
        "dataset": cfg.dataset, "scale": cfg.scale, "tokenizer": TOKENIZER_NAME,
        "title_len": int(news.tokens.shape[1]), "history_len": HISTORY_LEN, "npratio": NPRATIO,
        "n_articles": int(len(lengths)),
        "vocab_used": int(len(news.vocab) - 1),
        "title_tokens_median": float(np.median(lengths)),
        "titles_truncated_pct": float(100 * (lengths > news.tokens.shape[1]).mean()),
        "empty_titles": int((lengths == 0).sum()),
        "splits": {},
    }
    for split, t in tensors.items():
        hist_len = (t.user_history != PAD).sum(axis=1)
        sizes = np.diff(t.offsets)
        samples = sample_training_rows(t) if split == "train" else None
        out["splits"][split] = {
            "impressions": t.n_impressions,
            "users": int(len(t.user_history)),
            "clicks": int(t.labels.sum()),
            "candidates_per_impression_median": float(np.median(sizes)) if len(sizes) else 0.0,
            "users_empty_history_pct": float(100 * (hist_len == 0).mean()),
            "users_short_history_pct": float(100 * (hist_len < t.user_history.shape[1]).mean()),
            **({"train_rows": int(len(samples.candidates)),
                "rows_with_padded_negatives_pct": float(100 * (~samples.mask).any(axis=1).mean()),
                "clicks_skipped_no_negative": samples.n_skipped} if samples else {}),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rebuild", action="store_true", help="ignore the token cache")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    news = load_news_tokens(cfg, rebuild=args.rebuild)
    tensors = {s: load_split_tensors(cfg, s, news) for s in ("train", "val", "test")}
    stats = describe(cfg, news, tensors)
    print(json.dumps(stats, indent=2))

    out = args.out or REPO_ROOT / "reports" / f"q3_data_{cfg.dataset}.json"
    out.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
