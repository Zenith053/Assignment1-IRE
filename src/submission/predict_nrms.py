#!/usr/bin/env python3
"""Codabench submissions from a trained Q3 NRMS checkpoint (MIND and EB-NeRD).

Inference only, straight from the raw files of any split in the official format:

  MIND      <dir>/news.tsv + <dir>/behaviors.tsv       (MINDlarge_test, or MINDsmall_dev to verify)
  EB-NeRD   <dir>/articles.parquet + <dir>/<split>/{behaviors,history}.parquet
            (ebnerd_testset/test, or ebnerd_small/validation to verify)

Steps: tokenise every article title the way training did, extend the trained
vocabulary with pretrained xlm-roberta rows for tokens the model never saw (the
same initialisation training used), encode every article once, stream the
impressions in chunks (EB-NeRD's test set has 13.5M impressions and 206M
candidates), score, convert scores to ranks and write the submission.

Official submission rules this follows (codabench.org competitions 13967 and 2469):
  - zip containing ONLY the text file at its root: MIND `prediction.txt`,
    EB-NeRD `predictions.txt`; no folders, no __MACOSX entries
  - one line per impression: `<impression_id> [r1,r2,...]`, in the original row order
  - ranks are consecutive integers 1..n in the original candidate order, 1 = highest score
    (ties broken by candidate position, as A1's `ranks_from_scores`)

Only checkpoints whose signals exist on an unlabelled test set are accepted:
trailing click counts need click logs the hidden test sets do not contain.

`--evaluate` (labelled splits only) also scores the ranking with the harness metrics,
so a verification run must reproduce the checkpoint's offline test AUC before the
hidden test set is submitted. For MIND it additionally writes the official truth
file for tools/evaluate_official.py.

Usage
-----
    python src/submission/predict_nrms.py --dataset mind \\
        --checkpoint data/feature_store/mind/q3_runs/nrms_seed13.pt \\
        --dir data/raw/mind/MINDsmall_dev --name mind_small_dev --evaluate
    python src/submission/predict_nrms.py --dataset mind \\
        --checkpoint data/feature_store/mind/q3_runs/nrms_seed13.pt \\
        --dir data/raw/mind/MINDlarge_test --name mind_large_test
    python src/submission/predict_nrms.py --dataset ebnerd \\
        --checkpoint data/feature_store/ebnerd/q3_runs/nrms_fresh_submit_seed13.pt \\
        --dir data/raw/ebnerd/ebnerd_testset/ebnerd_testset --split test --name ebnerd_testset
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.baseline import nrms  # noqa: E402
from src.baseline.news_data import HISTORY_LEN, PAD, TITLE_LEN, TOKENIZER_NAME, load_news_tokens  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.data.clean import MIND_BEHAVIOR_COLUMNS, MIND_NEWS_COLUMNS, _parse_mind_impressions  # noqa: E402
from src.eval import metrics as M  # noqa: E402

SUBMISSION_FILE = {"mind": "prediction.txt", "ebnerd": "predictions.txt"}
USER_BATCH = 16_384
CANDIDATE_BATCH = 200_000
# Chunks are capped by candidates as well as impressions: EB-NeRD's last row group holds
# 200k beyond-accuracy impressions with ~250 candidates each (50.8M candidates), and
# ~110 bytes of per-candidate arrays per chunk otherwise pushed a 16 GB Mac into swap.
MAX_CANDIDATES_PER_CHUNK = 3_000_000


# --------------------------------------------------------------------------- #
# model with a vocabulary extended to the new articles
# --------------------------------------------------------------------------- #

def build_model(dataset: str, checkpoint: Path, article_ids: list[str], titles: list[str],
                device: torch.device) -> tuple[nrms.NRMS, dict, dict]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta, state = ckpt["meta"], ckpt["state_dict"]
    variant = meta.get("variant") or {}
    if variant.get("popularity"):
        raise SystemExit("this checkpoint uses trailing click counts, which unlabelled test sets cannot provide")

    cfg = load_config(REPO_ROOT / "config" / f"{dataset}.yaml")
    train_vocab = load_news_tokens(cfg).vocab                         # compact id -> tokenizer id
    trained_emb = state["news_encoder.embedding.weight"].numpy()
    if len(train_vocab) != trained_emb.shape[0]:
        raise SystemExit("checkpoint vocabulary does not match the dataset's token cache")
    compact_of = {int(t): i for i, t in enumerate(train_vocab) if i != PAD}

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    encoded = tokenizer(list(titles), add_special_tokens=False)["input_ids"]

    new_tokens: list[int] = []
    tokens = np.zeros((len(encoded) + 1, TITLE_LEN), dtype=np.int32)
    for row, ids in enumerate(encoded, start=1):
        ids = ids[:TITLE_LEN]
        out = []
        for t in ids:
            c = compact_of.get(t)
            if c is None:
                c = len(train_vocab) + len(new_tokens)
                compact_of[t] = c
                new_tokens.append(t)
            out.append(c)
        tokens[row, :len(out)] = out

    emb = trained_emb
    if new_tokens:
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        with safe_open(hf_hub_download(TOKENIZER_NAME, "model.safetensors"), framework="np") as f:
            key = next(k for k in f.keys() if k.endswith("embeddings.word_embeddings.weight"))
            pretrained = f.get_tensor(key)
        emb = np.concatenate([trained_emb, pretrained[np.asarray(new_tokens)].astype(np.float32)])
        del pretrained

    new_set = set(new_tokens)
    n_signals = len(variant.get("signals") or [])
    model = nrms.NRMS(tokens, emb, n_signals=n_signals, gate=variant.get("gate") or "learned")
    rest = {k: v for k, v in state.items() if k != "news_encoder.embedding.weight"}
    missing, unexpected = model.load_state_dict(rest, strict=False)
    if unexpected or [k for k in missing if k != "news_encoder.embedding.weight"]:
        raise SystemExit(f"checkpoint does not fit the model: missing={missing} unexpected={unexpected}")
    model = model.to(device).eval()
    info = {"checkpoint": str(checkpoint), "run": meta.get("run"), "variant": variant,
            "articles": len(article_ids), "tokens_trained": int(len(train_vocab) - 1),
            "tokens_new": len(new_tokens),
            "titles_with_new_tokens_pct": round(100 * float(np.mean(
                [any(t in new_set for t in ids[:TITLE_LEN]) for ids in encoded])), 2)}
    return model, info, variant


# --------------------------------------------------------------------------- #
# scoring and writing
# --------------------------------------------------------------------------- #

@torch.no_grad()
def score_chunk(model, news_vecs, hist: np.ndarray, cand_rows: np.ndarray, offsets: np.ndarray,
                feats: np.ndarray | None, device) -> np.ndarray:
    """Flat scores for one chunk: hist (n_imp, 20), cand_rows (C,), offsets (n_imp+1,)."""
    h = torch.from_numpy(hist.astype(np.int64)).to(device)
    users = torch.cat([model.user_encoder(news_vecs[h[i:i + USER_BATCH]], h[i:i + USER_BATCH] != PAD)
                       for i in range(0, len(h), USER_BATCH)])
    imp_of = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    out = np.empty(len(cand_rows), dtype=np.float32)
    for i in range(0, len(cand_rows), CANDIDATE_BATCH):
        c = torch.from_numpy(cand_rows[i:i + CANDIDATE_BATCH].astype(np.int64)).to(device)
        u = users[torch.from_numpy(imp_of[i:i + CANDIDATE_BATCH]).to(device)]
        content = (news_vecs[c] * u).sum(-1)
        f = torch.from_numpy(feats[i:i + CANDIDATE_BATCH]).to(device)[:, None] if feats is not None else None
        out[i:i + len(c)] = model.combine(content[:, None], u, f)[:, 0].float().cpu().numpy()
    return out


def ranks_by_impression(scores: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """1-based ranks per impression; ties keep candidate order (== A1's ranks_from_scores)."""
    lengths = np.diff(offsets)
    imp_of = np.repeat(np.arange(len(lengths)), lengths)
    pos = np.arange(len(scores)) - np.repeat(offsets[:-1], lengths)
    order = np.lexsort((pos, -scores.astype(np.float64), imp_of))
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(len(scores)) - np.repeat(offsets[:-1], lengths) + 1
    # Every impression's ranks must be exactly 1..n (a strict permutation).
    if len(scores) and ((ranks < 1) | (ranks > np.repeat(lengths, lengths))).any():
        raise AssertionError("rank out of range")
    check = np.zeros(len(scores), dtype=bool)
    check[offsets[:-1][imp_of] + ranks - 1] = True
    if not check.all():
        raise AssertionError("ranks are not a permutation within an impression")
    return ranks


RANK_STR = [str(i) for i in range(1024)]


def write_lines(fh, impression_ids, ranks: np.ndarray, offsets: np.ndarray) -> None:
    lines = []
    rk = ranks.tolist()
    for i, imp in enumerate(impression_ids):
        lo, hi = offsets[i], offsets[i + 1]
        lines.append(f"{imp} [{','.join(RANK_STR[r] if r < 1024 else str(r) for r in rk[lo:hi])}]\n")
    fh.write("".join(lines))


# --------------------------------------------------------------------------- #
# dataset readers: articles, then chunks of (ids, history rows, candidates, labels, times)
# --------------------------------------------------------------------------- #

def mind_articles(d: Path):
    news = pd.read_csv(d / "news.tsv", sep="\t", header=None, names=MIND_NEWS_COLUMNS, dtype=str,
                       quoting=csv.QUOTE_NONE, keep_default_na=False)
    news = news.drop_duplicates(subset="article_id", keep="first")
    return news["article_id"].astype(str).tolist(), news["title"].astype(str).tolist(), None


def mind_chunks(d: Path, row_of: dict, chunk: int):
    reader = pd.read_csv(d / "behaviors.tsv", sep="\t", header=None, names=MIND_BEHAVIOR_COLUMNS, dtype=str,
                         quoting=csv.QUOTE_NONE, keep_default_na=False, chunksize=chunk)
    for block in reader:
        n = len(block)
        hist = np.zeros((n, HISTORY_LEN), dtype=np.int32)
        offsets = np.zeros(n + 1, dtype=np.int64)
        cands, labels, has_labels = [], [], False
        for i, (h, imp) in enumerate(zip(block["history"], block["impressions"])):
            rows = [row_of[a] for a in (h.split() if h else []) if a in row_of][-HISTORY_LEN:]
            if rows:
                hist[i, HISTORY_LEN - len(rows):] = rows
            inview, clicked = _parse_mind_impressions(imp)
            has_labels = has_labels or ("-" in imp)
            clicked = set(clicked)
            cands.extend(row_of.get(a, PAD) for a in inview)
            labels.extend(1 if a in clicked else 0 for a in inview)
            offsets[i + 1] = offsets[i] + len(inview)
        yield (block["impression_id"].tolist(), hist, np.asarray(cands, dtype=np.int64), offsets,
               np.asarray(labels, dtype=np.int8) if has_labels else None, None)


def eb_articles(d: Path):
    a = pq.read_table(d / "articles.parquet", columns=["article_id", "title", "published_time"]).to_pandas()
    ids = a["article_id"].astype(str).tolist()
    pub = pd.to_datetime(a["published_time"]).to_numpy().astype("datetime64[us]").astype("float64") / 3.6e9
    pub[a["published_time"].isna().to_numpy()] = np.nan
    return ids, a["title"].fillna("").astype(str).tolist(), pub


def eb_histories(d: Path, split: str, row_of_int: np.ndarray) -> tuple[dict, np.ndarray]:
    f = pq.ParquetFile(d / split / "history.parquet")
    hist = np.zeros((f.metadata.num_rows, HISTORY_LEN), dtype=np.int32)
    slot_of, k = {}, 0
    for batch in f.iter_batches(batch_size=100_000, columns=["user_id", "article_id_fixed"]):
        users = batch.column("user_id").to_numpy()
        col = batch.column("article_id_fixed")
        flat = pc.list_flatten(col).to_numpy()
        offs = col.offsets.to_numpy()
        rows = lookup(row_of_int, flat)
        for i, u in enumerate(users):
            r = rows[offs[i]:offs[i + 1]]
            r = r[r != PAD][-HISTORY_LEN:]
            if len(r):
                hist[k, HISTORY_LEN - len(r):] = r
            slot_of[int(u)] = k
            k += 1
    return slot_of, hist


def lookup(row_of_int: np.ndarray, ids: np.ndarray) -> np.ndarray:
    ids = ids.astype(np.int64)
    out = np.zeros(len(ids), dtype=np.int64)
    ok = (ids >= 0) & (ids < len(row_of_int))
    out[ok] = row_of_int[ids[ok]]
    return out


def eb_chunks(d: Path, split: str, row_of_int: np.ndarray, slot_of: dict, user_hist: np.ndarray, chunk: int):
    f = pq.ParquetFile(d / split / "behaviors.parquet")
    names = f.schema_arrow.names
    cols = ["impression_id", "impression_time", "article_ids_inview", "user_id"]
    labelled = "article_ids_clicked" in names
    for batch in f.iter_batches(batch_size=chunk, columns=cols + (["article_ids_clicked"] if labelled else [])):
        inview = batch.column("article_ids_inview")
        offsets = inview.offsets.to_numpy().astype(np.int64)
        offsets = offsets - offsets[0]
        flat_ids = pc.list_flatten(inview).to_numpy()
        cands = lookup(row_of_int, flat_ids)
        users = batch.column("user_id").to_numpy()
        slots = np.fromiter((slot_of.get(int(u), -1) for u in users), dtype=np.int64, count=len(users))
        hist = np.zeros((len(users), HISTORY_LEN), dtype=np.int32)
        hist[slots >= 0] = user_hist[slots[slots >= 0]]
        labels = None
        if labelled:
            clicked = batch.column("article_ids_clicked").to_pylist()
            labels = np.zeros(len(flat_ids), dtype=np.int8)
            for i, cl in enumerate(clicked):
                if cl:
                    lo, hi = offsets[i], offsets[i + 1]
                    labels[lo:hi] = np.isin(flat_ids[lo:hi], cl)
        t_hours = batch.column("impression_time").to_numpy().astype("datetime64[us]").astype("float64") / 3.6e9
        yield (batch.column("impression_id").to_numpy().tolist(), hist, cands, offsets, labels,
               np.repeat(t_hours, np.diff(offsets)))


def split_by_candidates(chunks, budget: int = MAX_CANDIDATES_PER_CHUNK):
    """Re-cut (ids, hist, cands, offsets, labels, t_cand) chunks so none exceeds `budget` candidates."""
    for imp_ids, hist, cands, offsets, labels, t_cand in chunks:
        n = len(offsets) - 1
        start = 0
        while start < n:
            # largest end with offsets[end] - offsets[start] <= budget (at least one impression)
            end = int(np.searchsorted(offsets, offsets[start] + budget, side="right")) - 1
            end = min(max(end, start + 1), n)
            lo, hi = offsets[start], offsets[end]
            yield (imp_ids[start:end], hist[start:end], cands[lo:hi], offsets[start:end + 1] - lo,
                   None if labels is None else labels[lo:hi], None if t_cand is None else t_cand[lo:hi])
            start = end


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dir", type=Path, required=True, help="raw split directory (MIND) or bundle root (EB-NeRD)")
    parser.add_argument("--split", default="test", help="EB-NeRD sub-directory: test | validation")
    parser.add_argument("--name", required=True)
    parser.add_argument("--chunk", type=int, default=200_000, help="impressions per chunk")
    parser.add_argument("--evaluate", action="store_true", help="labelled splits: report harness metrics")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports" / "submissions")
    args = parser.parse_args(argv)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    torch.set_grad_enabled(False)

    ids, titles, pub_hours = (mind_articles if args.dataset == "mind" else eb_articles)(args.dir)
    model, info, variant = build_model(args.dataset, args.checkpoint, ids, titles, device)
    news_vecs = nrms.encode_all_news(model, device)
    freshness = bool(variant.get("freshness"))
    if freshness and pub_hours is None:
        raise SystemExit("freshness checkpoint needs published times, which this dataset lacks")
    print(f"[{args.name}] {info['articles']:,} articles encoded on {device}; {info['tokens_new']:,} tokens new "
          f"to the model ({info['titles_with_new_tokens_pct']}% of titles); signals={variant.get('signals')} "
          f"[{time.time() - t0:.0f}s]", flush=True)

    if args.dataset == "mind":
        row_of = {a: i + 1 for i, a in enumerate(ids)}
        chunks = mind_chunks(args.dir, row_of, args.chunk)
    else:
        int_ids = np.asarray([int(a) for a in ids], dtype=np.int64)
        row_of_int = np.zeros(int_ids.max() + 1, dtype=np.int64)
        row_of_int[int_ids] = np.arange(1, len(int_ids) + 1)
        slot_of, user_hist = eb_histories(args.dir, args.split, row_of_int)
        print(f"  {len(slot_of):,} user histories [{time.time() - t0:.0f}s]", flush=True)
        chunks = eb_chunks(args.dir, args.split, row_of_int, slot_of, user_hist, args.chunk)
    chunks = split_by_candidates(chunks)
    pub_full = None if pub_hours is None else np.concatenate([[np.nan], pub_hours])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    txt = args.out_dir / f"{SUBMISSION_FILE[args.dataset].split('.')[0]}_{args.name}.txt"
    truth = args.out_dir / f"truth_{args.name}.txt" if (args.evaluate and args.dataset == "mind") else None
    per_imp = {m: [] for m in ("auc", "mrr", "ndcg@5", "ndcg@10")}
    n_imp = n_cand = n_unknown = n_no_history = 0
    with open(txt, "w", encoding="utf-8") as fh, (open(truth, "w") if truth else open("/dev/null", "w")) as th:
        for imp_ids, hist, cands, offsets, labels, t_cand in chunks:
            feats = None
            if freshness:
                age = t_cand - pub_full[cands]
                feats = np.log1p(np.where(np.isnan(age), 0.0, np.maximum(age, 0.0))).astype(np.float32)[:, None]
            scores = score_chunk(model, news_vecs, hist, cands, offsets, feats, device)
            ranks = ranks_by_impression(scores, offsets)
            write_lines(fh, imp_ids, ranks, offsets)
            if args.evaluate and labels is not None:
                for i in range(len(imp_ids)):
                    lo, hi = offsets[i], offsets[i + 1]
                    lab, sc = labels[lo:hi], scores[lo:hi]
                    a = M.auc(lab, sc)
                    per_imp["auc"].append(a)
                    per_imp["mrr"].append(M.mrr(lab, sc))
                    per_imp["ndcg@5"].append(M.ndcg(lab, sc, 5))
                    per_imp["ndcg@10"].append(M.ndcg(lab, sc, 10))
                    if truth:
                        th.write(f"{imp_ids[i]} [{','.join(map(str, lab.tolist()))}]\n")
            del scores, ranks, feats
            if device.type == "mps":
                torch.mps.empty_cache()          # the MPS allocator otherwise keeps every chunk's buffers
            n_imp += len(imp_ids)
            n_cand += len(cands)
            n_unknown += int((cands == PAD).sum())
            n_no_history += int((hist == PAD).all(axis=1).sum())
            el = time.time() - t0
            print(f"\r  {n_imp:,} impressions, {n_cand:,} candidates [{el / 60:.1f} min]   ", end="", flush=True)
    print()

    archive = args.out_dir / f"submission_{args.name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(txt, arcname=SUBMISSION_FILE[args.dataset])
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
    if names != [SUBMISSION_FILE[args.dataset]]:
        raise SystemExit(f"zip must contain only {SUBMISSION_FILE[args.dataset]}, found {names}")

    result = {**info, "name": args.name, "dataset": args.dataset, "source": str(args.dir), "split": args.split,
              "impressions": n_imp, "candidates": n_cand, "candidates_unknown_article": n_unknown,
              "impressions_without_history": n_no_history, "device": str(device),
              "minutes": round((time.time() - t0) / 60, 1),
              "txt_mb": round(txt.stat().st_size / 1e6, 1), "zip_mb": round(archive.stat().st_size / 1e6, 1),
              "zip_contents": names}
    if per_imp["auc"]:
        result["metrics"] = {m: float(np.mean([v for v in vals if v is not None])) for m, vals in per_imp.items()}
    (args.out_dir / f"submission_{args.name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
