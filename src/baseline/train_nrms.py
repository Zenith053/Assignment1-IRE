#!/usr/bin/env python3
"""Q3: train NRMS on train, early-stop on val AUC, report on test.

Each epoch:
  1. draw fresh negatives (1 click + NPRATIO non-clicks from the same impression)
  2. for each batch: score the 1+K candidates, softmax, cross-entropy against
     position 0 (the click), one Adam step
  3. score the validation impressions; keep the weights with the best val AUC

After training, the best weights score the test split. Per-impression test
scores are saved so later variants (Step 3's improvement and ablations) can be
compared against this run with a paired bootstrap on identical impressions.

Usage
-----
    python src/baseline/train_nrms.py --config config/mind.yaml
    python src/baseline/train_nrms.py --config config/ebnerd.yaml --seed 14
    python src/baseline/train_nrms.py --config config/mind.yaml --max-steps 200 --val-sample 2000 --test-sample 2000   # smoke run

Q3 improvement and its ablation arms (tag defaults to the variant name):
    --popularity                       nrms_pop            trailing click counts
    --freshness                        nrms_fresh          article age (EB-NeRD)
    --popularity --freshness           nrms_popfresh       both, per-user gate
    --popularity --freshness --gate sum  nrms_popfresh_sum  both, no gate
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.baseline import candidate_signals as sig  # noqa: E402
from src.baseline import nrms  # noqa: E402
from src.baseline.news_data import (  # noqa: E402
    HISTORY_LEN, NPRATIO, TITLE_LEN, TOKENIZER_NAME, load_news_tokens, load_split_tensors,
    sample_training_rows,
)
from src.common.config import load_config  # noqa: E402
from src.eval import metrics as M  # noqa: E402

BATCH_SIZE = 32
LEARNING_RATE = 1e-4


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def per_impression(tensors, scores: np.ndarray) -> dict[str, list]:
    out = {"auc": [], "mrr": [], "ndcg@5": [], "ndcg@10": []}
    for i in range(tensors.n_impressions):
        lo, hi = tensors.offsets[i], tensors.offsets[i + 1]
        labels, s = tensors.labels[lo:hi], scores[lo:hi]
        out["auc"].append(M.auc(labels, s))
        out["mrr"].append(M.mrr(labels, s))
        out["ndcg@5"].append(M.ndcg(labels, s, 5))
        out["ndcg@10"].append(M.ndcg(labels, s, 10))
    return out


def mean_auc(tensors, scores: np.ndarray) -> float:
    vals = [v for v in per_impression(tensors, scores)["auc"] if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def train_one_epoch(model, samples, optimizer, device, batch_size: int, rng,
                    max_steps: int | None, log_every: int = 500) -> dict:
    model.train()
    order = rng.permutation(len(samples.candidates))
    n_batches = int(np.ceil(len(order) / batch_size))
    if max_steps:
        n_batches = min(n_batches, max_steps)

    total, seen, t0 = 0.0, 0, time.time()
    for step in range(n_batches):
        idx = order[step * batch_size:(step + 1) * batch_size]
        hist = torch.from_numpy(samples.history[idx].astype(np.int64)).to(device)
        cand = torch.from_numpy(samples.candidates[idx].astype(np.int64)).to(device)
        mask = torch.from_numpy(samples.mask[idx]).to(device)
        feats = (torch.from_numpy(samples.features[idx]).to(device)
                 if samples.features is not None else None)

        logits = model(hist, cand, feats).masked_fill(~mask, -1e4)   # padded negatives can't win
        target = torch.zeros(len(idx), dtype=torch.long, device=device)  # the click is column 0
        loss = F.cross_entropy(logits, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total += loss.item() * len(idx)
        seen += len(idx)
        if (step + 1) % log_every == 0 or step + 1 == n_batches:
            print(f"    step {step + 1:>6}/{n_batches}  loss {total / seen:.4f}  "
                  f"{(step + 1) / (time.time() - t0):.0f} batches/s", flush=True)
    return {"loss": total / max(seen, 1), "steps": n_batches, "seconds": time.time() - t0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=1,
                        help="stop after this many epochs without a val AUC improvement")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--train-sample", type=int, default=None)
    parser.add_argument("--val-sample", type=int, default=20000,
                        help="val impressions scored after each epoch (for early stopping)")
    parser.add_argument("--test-sample", type=int, default=None,
                        help="default: the full test split")
    parser.add_argument("--max-steps", type=int, default=None, help="cap batches per epoch (smoke runs)")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--popularity", action="store_true", help="Q3: add trailing click counts")
    parser.add_argument("--freshness", action="store_true", help="Q3: add article age (EB-NeRD)")
    parser.add_argument("--gate", choices=nrms.GATES, default="learned",
                        help="how the signal score is blended with the content score")
    parser.add_argument("--tag", default=None, help="run name prefix; default derives from the variant")
    args = parser.parse_args(argv)
    if args.gate != "learned" and not (args.popularity or args.freshness):
        parser.error("--gate only applies with --popularity and/or --freshness")
    if args.tag is None:
        args.tag = "nrms" + ("_" if args.popularity or args.freshness else "") \
            + ("pop" if args.popularity else "") + ("fresh" if args.freshness else "") \
            + ("_sum" if args.gate == "sum" else "")

    cfg = load_config(args.config)
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    run_name = f"{args.tag}_seed{args.seed}"
    print(f"[{cfg.dataset}] {run_name} on {device}")

    # ---- data ------------------------------------------------------------
    t0 = time.time()
    news = load_news_tokens(cfg)
    split = {
        "train": load_split_tensors(cfg, "train", news, args.train_sample, seed=args.seed),
        "val": load_split_tensors(cfg, "val", news, args.val_sample),
        "test": load_split_tensors(cfg, "test", news, args.test_sample),
    }
    embeddings = nrms.load_word_embeddings(cfg, news)
    names = sig.signal_names(args.popularity, args.freshness)
    if names:
        timeline = sig.build_timeline(cfg, news.row_of) if args.popularity else None
        for t in split.values():
            t.cand_features = sig.candidate_signals(cfg, t, news.article_ids, timeline,
                                                    args.popularity, args.freshness)
        print(f"  candidate signals: {names}")
    print(f"  data ready in {time.time() - t0:.1f}s: " + ", ".join(
        f"{s}={t.n_impressions:,} impressions" for s, t in split.items())
        + f"; vocab {embeddings.shape[0]:,} x {embeddings.shape[1]}")

    # ---- model -----------------------------------------------------------
    model = nrms.NRMS(news.tokens, embeddings, n_signals=len(names), gate=args.gate).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_embed = model.news_encoder.embedding.weight.numel()
    print(f"  parameters: {n_params:,} ({n_embed:,} word embeddings + {n_params - n_embed:,} attention)")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---- train with early stopping ----------------------------------------
    history, best_auc, best_state, bad_epochs = [], -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        rng = np.random.default_rng((args.seed, epoch))
        samples = sample_training_rows(split["train"], NPRATIO, rng)
        print(f"\n  epoch {epoch}: {len(samples.candidates):,} training rows "
              f"({samples.n_skipped} clicks skipped: no non-click to contrast)")
        stats = train_one_epoch(model, samples, optimizer, device, args.batch_size, rng, args.max_steps)

        val_auc = mean_auc(split["val"], nrms.score_split(model, split["val"], device))
        improved = val_auc > best_auc
        print(f"  epoch {epoch}: train loss {stats['loss']:.4f}  val AUC {val_auc:.4f}"
              f"{'  (best so far)' if improved else ''}  [{stats['seconds']:.0f}s]")
        history.append({"epoch": epoch, **stats, "val_auc": val_auc})

        if improved:
            best_auc, best_state, bad_epochs = val_auc, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"  early stop: no val improvement for {bad_epochs} epoch(s)")
                break

    model.load_state_dict(best_state)
    best_epoch = max(history, key=lambda h: h["val_auc"])["epoch"]

    # ---- test -------------------------------------------------------------
    t0 = time.time()
    test = split["test"]
    scores = nrms.score_split(model, test, device)
    raw = per_impression(test, scores)
    summary = {}
    for metric, values in raw.items():
        point, lo, hi = M.bootstrap_ci(values, n_boot=args.n_boot)
        summary[metric] = {"value": point, "ci_low": lo, "ci_high": hi,
                           "n": sum(v is not None for v in values)}
    print(f"\n  test ({test.n_impressions:,} impressions, best epoch {best_epoch}, "
          f"scored in {time.time() - t0:.0f}s)")
    for metric, s in summary.items():
        print(f"    {metric:<8} {s['value']:.4f}  [{s['ci_low']:.4f}, {s['ci_high']:.4f}]")

    # ---- artifacts --------------------------------------------------------
    out_dir = cfg.features / "q3_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{run_name}_test_scores.npz",
             impression_id=test.impressions["impression_id"].to_numpy(),
             offsets=test.offsets, cand_rows=test.cand_rows, labels=test.labels, scores=scores)
    meta = {
        "dataset": cfg.dataset, "scale": cfg.scale, "run": run_name, "seed": args.seed,
        "device": str(device),
        "hparams": {"tokenizer": TOKENIZER_NAME, "title_len": TITLE_LEN, "history_len": HISTORY_LEN,
                    "npratio": NPRATIO, "n_heads": nrms.N_HEADS, "head_dim": nrms.HEAD_DIM,
                    "attention_hidden": nrms.ATTENTION_HIDDEN, "dropout": nrms.DROPOUT,
                    "batch_size": args.batch_size, "lr": args.lr, "epochs_max": args.epochs,
                    "patience": args.patience, "max_steps": args.max_steps},
        "variant": {"popularity": args.popularity, "freshness": args.freshness,
                    "gate": args.gate if names else None, "signals": names,
                    "popularity_windows_hours": list(sig.POPULARITY_WINDOWS_HOURS) if args.popularity else None},
        "n_impressions": {s: t.n_impressions for s, t in split.items()},
        "n_parameters": n_params,
        "epochs": history, "best_epoch": best_epoch, "best_val_auc": best_auc,
        "test": summary,
    }
    nrms.save(model, out_dir / f"{run_name}.pt", meta)
    report = REPO_ROOT / "reports" / "q3" / f"{cfg.dataset}_{run_name}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"  -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
