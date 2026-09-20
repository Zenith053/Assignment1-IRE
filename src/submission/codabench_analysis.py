#!/usr/bin/env python3
"""Which model can go to Codabench, measured on our labelled test splits.

The hidden Codabench test sets (MINDlarge_test, ebnerd_testset) carry no clicks,
so any feature built from clicks during the test period is unavailable there.
This script measures, on our own labelled test split, what that does to each
candidate submission, and records the verification runs of the submitted models.

  1. popularity collapse (EB-NeRD): the best Q3 model (nrms_popfresh, seed 13)
     scored with (a) real trailing click counts, (b) counts from clicks before the
     test period only - the best a hidden test set allows - and (c) no clicks
  2. click-free blends: per-impression rank averages of click-free scorers
     (MIND: NRMS + A1 semantic; EB-NeRD: nrms_fresh + exposure counts, i.e. how
     often an article was shown in the trailing 24 h). Exploratory: weights are
     read off the test split, so these are upper bounds, not claims.

Writes reports/a2/codabench_analysis.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.baseline import candidate_signals as sig  # noqa: E402
from src.baseline import nrms  # noqa: E402
from src.baseline.news_data import load_news_tokens, load_split_tensors  # noqa: E402
from src.baseline.train_nrms import mean_auc  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.common.io import read_table  # noqa: E402
from src.eval import metrics as M  # noqa: E402

RUNS = "q3_runs"


def auc_of(off, lab, s) -> float:
    v = [M.auc(lab[off[i]:off[i + 1]], s[off[i]:off[i + 1]]) for i in range(len(off) - 1)]
    return float(np.mean([x for x in v if x is not None]))


def rank01(off, s) -> np.ndarray:
    lengths = np.diff(off)
    imp = np.repeat(np.arange(len(lengths)), lengths)
    pos = np.arange(len(s)) - np.repeat(off[:-1], lengths)
    order = np.lexsort((pos, -s.astype(np.float64), imp))
    r = np.empty(len(s))
    r[order] = pos + 1
    return 1.0 - (r - 1) / np.maximum(np.repeat(lengths, lengths) - 1, 1)


def popularity_collapse(dev) -> dict:
    cfg = load_config(REPO_ROOT / "config" / "ebnerd.yaml")
    news = load_news_tokens(cfg)
    test = load_split_tensors(cfg, "test", news)
    ck = torch.load(cfg.features / RUNS / "nrms_popfresh_seed13.pt", map_location="cpu", weights_only=False)
    model = nrms.NRMS(news.tokens, nrms.load_word_embeddings(cfg, news), n_signals=4).to(dev)
    model.load_state_dict(ck["state_dict"])
    fresh = sig.candidate_signals(cfg, test, news.article_ids, None, popularity=False, freshness=True)
    before = {s: read_table(cfg.processed / s / "impressions.parquet", "impressions") for s in ("train", "val")}
    out = {}
    for label, timeline in (("real_clicks_up_to_impression", sig.build_timeline(cfg, news.row_of)),
                            ("clicks_before_test_period_only", sig.ClickTimeline.from_impressions(before, news.row_of)),
                            ("no_clicks", None)):
        pop = (np.zeros((len(test.cand_rows), 3), np.float32) if timeline is None else
               sig.candidate_signals(cfg, test, news.article_ids, timeline, popularity=True, freshness=False))
        test.cand_features = np.concatenate([pop, fresh], axis=1)
        out[label] = {"auc": mean_auc(test, nrms.score_split(model, test, dev)),
                      "candidates_with_24h_clicks_pct": float(100 * (pop[:, 1] > 0).mean())}
        print(f"  popfresh with {label}: AUC {out[label]['auc']:.4f}", flush=True)
    return {"model": "nrms_popfresh seed 13", "test_impressions": test.n_impressions, **out}


def blends() -> dict:
    out = {}
    # MIND: NRMS (seed 13) + A1 semantic
    cfg = load_config(REPO_ROOT / "config" / "mind.yaml")
    z = np.load(cfg.features / RUNS / "nrms_seed13_test_scores.npz")
    a1 = np.load(cfg.features / RUNS / "a1_scorers_test_scores.npz")
    off, lab = z["offsets"], z["labels"]
    rn, rs = rank01(off, z["scores"]), rank01(off, a1["semantic"])
    out["mind"] = {"nrms": auc_of(off, lab, z["scores"]), "a1_semantic": auc_of(off, lab, a1["semantic"]),
                   **{f"rank_avg_{w:.1f}nrms_{1 - w:.1f}semantic": auc_of(off, lab, w * rn + (1 - w) * rs)
                      for w in (0.3, 0.5, 0.7)}}
    # EB-NeRD: nrms_fresh (submitted model) + exposure counts
    cfg = load_config(REPO_ROOT / "config" / "ebnerd.yaml")
    news = load_news_tokens(cfg)
    z = np.load(cfg.features / RUNS / "nrms_fresh_submit_seed13_test_scores.npz")
    t = load_split_tensors(cfg, "test", news)
    off, lab = z["offsets"], z["labels"]
    imp_of = np.repeat(np.arange(len(off) - 1), np.diff(off))
    ts = t.impressions["timestamp"].to_numpy().astype("datetime64[us]")[imp_of]
    row_of = news.row_of                     # property: builds a dict on each access, so read once
    rows, stamps = [], []
    for s in ("train", "val", "test"):
        imps = read_table(cfg.processed / s / "impressions.parquet", "impressions")
        for tt, inview in zip(imps["timestamp"].to_numpy(), imps["inview_ids"]):
            for a in inview:
                if a in row_of:
                    rows.append(row_of[a])
                    stamps.append(tt)
    exposure = sig.ClickTimeline(np.asarray(rows, np.int64), np.asarray(stamps, "datetime64[us]"))
    exp24 = np.log1p(exposure.counts_before(t.cand_rows, ts, 24.0)).astype(np.float32)
    rf, re_ = rank01(off, z["scores"]), rank01(off, exp24)
    out["ebnerd"] = {"nrms_fresh": auc_of(off, lab, z["scores"]), "exposure_24h": auc_of(off, lab, exp24),
                     **{f"rank_avg_{w:.1f}fresh_{1 - w:.1f}exposure": auc_of(off, lab, w * rf + (1 - w) * re_)
                        for w in (0.3, 0.5, 0.7)}}
    print(f"  blends: {json.dumps(out, indent=1)}", flush=True)
    return out


def main() -> int:
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    res = {"popularity_collapse_ebnerd": popularity_collapse(dev), "click_free_blends": blends(),
           "note": "blend weights read off the test split: exploratory upper bounds, not submitted"}
    sub = REPO_ROOT / "reports" / "submissions"
    res["submissions"] = {p.stem: json.loads(p.read_text()) for p in sorted(sub.glob("submission_*.json"))}
    out = REPO_ROOT / "reports" / "a2" / "codabench_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
