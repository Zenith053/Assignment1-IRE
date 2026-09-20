#!/usr/bin/env python3
"""Click rate of EB-NeRD test candidates by article age (why freshness helps NRMS).

Age = impression time - published time (clamped at 0, as in candidate_signals).
Also scores two freshness-only rules without any model: "newer ranks higher"
(monotone) and "closest to ~4 h old ranks higher" (the peak of the curve).

Writes reports/a2/freshness_curve.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.baseline.candidate_signals import candidate_signals  # noqa: E402
from src.baseline.news_data import load_news_tokens, load_split_tensors  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.eval import metrics as M  # noqa: E402

BINS_H = [0, 0.5, 1, 2, 4, 8, 24, 72, 24 * 30, np.inf]
LABELS = ["<30m", "30m-1h", "1-2h", "2-4h", "4-8h", "8-24h", "1-3d", "3-30d", ">30d"]


def main() -> int:
    cfg = load_config(REPO_ROOT / "config" / "ebnerd.yaml")
    news = load_news_tokens(cfg)
    t = load_split_tensors(cfg, "test", news)
    age = np.expm1(candidate_signals(cfg, t, news.article_ids, None, popularity=False, freshness=True)[:, 0])
    which = np.digitize(age, BINS_H[1:-1], right=False)
    rows = [{"age": LABELS[b], "share_of_candidates": float((which == b).mean()),
             "click_rate": float(t.labels[which == b].mean())} for b in range(len(LABELS))]

    def mean_auc(s):
        v = [M.auc(t.labels[t.offsets[i]:t.offsets[i + 1]], s[t.offsets[i]:t.offsets[i + 1]])
             for i in range(t.n_impressions)]
        return float(np.mean([x for x in v if x is not None]))

    jitter = np.random.default_rng(0).random(len(age)) * 1e-6
    out = {"test_impressions": t.n_impressions, "candidates": int(len(age)), "bins": rows,
           "auc_newer_is_better": mean_auc(-age + jitter),
           "auc_closest_to_4h": mean_auc(-np.abs(np.log1p(age) - np.log1p(4.0)) + jitter)}
    path = REPO_ROOT / "reports" / "a2" / "freshness_curve.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
