#!/usr/bin/env python3
"""Q4 phase 3 (Q4.2): p50/p95/p99 latency of one request through the two-stage pipeline.

Protocol (settings in config/serving.yaml):
  - the pipeline loads once; start-up is reported separately, never in latency
  - requests replay real (user_id, timestamp) pairs from the test split, sampled
    with a fixed seed; the same requests are used for every mode (paired)
  - `warmup_requests` different requests run first and are discarded
  - both stage-1 modes (as_is, serving) answer each request; the order alternates
    request by request so neither mode systematically runs on warmer caches
  - every library pinned to `threads` (1); single request at a time
  - latency = wall time around `handle()`, plus each stage's own timer

Reported per mode: mean/p50/p95/p99/max for the total and every stage; p99 by
history length (BM25 queries are built from the whole click history); the
slowest requests; machine state (power source, load average).

Writes reports/q4_latency_<dataset>.json.

Usage
-----
    python src/serving/benchmark_latency.py --config config/mind.yaml
    python src/serving/benchmark_latency.py --config config/ebnerd.yaml --n 500   # quicker
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.serving import load_serving_config  # noqa: E402  (sets BLAS thread env first)

import numpy as np  # noqa: E402

from src.common.config import load_config  # noqa: E402
from src.serving.pipeline import MODES, STAGES, ServingPipeline  # noqa: E402

HISTORY_BINS = [(0, 0), (1, 10), (11, 50), (51, 200), (201, 10**9)]


def summarise(values, percentiles) -> dict:
    v = np.asarray(values, dtype=np.float64)
    out = {"mean": float(v.mean())}
    out.update({f"p{p}": float(np.percentile(v, p)) for p in percentiles})
    out["max"] = float(v.max())
    return {k: round(x, 4) for k, x in out.items()}


def machine_state() -> dict:
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
        except OSError:
            return None
    batt = run(["pmset", "-g", "batt"]) or ""
    return {"load_average_1_5_15": [round(x, 2) for x in os.getloadavg()],
            "power": "AC" if "AC Power" in batt else ("battery" if "Battery Power" in batt else "unknown")}


def run_benchmark(dataset: str, n: int | None = None) -> dict:
    conf = load_serving_config()
    lat = conf["latency"]
    n = n or lat["n_requests"]
    pcts = lat["percentiles"]

    t0 = time.perf_counter()
    pipe = ServingPipeline(dataset, mode="serving").load()
    load_s = time.perf_counter() - t0
    state_before = machine_state()

    imps = pipe.impressions
    rng = np.random.default_rng(lat["seed"])
    order = rng.permutation(len(imps))
    warm_idx, req_idx = order[: lat["warmup_requests"]], order[lat["warmup_requests"]: lat["warmup_requests"] + n]
    users = imps["user_id"].to_numpy()
    stamps = imps["timestamp"].to_numpy()

    for i in warm_idx:                                  # discarded
        for mode in MODES:
            pipe.mode = mode
            pipe.handle(users[i], stamps[i])

    records = {m: [] for m in MODES}
    t_run = time.perf_counter()
    for k, i in enumerate(req_idx):
        modes = MODES if k % 2 == 0 else MODES[::-1]
        for mode in modes:
            pipe.mode = mode
            start = time.perf_counter()
            resp = pipe.handle(users[i], stamps[i])
            wall = 1e3 * (time.perf_counter() - start)
            records[mode].append({"wall": wall, **resp.timings_ms,
                                  "n_candidates": len(resp.candidate_ids),
                                  "history_len": len(pipe.clicks_of.get(users[i], [])),
                                  "impression_row": int(i)})
    run_s = time.perf_counter() - t_run
    pipe.mode = "serving"

    results = {}
    for mode, recs in records.items():
        wall = [r["wall"] for r in recs]
        hist = np.array([r["history_len"] for r in recs])
        by_hist = {}
        for lo, hi in HISTORY_BINS:
            mask = (hist >= lo) & (hist <= hi)
            if mask.sum() >= 20:
                by_hist[f"{lo}-{hi if hi < 10**9 else 'inf'}"] = {
                    "n": int(mask.sum()), **summarise(np.asarray(wall)[mask], pcts)}
        slowest = sorted(recs, key=lambda r: -r["wall"])[:5]
        stage_means = {s: float(np.mean([r[s] for r in recs])) for s in STAGES}
        results[mode] = {
            "total_ms": summarise(wall, pcts),
            "stages_ms": {s: summarise([r[s] for r in recs], pcts) for s in STAGES},
            "stage_share_of_mean": {s: round(v / sum(stage_means.values()), 4) for s, v in stage_means.items()},
            "stage1_ms": summarise([sum(r[s] for s in STAGES[:5]) for r in recs], pcts),
            "stage2_ms": summarise([sum(r[s] for s in STAGES[5:]) for r in recs], pcts),
            "by_history_length": by_hist,
            "slowest_requests": [{"wall_ms": round(r["wall"], 3), "history_len": r["history_len"],
                                  "n_candidates": r["n_candidates"],
                                  "top_stage": max(STAGES, key=lambda s: r[s])} for r in slowest],
            "meets_sla_p99": results_ok(wall, conf["cost"]["sla_p99_ms"]),
            "wall_ms_per_request": [round(w, 4) for w in wall],
        }
    speedup = {f"p{p}": round(results["as_is"]["total_ms"][f"p{p}"] / results["serving"]["total_ms"][f"p{p}"], 1)
               for p in pcts}
    env_file = REPO_ROOT / "reports" / "q4_environment.json"
    return {
        "dataset": dataset, "scale": load_config(REPO_ROOT / "config" / f"{dataset}.yaml").scale,
        "n_requests": len(req_idx), "warmup_requests": len(warm_idx),
        "threads": conf["hardware"]["threads"], "device": conf["hardware"]["device"],
        "pipeline": conf["pipeline"], "model": conf["models"][dataset],
        "load_seconds": round(load_s, 1), "benchmark_seconds": round(run_s, 1),
        "machine": json.loads(env_file.read_text())["machine"] if env_file.exists() else None,
        "machine_state_before": state_before, "machine_state_after": machine_state(),
        "modes": results, "serving_speedup_over_as_is": speedup,
    }


def results_ok(wall, sla_ms) -> bool:
    return bool(np.percentile(wall, 99) < sla_ms)


def print_report(res: dict) -> None:
    print(f"\n[{res['dataset']}] {res['n_requests']:,} requests (+{res['warmup_requests']} warm-up), "
          f"{res['threads']} thread, {res['machine_state_before']['power']} power, "
          f"load avg {res['machine_state_before']['load_average_1_5_15']}")
    for mode, r in res["modes"].items():
        t = r["total_ms"]
        print(f"  {mode:<8} total  mean {t['mean']:7.2f}  p50 {t['p50']:7.2f}  p95 {t['p95']:7.2f}  "
              f"p99 {t['p99']:7.2f}  max {t['max']:7.2f} ms   p99<SLA: {r['meets_sla_p99']}")
        for s in STAGES:
            st = r["stages_ms"][s]
            print(f"           {s:<10} mean {st['mean']:7.3f}  p50 {st['p50']:7.3f}  p99 {st['p99']:7.3f}  "
                  f"({100 * r['stage_share_of_mean'][s]:4.1f}% of mean)")
        print("           p99 by history length: " + ", ".join(
            f"{b} (n={v['n']}) {v['p99']:.2f}" for b, v in r["by_history_length"].items()))
        print("           slowest: " + ", ".join(
            f"{s['wall_ms']:.1f}ms hist={s['history_len']} [{s['top_stage']}]" for s in r["slowest_requests"][:3]))
    print(f"  serving speed-up over as_is: {res['serving_speedup_over_as_is']}  "
          f"(load {res['load_seconds']}s, benchmark {res['benchmark_seconds']}s)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n", type=int, default=None, help="requests (default from config/serving.yaml)")
    parser.add_argument("--out", type=Path, default=None, help="default reports/q4_latency_<dataset>.json")
    args = parser.parse_args(argv)

    dataset = load_config(args.config).dataset
    res = run_benchmark(dataset, args.n)
    print_report(res)
    out = args.out or REPO_ROOT / "reports" / f"q4_latency_{dataset}.json"
    out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
