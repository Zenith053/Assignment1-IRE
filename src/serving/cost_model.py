#!/usr/bin/env python3
"""Q4 phase 4 (Q4.3): cores, RAM and cost per 1,000 queries at a p99 SLA.

Back-of-envelope, from measured numbers plus stated assumptions (config/serving.yaml `cost`):

  1. capacity per core   = 1000 / mean service time (ms)            <- Phase 3, measured
  2. cores for a target  = ceil(QPS / (capacity per core x utilisation))
  3. p99 under load      = queueing simulation: Poisson arrivals at the target QPS,
                           first-free-core dispatch, service times resampled from the
                           2,000 measured requests. Checks the utilisation target
                           actually keeps p99 under the SLA, rather than assuming it.
  4. RAM                 = cores x (served structures + Python/torch runtime)  <- Phase 2
                           (one single-threaded worker process per core, each with its
                           own copy of the indexes and model)
  5. cost / 1,000 queries = cores x price per core-hour / (QPS x 3.6)

Assumptions, all parameters: price per vCPU-hour, cloud vCPU slowdown vs an M4
core, RAM per vCPU, utilisation target. The network, request parsing and any
database round trip are not in the measured service time.

Writes reports/q4_cost.json.

Usage
-----
    python src/serving/cost_model.py
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.serving import load_serving_config  # noqa: E402


# --------------------------------------------------------------------------- #
# queueing
# --------------------------------------------------------------------------- #

def simulate_queue(service_ms: np.ndarray, cores: int, qps: float, n_arrivals: int,
                   warmup_fraction: float = 0.05, seed: int = 13) -> dict:
    """Response times (wait + service) of an FCFS pool of `cores` identical workers.

    Arrivals are Poisson at `qps`; each request goes to the earliest-free worker and
    takes a service time drawn (with replacement) from `service_ms`.
    """
    rng = np.random.default_rng(seed)
    arrivals = np.cumsum(rng.exponential(1000.0 / qps, size=n_arrivals))    # ms
    services = rng.choice(np.asarray(service_ms, dtype=np.float64), size=n_arrivals, replace=True)
    free_at = [0.0] * cores
    response = np.empty(n_arrivals)
    for i in range(n_arrivals):
        earliest = free_at[0]
        start = arrivals[i] if arrivals[i] > earliest else earliest
        heapq.heapreplace(free_at, start + services[i])
        response[i] = start + services[i] - arrivals[i]
    kept = response[int(n_arrivals * warmup_fraction):]
    wait = kept - services[int(n_arrivals * warmup_fraction):]
    return {"p50_ms": float(np.percentile(kept, 50)), "p99_ms": float(np.percentile(kept, 99)),
            "mean_wait_ms": float(wait.mean()), "p99_wait_ms": float(np.percentile(wait, 99))}


def cores_needed(qps: float, capacity_per_core: float, utilisation: float) -> int:
    return max(1, math.ceil(qps / (capacity_per_core * utilisation)))


def cost_per_1k(cores: int, usd_per_core_hour: float, qps: float) -> float:
    return cores * usd_per_core_hour / (qps * 3600 / 1000)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

def load_inputs(dataset: str) -> tuple[dict, dict]:
    lat = json.loads((REPO_ROOT / "reports" / f"q4_latency_{dataset}.json").read_text())
    mem = json.loads((REPO_ROOT / "reports" / f"q4_memory_{dataset}.json").read_text())
    return lat, mem


def build(conf: dict, datasets: list[str]) -> dict:
    c = conf["cost"]
    sim = c["simulation"]
    sla, util, base_price = c["sla_p99_ms"], c["target_utilisation"], c["usd_per_core_hour"]
    out = {"assumptions": {
        "sla_p99_ms": sla, "target_utilisation": util, "usd_per_core_hour": base_price,
        "price_sensitivity_usd_per_core_hour": c["price_sensitivity_usd_per_core_hour"],
        "cloud_slowdown_factors": c["cloud_slowdown_factors"], "ram_gb_per_vcpu": c["ram_gb_per_vcpu"],
        "workers": "one single-threaded worker process per core, each holding its own copy of indexes + model",
        "not_included": "network, request parsing, load balancer, profile/click-counter storage round trips",
        "hardware_measured_on": "Apple M4, 1 thread (see reports/q4_environment.json)",
    }, "datasets": {}}

    for ds in datasets:
        lat, mem = load_inputs(ds)
        runtime_mb = mem["process"]["rss_after_all_imports_mb"]
        served_mb = mem["summary"]["served_total_mb"]
        worker_ram_gb = (served_mb + runtime_mb) / 1000
        ds_out = {"worker_ram_gb": round(worker_ram_gb, 2),
                  "worker_ram_fits_per_vcpu": worker_ram_gb <= c["ram_gb_per_vcpu"],
                  "modes": {}}

        for mode, res in lat["modes"].items():
            service = np.asarray(res["wall_ms_per_request"])
            rows = []
            for slowdown in c["cloud_slowdown_factors"]:
                svc = service * slowdown
                capacity = 1000.0 / svc.mean()
                for qps in c["target_qps"]:
                    cores = cores_needed(qps, capacity, util)
                    q = simulate_queue(svc, cores, qps, sim["n_arrivals"], sim["warmup_fraction"], sim["seed"])
                    rows.append({
                        "cloud_slowdown": slowdown, "qps": qps,
                        "service_mean_ms": round(float(svc.mean()), 3),
                        "service_p99_ms": round(float(np.percentile(svc, 99)), 3),
                        "capacity_per_core_rps": round(capacity, 1), "cores": cores,
                        "actual_utilisation": round(qps / (capacity * cores), 3),
                        "sim_p50_ms": round(q["p50_ms"], 3), "sim_p99_ms": round(q["p99_ms"], 3),
                        "sim_p99_wait_ms": round(q["p99_wait_ms"], 3),
                        "meets_sla": q["p99_ms"] < sla,
                        "ram_gb": round(cores * worker_ram_gb, 1),
                        "usd_per_hour": round(cores * base_price, 3),
                        "usd_per_1k_queries": cost_per_1k(cores, base_price, qps),
                        "usd_per_1k_by_price": {str(p): cost_per_1k(cores, p, qps)
                                                for p in c["price_sensitivity_usd_per_core_hour"]},
                        "usd_per_month_24x7": round(cores * base_price * 24 * 30, 2),
                    })

            # Why 60%: p99 against utilisation for a fixed pool, measured service times.
            n_cores = sim["utilisation_curve_cores"]
            capacity = 1000.0 / service.mean()
            curve = []
            for rho in sim["utilisation_curve"]:
                qps = rho * capacity * n_cores
                q = simulate_queue(service, n_cores, qps, sim["n_arrivals"], sim["warmup_fraction"], sim["seed"])
                curve.append({"utilisation": rho, "qps": round(qps, 1), "sim_p50_ms": round(q["p50_ms"], 3),
                              "sim_p99_ms": round(q["p99_ms"], 3), "p99_vs_no_queue":
                              round(q["p99_ms"] / float(np.percentile(service, 99)), 2)})
            within = [pt["utilisation"] for pt in curve if pt["sim_p99_ms"] < sla]
            ds_out["modes"][mode] = {"table": rows, "utilisation_curve": {
                "cores": n_cores, "points": curve,
                "highest_utilisation_meeting_sla": max(within) if within else None}}

        base = {m: next(r for r in ds_out["modes"][m]["table"] if r["cloud_slowdown"] == 1 and r["qps"] == 1000)
                for m in ds_out["modes"]}
        ds_out["serving_vs_as_is_cores_at_1000qps"] = {m: base[m]["cores"] for m in base}
        out["datasets"][ds] = ds_out
    return out


def print_report(res: dict) -> None:
    a = res["assumptions"]
    print(f"assumptions: ${a['usd_per_core_hour']}/core-hour, utilisation target {a['target_utilisation']}, "
          f"SLA p99 < {a['sla_p99_ms']} ms, {a['ram_gb_per_vcpu']} GB RAM/vCPU")
    for ds, d in res["datasets"].items():
        print(f"\n[{ds}] worker RAM {d['worker_ram_gb']} GB (fits {a['ram_gb_per_vcpu']} GB/vCPU: {d['worker_ram_fits_per_vcpu']})")
        for mode, m in d["modes"].items():
            print(f"  {mode}")
            print(f"    {'slowdown':>8} {'QPS':>6} {'mean ms':>8} {'req/s/core':>10} {'cores':>6} "
                  f"{'sim p99':>8} {'SLA':>4} {'RAM GB':>7} {'$/hour':>7} {'$ per 1k queries':>17}")
            for r in m["table"]:
                print(f"    {r['cloud_slowdown']:>7}x {r['qps']:>6} {r['service_mean_ms']:>8.2f} "
                      f"{r['capacity_per_core_rps']:>10.0f} {r['cores']:>6} {r['sim_p99_ms']:>8.2f} "
                      f"{'ok' if r['meets_sla'] else 'NO':>4} {r['ram_gb']:>7.1f} {r['usd_per_hour']:>7.2f} "
                      f"{r['usd_per_1k_queries']:>17.6f}")
            uc = m["utilisation_curve"]
            print(f"    utilisation curve ({uc['cores']} cores, measured service times): " + ", ".join(
                f"{p['utilisation']:.0%} p99 {p['sim_p99_ms']:.2f} ms" for p in uc["points"]))
            print(f"    highest tested utilisation still meeting the SLA: {uc['highest_utilisation_meeting_sla']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=["mind", "ebnerd"])
    args = parser.parse_args(argv)
    res = build(load_serving_config(), args.datasets)
    print_report(res)
    out = REPO_ROOT / "reports" / "q4_cost.json"
    out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
