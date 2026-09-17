"""Q4 phase 4: the queueing simulation and cost arithmetic.

The simulator is checked against queueing theory where an exact answer exists:
an M/M/1 queue (Poisson arrivals, exponential service, one server) has
exponentially distributed response time with rate mu - lambda, so its mean is
1/(mu - lambda) and its p99 is ln(100)/(mu - lambda).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.serving.cost_model import cores_needed, cost_per_1k, simulate_queue  # noqa: E402


def test_mm1_matches_theory():
    mean_service_ms = 2.0                    # mu = 500 req/s
    qps = 250.0                              # rho = 0.5 -> mu - lambda = 250 req/s = 0.25 per ms
    service = np.random.default_rng(1).exponential(mean_service_ms, size=200_000)
    res = simulate_queue(service, cores=1, qps=qps, n_arrivals=200_000, seed=3)
    rate = 1 / mean_service_ms - qps / 1000  # per ms
    assert res["p99_ms"] == pytest.approx(math.log(100) / rate, rel=0.08)
    assert res["p50_ms"] == pytest.approx(math.log(2) / rate, rel=0.08)


def test_no_queueing_when_lightly_loaded():
    service = np.full(1000, 1.0)             # deterministic 1 ms
    res = simulate_queue(service, cores=8, qps=50, n_arrivals=20_000)
    assert res["p99_ms"] == pytest.approx(1.0, abs=1e-9) and res["p99_wait_ms"] == pytest.approx(0.0, abs=1e-9)


def test_more_load_never_lowers_p99():
    service = np.random.default_rng(0).gamma(2.0, 0.5, size=5000)
    p99 = [simulate_queue(service, cores=4, qps=q, n_arrivals=50_000)["p99_ms"] for q in (500, 2000, 3500)]
    assert p99[0] <= p99[1] <= p99[2]


def test_cores_and_cost_arithmetic():
    # 1.25 ms mean -> 800 req/s per core; 1000 QPS at 60% -> 1000 / 480 -> 3 cores
    assert cores_needed(1000, 800, 0.6) == 3
    assert cores_needed(1, 800, 0.6) == 1
    # 3 cores x $0.04/h = $0.12/h over 3.6M queries/h -> $0.0000333 per 1k
    assert cost_per_1k(3, 0.04, 1000) == pytest.approx(0.12 / 3600)
