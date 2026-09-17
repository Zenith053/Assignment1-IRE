"""Q4: serving & scale analysis of the two-stage pipeline.

Shared helpers for every Q4 script: the serving config (`config/serving.yaml`)
and single-thread pinning. Latency and cost are reported *per core*, so every
library that can spawn threads (BLAS, FAISS/OpenMP, PyTorch) must be held to
the configured count; by default FAISS uses every core and PyTorch several.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVING_CONFIG = REPO_ROOT / "config" / "serving.yaml"

# BLAS libraries read these once, when numpy/scipy first load them, so they are
# set at import of this package; Q4 scripts import `src.serving` before numpy.
# numpy/scipy here use Apple Accelerate, which threadpoolctl cannot see or limit -
# VECLIB_MAXIMUM_THREADS is its only control. Measured effect on this M4 is small
# (Accelerate offloads matmuls to the AMX coprocessor rather than to CPU threads),
# which is itself a caveat for "per core" numbers: an M4 core is not a cloud vCPU.
THREAD_ENV_VARS = ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def load_serving_config(path: Path = SERVING_CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


for _var in THREAD_ENV_VARS:
    os.environ.setdefault(_var, str(load_serving_config()["hardware"]["threads"]))


def pin_threads(n: int) -> dict:
    """Limit BLAS, OpenMP (FAISS) and PyTorch to `n` threads; return what is now in effect."""
    import faiss
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits

    threadpool_limits(limits=n)
    faiss.omp_set_num_threads(n)
    torch.set_num_threads(n)
    return {
        "faiss_omp": faiss.omp_get_max_threads(),
        "torch": torch.get_num_threads(),
        "threadpools": {f"{p['internal_api']}:{Path(p['filepath']).name}": p["num_threads"]
                        for p in threadpool_info()},
    }
