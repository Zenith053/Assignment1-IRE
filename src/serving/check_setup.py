#!/usr/bin/env python3
"""Q4 phase 0: verify the serving setup before anything is measured.

Checks, per dataset in config/serving.yaml:
  1. the checkpoint and its saved test scores exist
  2. the checkpoint's recorded variant (popularity/freshness/gate) matches the config
  3. the checkpoint loads into `nrms.NRMS` on CPU with every key matching
  4. the saved test scores cover the full test split
and, once: that thread pinning takes effect, and the machine/library versions.

Writes reports/q4_environment.json (latency numbers are reported with it).

Usage
-----
    python src/serving/check_setup.py
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.serving import THREAD_ENV_VARS, load_serving_config, pin_threads  # noqa: E402


def machine_info() -> dict:
    def sysctl(key):
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
        except OSError:
            return None

    import faiss
    import numpy
    import pandas
    import torch
    mem = sysctl("hw.memsize")
    return {
        "cpu": sysctl("machdep.cpu.brand_string") or platform.processor(),
        "cores_logical": os.cpu_count(),
        "cores_performance": sysctl("hw.perflevel0.physicalcpu"),
        "cores_efficiency": sysctl("hw.perflevel1.physicalcpu"),
        "ram_gb": round(int(mem) / 2**30, 1) if mem else None,
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": numpy.__version__, "pandas": pandas.__version__,
        "torch": torch.__version__, "faiss": faiss.__version__,
    }


def check_dataset(name: str, spec: dict) -> dict:
    import numpy as np
    import torch

    from src.baseline import candidate_signals as sig
    from src.baseline import nrms
    from src.baseline.news_data import load_news_tokens
    from src.common.config import load_config
    from src.common.io import read_table

    report = {"dataset": name, "checks": {}}
    ok = report["checks"]

    ckpt_path, scores_path = REPO_ROOT / spec["checkpoint"], REPO_ROOT / spec["test_scores"]
    ok["checkpoint_exists"] = ckpt_path.exists()
    ok["test_scores_exist"] = scores_path.exists()
    if not (ok["checkpoint_exists"] and ok["test_scores_exist"]):
        return report

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    recorded = ckpt["meta"]["variant"]
    ok["variant_matches_config"] = (recorded["popularity"] == spec["popularity"]
                                    and recorded["freshness"] == spec["freshness"]
                                    and recorded["gate"] == spec["gate"])
    report["checkpoint_variant"] = recorded

    cfg = load_config(REPO_ROOT / "config" / f"{name}.yaml")
    news = load_news_tokens(cfg)
    n_signals = len(sig.signal_names(spec["popularity"], spec["freshness"]))
    model = nrms.NRMS(news.tokens, nrms.load_word_embeddings(cfg, news),
                      n_signals=n_signals, gate=spec["gate"])
    result = model.load_state_dict(ckpt["state_dict"], strict=True)
    ok["checkpoint_loads_strict"] = not result.missing_keys and not result.unexpected_keys
    report["n_parameters"] = sum(p.numel() for p in model.parameters())

    z = np.load(scores_path)
    n_test = len(read_table(cfg.processed / "test" / "impressions.parquet", "impressions"))
    ok["scores_cover_full_test_split"] = len(z["offsets"]) - 1 == n_test
    report["n_test_impressions"] = n_test
    return report


def main() -> int:
    conf = load_serving_config()
    threads = pin_threads(conf["hardware"]["threads"])
    want = conf["hardware"]["threads"]
    pinned = (threads["faiss_omp"] == want and threads["torch"] == want
              and all(v == want for v in threads["threadpools"].values()))

    datasets = [check_dataset(name, spec) for name, spec in conf["models"].items()]
    import numpy as np
    blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"].get("name")
    env_vars = {v: os.environ.get(v) for v in THREAD_ENV_VARS}
    env_pinned = all(val == str(want) for val in env_vars.values())
    pinned = pinned and env_pinned
    env = {"machine": machine_info(), "threads": threads, "thread_env": env_vars,
           "numpy_blas": blas, "threads_pinned": pinned, "datasets": datasets,
           "caveat": ("numpy/scipy BLAS is Apple Accelerate, limited only via "
                      "VECLIB_MAXIMUM_THREADS and partly offloaded to the AMX coprocessor; "
                      "per-core figures from this M4 are indicative, not cloud-vCPU exact.")}

    all_ok = pinned and all(all(d["checks"].values()) for d in datasets)
    print(f"threads pinned to {want}: {'OK' if pinned else 'FAILED'}  {threads}  env={env_vars}  numpy BLAS={blas}")
    for d in datasets:
        for check, passed in d["checks"].items():
            print(f"  [{d['dataset']}] {check:<30} {'OK' if passed else 'FAILED'}")
    m = env["machine"]
    print(f"machine: {m['cpu']}, {m['cores_performance']}P+{m['cores_efficiency']}E cores, "
          f"{m['ram_gb']} GB RAM, torch {m['torch']}, faiss {m['faiss']}")

    out = REPO_ROOT / "reports" / "q4_environment.json"
    out.write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    print(f"-> {out}\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
