"""Q4 phase 2: the byte counter every memory number depends on."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.serving.measure_memory import deep_bytes  # noqa: E402


def test_buffers_are_counted_exactly():
    assert deep_bytes(np.zeros((1000, 384), dtype=np.float32)) == 1000 * 384 * 4
    assert deep_bytes(torch.zeros(500, 400)) == 500 * 400 * 4
    m = sparse.random(200, 300, density=0.05, format="csr", dtype=np.float32, random_state=0)
    assert deep_bytes(m) == m.data.nbytes + m.indices.nbytes + m.indptr.nbytes


def test_faiss_index_counts_its_vectors():
    import faiss
    idx = faiss.IndexFlatIP(64)
    idx.add(np.zeros((1000, 64), dtype=np.float32))
    assert 1000 * 64 * 4 <= deep_bytes(idx) < 1000 * 64 * 4 + 1024


def test_shared_objects_counted_once_within_a_call():
    arr = np.zeros(10_000, dtype=np.float64)
    assert deep_bytes([arr, arr, {"a": arr}]) < 2 * arr.nbytes


class _Holder:
    def __init__(self):
        self.vectors = np.zeros((100, 100), dtype=np.float32)
        self.lookup = {"x": np.zeros(1000, dtype=np.int64)}


def test_custom_objects_are_walked_not_just_shallow_sized():
    assert deep_bytes(_Holder()) >= 100 * 100 * 4 + 1000 * 8


def test_dataframe_counts_object_columns_deeply():
    df = pd.DataFrame({"tokens": [["word"] * 50 for _ in range(100)], "n": np.arange(100)})
    shallow = df.memory_usage(deep=False).sum()
    assert deep_bytes(df) > shallow + 100 * sys.getsizeof([None] * 50) * 0.9
