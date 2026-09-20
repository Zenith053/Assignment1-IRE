#!/usr/bin/env python3
"""Check a Codabench submission zip against its source test file, independently of the writer.

Rules checked (MIND competition 13967 / EB-NeRD competition 2469 submission guidelines):
  1. the zip contains exactly one entry, at its root: `prediction.txt` (MIND) or
     `predictions.txt` (EB-NeRD); no folders, no __MACOSX
  2. one line per impression, in the source file's row order, with the same impression ids
  3. each line is `<id> [r1,...,rn]` where n is that impression's candidate count and the
     ranks are exactly the integers 1..n

Streams both files, so it runs on the 13.5M-impression EB-NeRD test set in bounded memory.

Usage
-----
    python src/submission/validate_zip.py --dataset mind \\
        --zip reports/submissions/submission_mind_large_test_nrms.zip --source data/raw/mind/MINDlarge_test
    python src/submission/validate_zip.py --dataset ebnerd \\
        --zip reports/submissions/submission_ebnerd_testset_nrms_fresh.zip \\
        --source data/raw/ebnerd/ebnerd_testset/ebnerd_testset/test
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

EXPECTED = {"mind": "prediction.txt", "ebnerd": "predictions.txt"}


def mind_source(d: Path):
    with open(d / "behaviors.tsv", encoding="utf-8") as fh:
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            yield cols[0], len(cols[4].split())


def ebnerd_source(d: Path):
    f = pq.ParquetFile(d / "behaviors.parquet")
    for batch in f.iter_batches(batch_size=500_000, columns=["impression_id", "article_ids_inview"]):
        ids = batch.column("impression_id").to_pylist()
        lens = pc.list_value_length(batch.column("article_ids_inview")).to_pylist()
        yield from zip((str(i) for i in ids), lens)


def validate(dataset: str, zip_path: Path, source: Path, max_errors: int = 10) -> dict:
    errors: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if names != [EXPECTED[dataset]]:
            errors.append(f"zip entries {names}, expected exactly ['{EXPECTED[dataset]}']")
            return {"valid": False, "errors": errors}
        n = 0
        src = mind_source(source) if dataset == "mind" else ebnerd_source(source)
        with zf.open(names[0]) as raw:
            for line in io.TextIOWrapper(raw, encoding="utf-8"):
                n += 1
                try:
                    exp_id, exp_len = next(src)
                except StopIteration:
                    errors.append(f"line {n}: more lines than impressions in the source")
                    break
                parts = line.rstrip("\n").split(" ")
                if len(parts) != 2 or not (parts[1].startswith("[") and parts[1].endswith("]")):
                    errors.append(f"line {n}: not '<id> [ranks]': {line[:60]!r}")
                elif parts[0] != exp_id:
                    errors.append(f"line {n}: impression id {parts[0]} != source {exp_id}")
                else:
                    body = parts[1][1:-1]
                    ranks = [int(x) for x in body.split(",")] if body else []
                    if len(ranks) != exp_len or sorted(ranks) != list(range(1, exp_len + 1)):
                        errors.append(f"line {n} (id {exp_id}): ranks are not a permutation of 1..{exp_len}")
                if len(errors) >= max_errors:
                    break
        if not errors and next(src, None) is not None:
            errors.append(f"source has more impressions than the {n:,} lines written")
    return {"valid": not errors, "zip": str(zip_path), "entries": names, "lines": n,
            "zip_mb": round(zip_path.stat().st_size / 1e6, 1), "errors": errors}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=list(EXPECTED), required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True,
                        help="MIND split dir with behaviors.tsv, or EB-NeRD dir with behaviors.parquet")
    args = parser.parse_args(argv)
    res = validate(args.dataset, args.zip, args.source)
    print(json.dumps(res, indent=2))
    return 0 if res["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
