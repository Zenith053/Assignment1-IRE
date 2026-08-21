"""Configuration loading and path resolution.

One YAML file per dataset drives every stage of the pipeline. This module turns
that file into a `Config` object with absolute paths already resolved, so no
downstream module ever builds a path by hand or cares about the cwd.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# src/common/config.py -> repo root is three levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_FEATURE_DIR = REPO_ROOT / "data" / "feature_store"


@dataclass
class Config:
    """Resolved configuration for a single dataset."""

    dataset: str
    language: str
    scale: str | None
    raw: dict[str, Path]              # logical name -> absolute path
    capabilities: dict[str, bool]
    text: dict[str, Any]
    split: dict[str, Any]
    serving_time_unavailable: list[str] = field(default_factory=list)
    raw_dir: Path = DEFAULT_RAW_DIR
    processed_dir: Path = DEFAULT_PROCESSED_DIR
    feature_dir: Path = DEFAULT_FEATURE_DIR

    def can(self, capability: str) -> bool:
        """Check a declared capability; unknown capabilities are False, not an error.

        Downstream code branches on this rather than on `dataset`, which keeps
        the pipeline dataset-agnostic.
        """
        return bool(self.capabilities.get(capability, False))

    def require(self, *names: str) -> None:
        """Fail early with a clear message if a declared raw input is missing."""
        missing = [n for n in names if not self.raw[n].exists()]
        if missing:
            details = "\n".join(f"  {n}: {self.raw[n]}" for n in missing)
            raise FileNotFoundError(
                f"[{self.dataset}] missing raw input(s):\n{details}\n"
                f"  Run: python src/data/download.py"
            )

    @property
    def processed(self) -> Path:
        """Where clean.py writes the unified tables for this dataset."""
        return self.processed_dir / self.dataset

    @property
    def features(self) -> Path:
        return self.feature_dir / self.dataset


def load_config(path: str | Path, raw_dir: Path | None = None) -> Config:
    """Read a dataset YAML and resolve every declared path to an absolute one."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))

    root = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    scale = spec.get("scale")

    resolved: dict[str, Path] = {}
    for name, rel in (spec.get("raw") or {}).items():
        # `{scale}` lets ebnerd.yaml switch demo -> small without touching paths.
        if scale:
            rel = rel.format(scale=scale)
        resolved[name] = root / rel

    return Config(
        dataset=spec["dataset"],
        language=spec.get("language", "en"),
        scale=scale,
        raw=resolved,
        capabilities=spec.get("capabilities") or {},
        text=spec.get("text") or {},
        split=spec.get("split") or {},
        serving_time_unavailable=spec.get("serving_time_unavailable") or [],
        raw_dir=root,
    )
