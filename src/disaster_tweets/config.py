"""Run configuration.

A run is fully described by a small YAML file (see ``configs/``) so that every
result in ``reports/`` can be regenerated with one command.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _project_root() -> Path:
    """Where data/, configs/, runs/ and reports/ live.

    Editable installs resolve to the repository; a plain ``pip install`` (the
    Docker image) resolves to site-packages, so prefer an explicit
    ``DT_PROJECT_ROOT`` or a working directory that already holds the data.
    """
    explicit = os.getenv("DT_PROJECT_ROOT")
    if explicit:
        return Path(explicit).resolve()
    cwd = Path.cwd()
    if (cwd / "data" / "raw" / "train.csv").exists():
        return cwd
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW_TRAIN = DATA_DIR / "raw" / "train.csv"
RUNS_DIR = PROJECT_ROOT / "runs"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


@dataclass(frozen=True)
class RunConfig:
    """Everything that determines a training run."""

    name: str
    model: str
    params: dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    #: Text preprocessing options handed to ``preprocess.normalise``.
    preprocess: dict[str, Any] = field(default_factory=dict)
    #: Prepend the Kaggle ``keyword`` column to the text when present.
    use_keyword: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "params": dict(self.params),
            "seed": self.seed,
            "test_size": self.test_size,
            "cv_folds": self.cv_folds,
            "preprocess": dict(self.preprocess),
            "use_keyword": self.use_keyword,
        }
