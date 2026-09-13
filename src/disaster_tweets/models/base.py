from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class TextClassifier(ABC):
    """Common interface. ``kind`` must equal the registry key."""

    kind: str = "base"

    @abstractmethod
    def fit(self, texts: list[str], y: np.ndarray) -> TextClassifier: ...

    @abstractmethod
    def predict_proba(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> TextClassifier: ...

    def params(self) -> dict[str, Any]:
        return {}

    def explain(self, text: str, top_n: int = 8) -> list[tuple[str, float]] | None:
        """Per-prediction token contributions, or None if not supported."""
        return None

    def _write_meta(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.json").write_text(
            json.dumps({"kind": self.kind, "params": self.params()}, indent=2)
        )
