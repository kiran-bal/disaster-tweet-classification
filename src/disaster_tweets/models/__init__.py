"""Model registry.

Every model exposes the same tiny interface so that ``train`` and ``evaluate``
never branch on model type:

    model.fit(texts: list[str], y: np.ndarray) -> self
    model.predict_proba(texts: list[str]) -> np.ndarray  # P(disaster), shape (n,)
    model.save(dir) / Model.load(dir)

Names map to builders; ``params`` come straight from the run's YAML.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import TextClassifier
from .sklearn_models import TfidfLogReg, TfidfNaiveBayes, TfidfSVM

_REGISTRY: dict[str, Callable[..., TextClassifier]] = {
    "tfidf_lr": TfidfLogReg,
    "tfidf_svm": TfidfSVM,
    "tfidf_nb": TfidfNaiveBayes,
}


def _lazy_optional() -> None:
    """Register models whose dependencies are optional extras."""
    try:
        from .embeddings import SentenceEmbeddingLogReg

        _REGISTRY.setdefault("minilm_lr", SentenceEmbeddingLogReg)
    except ImportError:
        pass
    try:
        from .transformer import FineTunedTransformer

        _REGISTRY.setdefault("distilbert", FineTunedTransformer)
    except ImportError:
        pass


def build(name: str, **params: Any) -> TextClassifier:
    _lazy_optional()
    if name not in _REGISTRY:
        raise KeyError(f"unknown model '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**params)


def available() -> list[str]:
    _lazy_optional()
    return sorted(_REGISTRY)


def load(path) -> TextClassifier:
    """Load any saved model by reading the ``kind`` recorded at save time."""
    import json
    from pathlib import Path

    meta = json.loads((Path(path) / "model.json").read_text())
    _lazy_optional()
    cls = _REGISTRY[meta["kind"]]
    return cls.load(path)  # type: ignore[attr-defined]


__all__ = ["TextClassifier", "available", "build", "load"]
