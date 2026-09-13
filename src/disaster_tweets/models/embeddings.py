"""Frozen sentence embeddings + logistic regression.

A cheap way to test whether pretrained semantics beat lexical features on this
task without fine-tuning anything. Encoding 7k tweets with MiniLM takes seconds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from .base import TextClassifier


class SentenceEmbeddingLogReg(TextClassifier):
    kind = "minilm_lr"

    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", C: float = 1.0, **_: Any  # noqa: N803
    ) -> None:
        self.model_name = model_name
        self.C = C
        self._encoder: SentenceTransformer | None = None
        self.clf: LogisticRegression | None = None

    def params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "C": self.C}

    @property
    def encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self.encoder.encode(list(texts), batch_size=128, normalize_embeddings=True,
                                   show_progress_bar=False)

    def fit(self, texts: list[str], y: np.ndarray) -> SentenceEmbeddingLogReg:
        self.clf = LogisticRegression(C=self.C, max_iter=2000)
        self.clf.fit(self._encode(texts), y)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        assert self.clf is not None
        return self.clf.predict_proba(self._encode(texts))[:, 1]

    def save(self, path: Path) -> None:
        self._write_meta(path)
        joblib.dump(self.clf, Path(path) / "clf.joblib")

    @classmethod
    def load(cls, path: Path) -> SentenceEmbeddingLogReg:
        import json

        meta = json.loads((Path(path) / "model.json").read_text())
        obj = cls(**meta["params"])
        obj.clf = joblib.load(Path(path) / "clf.joblib")
        return obj
