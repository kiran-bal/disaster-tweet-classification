"""Linear baselines on TF-IDF features.

These are the models to beat. On short-text classification they are fast,
well calibrated after a simple calibration step, and interpretable down to the
individual n-gram, which is why the API serves one of them by default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ..features import tfidf_word_char
from .base import TextClassifier


class _SklearnPipelineModel(TextClassifier):
    def __init__(self, **params: Any) -> None:
        self._params = params
        self.pipeline: Pipeline | None = None

    def params(self) -> dict[str, Any]:
        return dict(self._params)

    def _features(self):
        feat_keys = {"word_ngram_max", "char_ngram", "min_df", "max_features_word",
                     "max_features_char", "sublinear_tf"}
        kwargs = {k: v for k, v in self._params.items() if k in feat_keys}
        if "char_ngram" in kwargs:
            kwargs["char_ngram"] = tuple(kwargs["char_ngram"])
        return tfidf_word_char(**kwargs)

    def _estimator(self):
        raise NotImplementedError

    def fit(self, texts: list[str], y: np.ndarray) -> _SklearnPipelineModel:
        self.pipeline = Pipeline([("features", self._features()), ("clf", self._estimator())])
        self.pipeline.fit(texts, y)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        assert self.pipeline is not None, "call fit() or load() first"
        return self.pipeline.predict_proba(texts)[:, 1]

    def save(self, path: Path) -> None:
        self._write_meta(path)
        joblib.dump(self.pipeline, Path(path) / "pipeline.joblib")

    @classmethod
    def load(cls, path: Path):
        import json

        meta = json.loads((Path(path) / "model.json").read_text())
        obj = cls(**meta["params"])
        obj.pipeline = joblib.load(Path(path) / "pipeline.joblib")
        return obj

    # --- explainability for linear models --------------------------------
    def _linear_coef(self) -> np.ndarray | None:
        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf, "coef_"):
            return clf.coef_.ravel()
        return None

    def feature_names(self) -> np.ndarray:
        return self.pipeline.named_steps["features"].get_feature_names_out()

    def top_features(self, n: int = 25) -> dict[str, list[tuple[str, float]]]:
        """Globally most positive / negative n-grams."""
        coef = self._linear_coef()
        if coef is None:
            return {}
        names = self.feature_names()
        order = np.argsort(coef)
        neg = [(str(names[i]).split("__", 1)[-1], float(coef[i])) for i in order[:n]]
        pos = [(str(names[i]).split("__", 1)[-1], float(coef[i])) for i in order[::-1][:n]]
        return {"disaster": pos, "not_disaster": neg}

    def explain(self, text: str, top_n: int = 8) -> list[tuple[str, float]] | None:
        coef = self._linear_coef()
        if coef is None:
            return None
        x = self.pipeline.named_steps["features"].transform([text])
        contrib = x.multiply(coef).tocoo()
        names = self.feature_names()
        pairs = sorted(
            (
                (str(names[j]).split("__", 1)[-1], float(v))
                for j, v in zip(contrib.col, contrib.data, strict=True)
            ),
            key=lambda p: -abs(p[1]),
        )
        return pairs[:top_n]


class TfidfLogReg(_SklearnPipelineModel):
    kind = "tfidf_lr"

    def _estimator(self):
        return LogisticRegression(
            C=self._params.get("C", 4.0),
            max_iter=2000,
            class_weight=self._params.get("class_weight"),
            solver="liblinear",
        )


class TfidfSVM(_SklearnPipelineModel):
    """Linear SVM wrapped in isotonic/sigmoid calibration so it yields probabilities."""

    kind = "tfidf_svm"

    def _estimator(self):
        base = LinearSVC(C=self._params.get("C", 0.5))
        return CalibratedClassifierCV(base, method=self._params.get("calibration", "sigmoid"), cv=5)

    def _linear_coef(self) -> np.ndarray | None:
        clf = self.pipeline.named_steps["clf"]
        try:
            coefs = [c.estimator.coef_.ravel() for c in clf.calibrated_classifiers_]
            return np.mean(coefs, axis=0)
        except AttributeError:
            return None


class TfidfNaiveBayes(_SklearnPipelineModel):
    kind = "tfidf_nb"

    def _estimator(self):
        return ComplementNB(alpha=self._params.get("alpha", 0.5))

    def _linear_coef(self) -> np.ndarray | None:
        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf, "feature_log_prob_") and clf.feature_log_prob_.shape[0] == 2:
            return clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        return None
