"""Metrics, calibration and error analysis.

Two rules keep the numbers honest:

1. The decision threshold is chosen on out-of-fold predictions from the
   training portion, never on the held-out split.
2. Every metric on the held-out split is computed once, from one set of
   probabilities, and written to disk alongside the config that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Standard ECE with equal-width bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximising F1 on (y, p). Call this on out-of-fold predictions only."""
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


@dataclass(frozen=True)
class Metrics:
    threshold: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    roc_auc: float
    pr_auc: float
    brier: float
    ece: float
    tn: int
    fp: int
    fn: int
    tp: int

    def as_dict(self) -> dict[str, float | int]:
        return self.__dict__.copy()


def compute_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> Metrics:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return Metrics(
        threshold=float(threshold),
        f1=float(f1_score(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred)),
        accuracy=float((pred == y).mean()),
        roc_auc=float(roc_auc_score(y, p)),
        pr_auc=float(average_precision_score(y, p)),
        brier=float(brier_score_loss(y, p)),
        ece=expected_calibration_error(y, p),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def error_table(df: pd.DataFrame, p: np.ndarray, threshold: float) -> pd.DataFrame:
    """Misclassified rows, most confident mistakes first."""
    cols = ["id", "keyword", "text", "target"] if "keyword" in df else ["id", "text", "target"]
    out = df[cols].copy()
    out["p_disaster"] = np.round(p, 4)
    out["pred"] = (p >= threshold).astype(int)
    out = out[out["pred"] != out["target"]].copy()
    out["error_type"] = np.where(out["target"] == 1, "false_negative", "false_positive")
    out["confidence_of_error"] = np.where(out["target"] == 1, 1 - out["p_disaster"], out["p_disaster"])
    return out.sort_values("confidence_of_error", ascending=False).reset_index(drop=True)


# --- figures ----------------------------------------------------------------


def plot_reliability(y: np.ndarray, p: np.ndarray, path: Path, title: str, n_bins: int = 10) -> None:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            xs.append(p[m].mean())
            ys.append(y[m].mean())
            ns.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9AA6A9", linewidth=1)
    ax.plot(xs, ys, marker="o", color="#0E6B6B")
    for x, yv, n in zip(xs, ys, ns, strict=True):
        ax.annotate(str(n), (x, yv), textcoords="offset points", xytext=(4, -10), fontsize=7, color="#5B666C")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed disaster rate")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pr_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for name, (y, p) in curves.items():
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, label=f"{name} (AP {average_precision_score(y, p):.3f})", linewidth=1.4)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall on the held-out split", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, 1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(m: Metrics, path: Path, title: str) -> None:
    mat = np.array([[m.tn, m.fp], [m.fn, m.tp]])
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            colour = "white" if mat[i, j] > mat.max() / 2 else "black"
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=12, color=colour)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred: not disaster", "pred: disaster"], fontsize=8)
    ax.set_yticklabels(["true: not disaster", "true: disaster"], fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_top_features(top: dict[str, list[tuple[str, float]]], path: Path, n: int = 20) -> None:
    if not top:
        return
    pos = top["disaster"][:n][::-1]
    neg = top["not_disaster"][:n][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 5.2))
    axes[0].barh([t for t, _ in pos], [w for _, w in pos], color="#B2413A")
    axes[0].set_title("Pushes towards disaster", fontsize=10)
    axes[1].barh([t for t, _ in neg], [w for _, w in neg], color="#2B7A4B")
    axes[1].set_title("Pushes towards not-disaster", fontsize=10)
    for ax in axes:
        ax.tick_params(labelsize=7)
        ax.set_xlabel("coefficient", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
