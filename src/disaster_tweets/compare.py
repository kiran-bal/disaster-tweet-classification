"""Aggregate every run under ``runs/`` into ``reports/results.md`` and a PR-curve figure.

    dt-compare
"""

from __future__ import annotations

import json

import pandas as pd

from .config import FIGURES_DIR, REPORTS_DIR, RUNS_DIR
from .evaluate import plot_pr_curves

COLUMNS = ["f1", "precision", "recall", "accuracy", "roc_auc", "pr_auc", "brier", "ece"]


def main() -> None:
    rows, curves = [], {}
    for run_dir in sorted(RUNS_DIR.glob("*/metrics.json")):
        m = json.loads(run_dir.read_text())
        h = m["heldout_at_tuned_threshold"]
        cv = m.get("cv", {})
        rows.append({
            "run": m["name"], "model": m["model"],
            "cv_f1_mean": cv.get("mean_f1_at_0.5"), "cv_f1_std": cv.get("std_f1_at_0.5"),
            "threshold": m["threshold"], **{c: h[c] for c in COLUMNS},
            "f1_at_0.5": m["heldout_at_0.5"]["f1"], "fit_seconds": m["fit_seconds"],
        })
        pred_path = run_dir.parent / "heldout_predictions.csv"
        if pred_path.exists():
            df = pd.read_csv(pred_path)
            curves[m["name"]] = (df["target"].to_numpy(), df["p"].to_numpy())
    if not rows:
        print("no runs found under", RUNS_DIR)
        return
    table = pd.DataFrame(rows).sort_values("f1", ascending=False)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS_DIR / "results.csv", index=False)
    md = [
        "| run | CV F1 (5-fold, t=0.5) | t | F1 | Precision | Recall | ROC-AUC | PR-AUC "
        "| Brier | ECE | fit (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in table.itertuples():
        cv = f"{r.cv_f1_mean:.3f} ± {r.cv_f1_std:.3f}" if pd.notna(r.cv_f1_mean) else "—"
        md.append(
            f"| {r.run} | {cv} | {r.threshold:.2f} | **{r.f1:.3f}** | {r.precision:.3f} | {r.recall:.3f} | "
            f"{r.roc_auc:.3f} | {r.pr_auc:.3f} | {r.brier:.3f} | {r.ece:.3f} | {r.fit_seconds:.0f} |"
        )
    (REPORTS_DIR / "results.md").write_text("\n".join(md) + "\n")
    if curves:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plot_pr_curves(curves, FIGURES_DIR / "pr_curves.png")
    print("\n".join(md))


if __name__ == "__main__":
    main()
