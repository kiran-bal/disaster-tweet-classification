"""Train, cross-validate and evaluate one run.

    dt-train --config configs/tfidf_lr.yaml

Writes ``runs/<name>/``:
    config.yaml       the exact configuration used
    metrics.json      dataset summary, CV summary, held-out metrics
    model/            the saved model (loadable with ``models.load``)
    errors.csv        misclassified held-out rows, most confident first
    oof.csv           out-of-fold probabilities on the training portion
and figures under ``reports/figures/<name>_*.png``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

from . import data, models
from .config import FIGURES_DIR, RUNS_DIR, RunConfig
from .evaluate import (
    best_f1_threshold,
    compute_metrics,
    error_table,
    plot_confusion,
    plot_reliability,
    plot_top_features,
)

logger = logging.getLogger("disaster_tweets.train")


def cross_validate(cfg: RunConfig, texts: list[str], y: np.ndarray) -> tuple[np.ndarray, dict]:
    """Out-of-fold probabilities and per-fold F1 at 0.5."""
    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    oof = np.zeros(len(y), dtype=float)
    fold_f1: list[float] = []
    texts_arr = np.asarray(texts, dtype=object)
    for fold, (tr, va) in enumerate(skf.split(texts_arr, y), 1):
        t0 = time.perf_counter()
        model = models.build(cfg.model, **cfg.params)
        model.fit(list(texts_arr[tr]), y[tr])
        oof[va] = model.predict_proba(list(texts_arr[va]))
        m = compute_metrics(y[va], oof[va], 0.5)
        fold_f1.append(m.f1)
        logger.info("fold %d/%d  F1@0.5=%.4f  (%.1fs)", fold, cfg.cv_folds, m.f1, time.perf_counter() - t0)
    return oof, {
        "fold_f1_at_0.5": fold_f1,
        "mean_f1_at_0.5": float(np.mean(fold_f1)),
        "std_f1_at_0.5": float(np.std(fold_f1)),
    }


def run(cfg: RunConfig, *, skip_cv: bool = False) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    np.random.seed(cfg.seed)
    run_dir = RUNS_DIR / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))

    raw = data.load_raw()
    clean, summary = data.deduplicate(raw)
    train_df, test_df = data.split(clean, test_size=cfg.test_size, seed=cfg.seed)
    logger.info(
        "rows: raw=%d clean=%d train=%d held-out=%d positive_rate=%.3f",
        summary.rows_raw, summary.rows_after_dedupe, len(train_df), len(test_df), summary.positive_rate,
    )

    x_train = data.build_input_text(train_df, use_keyword=cfg.use_keyword, **cfg.preprocess).tolist()
    x_test = data.build_input_text(test_df, use_keyword=cfg.use_keyword, **cfg.preprocess).tolist()
    y_train = train_df["target"].to_numpy()
    y_test = test_df["target"].to_numpy()

    # 1. Cross-validation on the training portion: model selection + threshold.
    if skip_cv:
        cv_summary, threshold = {"skipped": True}, 0.5
    else:
        oof, cv_summary = cross_validate(cfg, x_train, y_train)
        threshold = best_f1_threshold(y_train, oof)
        cv_summary["oof_metrics_at_0.5"] = compute_metrics(y_train, oof, 0.5).as_dict()
        cv_summary["oof_metrics_at_tuned"] = compute_metrics(y_train, oof, threshold).as_dict()
        oof_df = pd.DataFrame({"id": train_df["id"], "target": y_train, "oof_p": oof})
        oof_df.to_csv(run_dir / "oof.csv", index=False)

    # 2. Fit on the full training portion, evaluate once on the held-out split.
    t0 = time.perf_counter()
    model = models.build(cfg.model, **cfg.params).fit(x_train, y_train)
    fit_seconds = time.perf_counter() - t0
    p_test = model.predict_proba(x_test)
    heldout_default = compute_metrics(y_test, p_test, 0.5)
    heldout_tuned = compute_metrics(y_test, p_test, threshold)
    logger.info(
        "held-out  F1@0.5=%.4f  F1@%.3f=%.4f  ROC-AUC=%.4f  PR-AUC=%.4f  Brier=%.4f  ECE=%.4f",
        heldout_default.f1, threshold, heldout_tuned.f1, heldout_tuned.roc_auc,
        heldout_tuned.pr_auc, heldout_tuned.brier, heldout_tuned.ece,
    )

    # 3. Artefacts.
    model.save(run_dir / "model")
    error_table(test_df, p_test, threshold).to_csv(run_dir / "errors.csv", index=False)
    heldout = pd.DataFrame({"id": test_df["id"], "target": y_test, "p": p_test})
    heldout.to_csv(run_dir / "heldout_predictions.csv", index=False)
    plot_reliability(
        y_test, p_test, FIGURES_DIR / f"{cfg.name}_reliability.png", f"{cfg.name}: reliability (held-out)"
    )
    plot_confusion(
        heldout_tuned,
        FIGURES_DIR / f"{cfg.name}_confusion.png",
        f"{cfg.name}: confusion at t={threshold:.2f}",
    )
    if hasattr(model, "top_features"):
        top = model.top_features(25)
        if top:
            plot_top_features(top, FIGURES_DIR / f"{cfg.name}_top_features.png")
            (run_dir / "top_features.json").write_text(json.dumps(top, indent=2))

    result = {
        "name": cfg.name,
        "model": cfg.model,
        "params": cfg.params,
        "dataset": summary.as_dict(),
        "n_train": int(len(train_df)),
        "n_heldout": int(len(test_df)),
        "cv": cv_summary,
        "threshold": threshold,
        "heldout_at_0.5": heldout_default.as_dict(),
        "heldout_at_tuned_threshold": heldout_tuned.as_dict(),
        "fit_seconds": round(fit_seconds, 2),
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="path to a run YAML")
    ap.add_argument(
        "--skip-cv", action="store_true", help="skip cross-validation (threshold stays 0.5)"
    )
    args = ap.parse_args(argv)
    run(RunConfig.from_yaml(Path(args.config)), skip_cv=args.skip_cv)


if __name__ == "__main__":
    main()
