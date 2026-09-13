"""Loading, validation, de-duplication and splitting of the Kaggle data.

The train file has 7,613 rows. It also has duplicate texts, some of which are
labelled *differently* in different rows. Left alone those duplicates leak
across a random split and inflate every metric, and the conflicting labels
put a hard ceiling on achievable accuracy. This module makes both facts
explicit instead of hiding them.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RAW_TRAIN
from .preprocess import dedupe_key, normalise

REQUIRED_COLUMNS = ("id", "text", "target")


class DataValidationError(ValueError):
    """Raised when the raw file does not look like the dataset we expect."""


@dataclass(frozen=True)
class DatasetSummary:
    rows_raw: int
    rows_after_dedupe: int
    exact_duplicates_dropped: int
    conflicting_texts_dropped: int
    positive_rate: float

    def as_dict(self) -> dict[str, float | int]:
        return self.__dict__.copy()


def load_raw(path=RAW_TRAIN) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing columns: {missing}")
    if df["text"].isna().any():
        raise DataValidationError("text column contains nulls")
    labels = set(df["target"].dropna().unique().tolist())
    if not labels <= {0, 1}:
        raise DataValidationError(f"target must be binary 0/1, found {sorted(labels)}")
    if len(df) == 0:
        raise DataValidationError("empty dataset")


def build_input_text(df: pd.DataFrame, *, use_keyword: bool, **preprocess_opts) -> pd.Series:
    """The string the model actually sees: optional keyword, then cleaned text."""
    text = df["text"].map(lambda t: normalise(t, **preprocess_opts))
    if use_keyword and "keyword" in df.columns:
        kw = df["keyword"].fillna("").str.replace("%20", " ", regex=False).str.strip()
        kw = kw.map(lambda k: normalise(k, **preprocess_opts))
        text = (kw + " " + text).str.strip()
    return text


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, DatasetSummary]:
    """Drop exact duplicates and texts that carry both labels.

    Returns the cleaned frame plus a summary that goes into every run's
    metrics file, so the reported numbers are traceable to a known row count.
    """
    work = df.copy()
    work["_key"] = work["text"].map(dedupe_key)
    rows_raw = len(work)

    before = len(work)
    work = work.drop_duplicates(subset=["_key", "target"], keep="first")
    exact_dropped = before - len(work)

    label_counts = work.groupby("_key")["target"].nunique()
    conflicting_keys = set(label_counts[label_counts > 1].index)
    before = len(work)
    work = work[~work["_key"].isin(conflicting_keys)]
    conflicting_dropped = before - len(work)

    work = work.drop(columns="_key").reset_index(drop=True)
    summary = DatasetSummary(
        rows_raw=rows_raw,
        rows_after_dedupe=len(work),
        exact_duplicates_dropped=exact_dropped,
        conflicting_texts_dropped=conflicting_dropped,
        positive_rate=float(work["target"].mean()),
    )
    return work, summary


def split(
    df: pd.DataFrame, *, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train / held-out split. The held-out part is touched once."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["target"]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
