"""TF-IDF feature builders shared by the linear baselines."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def tfidf_word_char(
    *,
    word_ngram_max: int = 2,
    char_ngram: tuple[int, int] = (2, 5),
    min_df: int = 2,
    max_features_word: int | None = 50_000,
    max_features_char: int | None = 100_000,
    sublinear_tf: bool = True,
) -> FeatureUnion:
    """Word n-grams plus character n-grams.

    Character n-grams are what make a linear model robust to the spelling
    variety in tweets ("fireee", "#earthquakee"); word n-grams capture the
    phrases ("forest fire", "suicide bomber") that decide most examples.
    """
    word = TfidfVectorizer(
        ngram_range=(1, word_ngram_max),
        min_df=min_df,
        max_features=max_features_word,
        sublinear_tf=sublinear_tf,
        token_pattern=r"(?u)\b\w+\b",
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=char_ngram,
        min_df=min_df,
        max_features=max_features_char,
        sublinear_tf=sublinear_tf,
    )
    return FeatureUnion([("word", word), ("char", char)])
