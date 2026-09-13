import pandas as pd
import pytest

from disaster_tweets import data


def test_validate_rejects_missing_columns():
    with pytest.raises(data.DataValidationError):
        data.validate(pd.DataFrame({"id": [1], "text": ["x"]}))


def test_validate_rejects_non_binary_target():
    with pytest.raises(data.DataValidationError):
        data.validate(pd.DataFrame({"id": [1], "text": ["x"], "target": [2]}))


def test_deduplicate_drops_exact_and_conflicting(small_df):
    df = pd.concat([
        small_df,
        pd.DataFrame({"id": [900, 901], "keyword": [None, None],
                      "text": [small_df.text[0], small_df.text[0]], "target": [1, 0]}),
    ], ignore_index=True)
    clean, summary = data.deduplicate(df)
    assert summary.rows_raw == len(df)
    assert summary.exact_duplicates_dropped == 1      # the (text0, 1) duplicate
    assert summary.conflicting_texts_dropped == 2     # both remaining rows of text0
    assert small_df.text[0] not in clean.text.tolist()
    assert 0 < summary.positive_rate < 1


def test_split_is_stratified_and_disjoint(small_df):
    df = pd.concat([small_df] * 5, ignore_index=True)
    df["id"] = range(len(df))
    train, test = data.split(df, test_size=0.2, seed=1)
    assert len(train) + len(test) == len(df)
    assert set(train.id).isdisjoint(set(test.id))
    assert abs(train.target.mean() - test.target.mean()) < 0.15


def test_build_input_text_prepends_keyword(small_df):
    out = data.build_input_text(small_df, use_keyword=True)
    assert out.iloc[0].startswith("fire ")
    out2 = data.build_input_text(small_df, use_keyword=False)
    assert not out2.iloc[0].startswith("fire forest")
