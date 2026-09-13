import numpy as np
import pytest

from disaster_tweets import models


@pytest.mark.parametrize("name", ["tfidf_lr", "tfidf_svm", "tfidf_nb"])
def test_sklearn_models_fit_predict_save_load(name, texts_and_labels, tmp_path):
    texts, y, _ = texts_and_labels
    model = models.build(name, min_df=1)
    model.fit(texts, y)
    p = model.predict_proba(texts)
    assert p.shape == (len(texts),)
    assert np.all((p >= 0) & (p <= 1))
    assert ((p >= 0.5).astype(int) == y).mean() > 0.9  # separable toy data

    model.save(tmp_path / "m")
    loaded = models.load(tmp_path / "m")
    assert loaded.kind == name
    np.testing.assert_allclose(loaded.predict_proba(texts), p, atol=1e-8)


def test_linear_explanations_name_real_ngrams(texts_and_labels):
    texts, y, _ = texts_and_labels
    model = models.build("tfidf_lr", min_df=1).fit(texts, y)
    top = model.top_features(5)
    assert set(top) == {"disaster", "not_disaster"}
    assert len(top["disaster"]) == 5
    contribs = model.explain("forest fire evacuation")
    assert contribs and all(isinstance(w, float) for _, w in contribs)


def test_unknown_model_name():
    with pytest.raises(KeyError):
        models.build("does_not_exist")
