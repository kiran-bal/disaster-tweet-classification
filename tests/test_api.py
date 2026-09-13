import json

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from disaster_tweets import models
from disaster_tweets.config import RunConfig


@pytest.fixture
def run_dir(tmp_path, texts_and_labels):
    texts, y, _ = texts_and_labels
    cfg = RunConfig(name="toy", model="tfidf_lr", params={"min_df": 1})
    model = models.build(cfg.model, **cfg.params).fit(texts, np.asarray(y))
    d = tmp_path / "run"
    d.mkdir()
    model.save(d / "model")
    (d / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict()))
    (d / "metrics.json").write_text(json.dumps({"threshold": 0.5}))
    return d


def test_predict_endpoint(run_dir, monkeypatch):
    monkeypatch.setenv("DT_RUN_DIR", str(run_dir))
    from disaster_tweets.api import app

    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        payload = {"texts": ["Forest fire near La Ronge", "love this album"], "explain": True}
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        preds = r.json()["predictions"]
        assert preds[0]["label"] == 1 and preds[1]["label"] == 0
        assert preds[0]["top_contributions"]


def test_predict_rejects_bad_keyword_length(run_dir, monkeypatch):
    monkeypatch.setenv("DT_RUN_DIR", str(run_dir))
    from disaster_tweets.api import app

    with TestClient(app) as client:
        r = client.post("/predict", json={"texts": ["a"], "keywords": ["x", "y"]})
        assert r.status_code == 422
