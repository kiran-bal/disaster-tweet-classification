"""FastAPI inference service.

    DT_RUN_DIR=runs/tfidf_lr uvicorn disaster_tweets.api:app --port 8000

    curl -X POST localhost:8000/predict -H 'content-type: application/json' \
         -d '{"texts": ["Forest fire near La Ronge Sask. Canada"], "explain": true}'
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .predict import Predictor

MAX_BATCH = 256


class PredictRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_BATCH)
    keywords: list[str | None] | None = None
    explain: bool = False


class Prediction(BaseModel):
    text: str
    p_disaster: float
    label: int
    threshold: float
    top_contributions: list[tuple[str, float]] | None = None


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    model: str
    run: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    run_dir = Path(os.environ.get("DT_RUN_DIR", "runs/tfidf_lr"))
    if not (run_dir / "model").exists():
        raise RuntimeError(
            f"no saved model at {run_dir}. Train one with: dt-train --config configs/tfidf_lr.yaml"
        )
    app.state.predictor = Predictor(run_dir)
    yield


app = FastAPI(title="Disaster tweet classifier", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    p: Predictor = app.state.predictor
    return {"status": "ok", "model": p.config.model, "run": p.config.name, "threshold": p.threshold}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    p: Predictor = app.state.predictor
    if req.keywords is not None and len(req.keywords) != len(req.texts):
        raise HTTPException(status_code=422, detail="keywords must match texts in length")
    rows = p.predict(req.texts, req.keywords, explain=req.explain)
    return PredictResponse(
        predictions=[Prediction(**r) for r in rows], model=p.config.model, run=p.config.name
    )
