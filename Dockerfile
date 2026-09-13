# Serves the TF-IDF + logistic-regression model behind FastAPI.
# The model is trained during the build (about 20 s) so the image is
# self-contained and reproducible from the committed data and config.
FROM python:3.12-slim AS base

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[api]"

COPY data/raw ./data/raw
COPY configs ./configs
RUN dt-train --config configs/tfidf_lr.yaml --skip-cv

ENV DT_RUN_DIR=/app/runs/tfidf_lr
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "disaster_tweets.api:app", "--host", "0.0.0.0", "--port", "8000"]
