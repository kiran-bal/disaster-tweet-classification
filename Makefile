.PHONY: setup lint test train train-all compare serve docker

PY ?= .venv/bin/python

setup:              ## create a virtualenv with every extra
	uv venv -p 3.12 .venv
	uv pip install -p $(PY) -e ".[transformer,api,dev]"

lint:
	.venv/bin/ruff check src tests

test:
	.venv/bin/pytest -q

train:              ## fastest baseline, ~20 s
	.venv/bin/dt-train --config configs/tfidf_lr.yaml

train-all:          ## every config in configs/, then the comparison table
	for c in configs/*.yaml; do .venv/bin/dt-train --config $$c; done
	.venv/bin/dt-compare

compare:
	.venv/bin/dt-compare

serve:              ## API on :8000 using the tfidf_lr run
	DT_RUN_DIR=runs/tfidf_lr .venv/bin/uvicorn disaster_tweets.api:app --port 8000

docker:
	docker build -t disaster-tweets .
	docker run --rm -p 8000:8000 disaster-tweets
