"""Disaster tweet classification package.

Layout:
- ``data``        load, validate, deduplicate and split the raw CSVs
- ``preprocess``  text normalisation shared by every model
- ``features``    TF-IDF feature builders
- ``models``      model registry: sklearn baselines, sentence embeddings, fine-tuned transformer
- ``evaluate``    metrics, calibration, error analysis, figures
- ``train``       CLI: cross-validate, fit, evaluate on the held-out split, save a run
- ``predict``     CLI and library entry point for scoring new text
- ``compare``     CLI: aggregate runs into a results table
- ``api``         FastAPI service exposing the saved model
"""

__version__ = "1.0.0"
