"""Score new tweets with a saved run.

    dt-predict --run runs/tfidf_lr "Forest fire near La Ronge Sask. Canada"
    echo "some text" | dt-predict --run runs/tfidf_lr
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from . import models
from .config import RunConfig
from .preprocess import normalise


class Predictor:
    """Loads a run directory once and scores text with the same preprocessing it was trained with."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.config = RunConfig(**yaml.safe_load((self.run_dir / "config.yaml").read_text()))
        self.model = models.load(self.run_dir / "model")
        metrics = json.loads((self.run_dir / "metrics.json").read_text())
        self.threshold = float(metrics.get("threshold", 0.5))

    def prepare(self, text: str, keyword: str | None = None) -> str:
        cleaned = normalise(text, **self.config.preprocess)
        if self.config.use_keyword and keyword:
            cleaned = f"{normalise(keyword, **self.config.preprocess)} {cleaned}".strip()
        return cleaned

    def predict(
        self, texts: list[str], keywords: list[str | None] | None = None, *, explain: bool = False
    ) -> list[dict]:
        keywords = keywords or [None] * len(texts)
        prepared = [self.prepare(t, k) for t, k in zip(texts, keywords, strict=True)]
        probs = self.model.predict_proba(prepared)
        out = []
        for text, prepared_text, p in zip(texts, prepared, probs, strict=True):
            row = {
                "text": text,
                "p_disaster": round(float(p), 4),
                "label": int(p >= self.threshold),
                "threshold": self.threshold,
            }
            if explain:
                row["top_contributions"] = self.model.explain(prepared_text)
            out.append(row)
        return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory, e.g. runs/tfidf_lr")
    ap.add_argument("--explain", action="store_true", help="include top n-gram contributions when available")
    ap.add_argument("text", nargs="*", help="one or more tweets; reads stdin lines if omitted")
    args = ap.parse_args(argv)
    texts = args.text or [line.rstrip("\n") for line in sys.stdin if line.strip()]
    predictor = Predictor(args.run)
    for row in predictor.predict(texts, explain=args.explain):
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
