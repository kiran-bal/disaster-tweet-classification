"""Fine-tuned transformer (DistilBERT by default).

A plain PyTorch loop rather than the HF Trainer: fewer moving parts, works the
same on CUDA, Apple MPS and CPU, and makes the training procedure fully
visible in ~80 lines.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .base import TextClassifier

logger = logging.getLogger(__name__)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FineTunedTransformer(TextClassifier):
    kind = "distilbert"

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        epochs: int = 3,
        lr: float = 3e-5,
        batch_size: int = 32,
        max_length: int = 64,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.06,
        seed: int = 42,
        **_: Any,
    ) -> None:
        self.model_name = model_name
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.max_length = max_length
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.seed = seed
        self.tokenizer = None
        self.model = None

    def params(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name, "epochs": self.epochs, "lr": self.lr,
            "batch_size": self.batch_size, "max_length": self.max_length,
            "weight_decay": self.weight_decay, "warmup_ratio": self.warmup_ratio, "seed": self.seed,
        }

    def _tensors(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(list(texts), padding=True, truncation=True, max_length=self.max_length,
                             return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

    def fit(self, texts: list[str], y: np.ndarray) -> FineTunedTransformer:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = _device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        self.model.to(device)

        ids, mask = self._tensors(texts)
        labels = torch.tensor(np.asarray(y), dtype=torch.long)
        loader = DataLoader(TensorDataset(ids, mask, labels), batch_size=self.batch_size, shuffle=True)

        no_decay = ("bias", "LayerNorm.weight")
        grouped = [
            {
                "params": [
                    p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimiser = torch.optim.AdamW(grouped, lr=self.lr)
        total_steps = len(loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimiser, int(total_steps * self.warmup_ratio), total_steps
        )

        self.model.train()
        for epoch in range(self.epochs):
            running = 0.0
            for b_ids, b_mask, b_y in loader:
                b_ids, b_mask, b_y = b_ids.to(device), b_mask.to(device), b_y.to(device)
                out = self.model(input_ids=b_ids, attention_mask=b_mask, labels=b_y)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimiser.step()
                scheduler.step()
                optimiser.zero_grad()
                running += out.loss.item()
            logger.info("epoch %d/%d  mean loss %.4f", epoch + 1, self.epochs, running / len(loader))
        self.model.eval()
        return self

    @torch.no_grad()
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        device = _device()
        self.model.to(device).eval()
        ids, mask = self._tensors(texts)
        loader = DataLoader(TensorDataset(ids, mask), batch_size=128)
        probs: list[np.ndarray] = []
        for b_ids, b_mask in loader:
            logits = self.model(input_ids=b_ids.to(device), attention_mask=b_mask.to(device)).logits
            probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        return np.concatenate(probs)

    def save(self, path: Path) -> None:
        self._write_meta(path)
        self.model.save_pretrained(Path(path) / "hf")
        self.tokenizer.save_pretrained(Path(path) / "hf")

    @classmethod
    def load(cls, path: Path) -> FineTunedTransformer:
        meta = json.loads((Path(path) / "model.json").read_text())
        obj = cls(**meta["params"])
        obj.tokenizer = AutoTokenizer.from_pretrained(Path(path) / "hf")
        obj.model = AutoModelForSequenceClassification.from_pretrained(Path(path) / "hf")
        obj.model.eval()
        return obj
