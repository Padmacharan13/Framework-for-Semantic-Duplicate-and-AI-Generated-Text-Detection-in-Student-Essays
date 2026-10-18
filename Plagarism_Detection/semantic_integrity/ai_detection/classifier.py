"""
semantic_integrity/ai_detection/classifier.py

Transformer-based classifier for AI-generated text detection (RoBERTa / DistilRoBERTa).
Provides training routines, checkpoints saving/loading, and prediction APIs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from semantic_integrity.config import (
    AI_DETECTOR_DIR,
    AI_DETECTOR_TRANSFORMER_NAME,
    AI_DETECTOR_TRANSFORMER_PATH,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)


class EssayDataset(Dataset):
    """PyTorch Dataset for essay texts and labels."""

    def __init__(self, texts: List[str], labels: Optional[List[int]] = None):
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {"text": str(self.texts[idx])}
        if self.labels is not None:
            item["label"] = int(self.labels[idx])
        return item


class TransformerAIDetector:
    """
    Sequence classification model for detecting AI-generated text using
    RoBERTa / DistilRoBERTa.
    """

    def __init__(
        self,
        model_name_or_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        max_length: int = 512,
        num_labels: int = 2,
    ):
        if model_name_or_path is None:
            if Path(AI_DETECTOR_TRANSFORMER_PATH).exists():
                self.model_name_or_path = str(AI_DETECTOR_TRANSFORMER_PATH)
            else:
                self.model_name_or_path = AI_DETECTOR_TRANSFORMER_NAME
        else:
            self.model_name_or_path = str(model_name_or_path)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.max_length = max_length
        self.num_labels = num_labels

        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForSequenceClassification] = None

    def _load_model(self) -> None:
        """Lazy load tokenizer and model."""
        if self._model is None or self._tokenizer is None:
            logger.info(
                "Loading Transformer AI Detector from '%s' on %s",
                self.model_name_or_path,
                self.device,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path,
                num_labels=self.num_labels,
                id2label={0: "Human", 1: "AI"},
                label2id={"Human": 0, "AI": 1},
                ignore_mismatched_sizes=True,
            )
            self._model.to(self.device)
            self._model.eval()

    @property
    def tokenizer(self) -> AutoTokenizer:
        self._load_model()
        assert self._tokenizer is not None
        return self._tokenizer

    @property
    def model(self) -> AutoModelForSequenceClassification:
        self._load_model()
        assert self._model is not None
        return self._model

    def predict_proba(self, texts: Union[str, List[str]], batch_size: int = 16) -> np.ndarray:
        """
        Compute predicted probabilities for given text(s).
        Returns array of shape (N, 2), where column 0 is P(Human) and column 1 is P(AI).
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty((0, 2), dtype=np.float32)

        self._load_model()
        self.model.eval()

        all_probs: List[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                input_ids = encoded.input_ids.to(self.device)
                attention_mask = encoded.attention_mask.to(self.device)

                logits = self.model(input_ids, attention_mask=attention_mask).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)

    def predict(self, texts: Union[str, List[str]], threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary labels (0 = Human, 1 = AI).
        """
        probs = self.predict_proba(texts)[:, 1]
        return (probs >= threshold).astype(int)

    def save(self, output_dir: Optional[Union[str, Path]] = None) -> Path:
        """Save model and tokenizer to directory."""
        save_path = Path(output_dir or AI_DETECTOR_TRANSFORMER_PATH)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info("Saved Transformer AI Detector to %s", save_path)
        return save_path

    def train_model(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 2e-5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Fine-tune sequence classification model.
        """
        self._load_model()
        self.model.train()

        torch.manual_seed(RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(RANDOM_SEED)

        train_dataset = EssayDataset(train_texts, train_labels)

        def collate_fn(batch):
            texts = [b["text"] for b in batch]
            labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
            enc = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            return enc.input_ids, enc.attention_mask, labels

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        criterion = nn.CrossEntropyLoss()

        logger.info(
            "Starting fine-tuning: %d epochs, %d batches/epoch, total steps=%d",
            epochs,
            len(train_loader),
            total_steps,
        )

        history = {"loss": [], "val_accuracy": []}

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0

            for step, (input_ids, attention_mask, labels) in enumerate(train_loader, start=1):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                running_loss += loss.item()

            avg_loss = running_loss / max(len(train_loader), 1)
            history["loss"].append(avg_loss)

            val_acc = 0.0
            if val_texts is not None and val_labels is not None:
                preds = self.predict(val_texts)
                val_acc = float(np.mean(preds == np.array(val_labels)))
                history["val_accuracy"].append(val_acc)
                logger.info(
                    "Epoch %d/%d - Loss: %.4f - Val Accuracy: %.4f",
                    epoch,
                    epochs,
                    avg_loss,
                    val_acc,
                )
            else:
                logger.info("Epoch %d/%d - Loss: %.4f", epoch, epochs, avg_loss)

        save_path = self.save(output_dir)
        return {"save_path": str(save_path), "history": history}
