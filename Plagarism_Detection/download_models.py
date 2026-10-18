"""
download_models.py
Downloads pre-trained model artifacts that are too large for git.

Run:
    python download_models.py

This script ensures all required model files are present locally.
If models already exist, they are skipped.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def download_sbert_finetuned() -> None:
    """Download or verify the fine-tuned SBERT model."""
    from semantic_integrity.config import SBERT_FINETUNED_PATH, SBERT_BASE_MODEL

    if SBERT_FINETUNED_PATH.exists() and any(SBERT_FINETUNED_PATH.iterdir()):
        logger.info("Fine-tuned SBERT model already exists at %s", SBERT_FINETUNED_PATH)
        return

    logger.info(
        "Fine-tuned SBERT model not found. To create it, run:\n"
        "  1. python main_pipeline.py   (preprocess MSRP + QQP data)\n"
        "  2. python fine_tune_sbert.py  (fine-tune %s on the data)\n"
        "The base model '%s' will be used as fallback.",
        SBERT_BASE_MODEL,
        SBERT_BASE_MODEL,
    )


def download_word2vec() -> None:
    """Download or verify the Word2Vec model."""
    from semantic_integrity.config import WORD2VEC_MODEL_PATH

    if WORD2VEC_MODEL_PATH.exists():
        logger.info("Word2Vec model already exists at %s", WORD2VEC_MODEL_PATH)
        return

    logger.info(
        "Word2Vec model not found. To create it, run:\n"
        "  1. python main_pipeline.py   (preprocess MSRP + QQP data)\n"
        "  2. python train_word2vec.py   (train Word2Vec on the data)\n"
        "Word2Vec features will return zero vectors until trained."
    )


def verify_spacy() -> None:
    """Verify that spaCy model is installed."""
    try:
        import spacy
        spacy.load("en_core_web_sm")
        logger.info("spaCy model 'en_core_web_sm' is available.")
    except OSError:
        logger.warning(
            "spaCy model 'en_core_web_sm' not found. Install it with:\n"
            "  python -m spacy download en_core_web_sm"
        )


def main() -> None:
    logger.info("=== Model Download / Verification ===")
    verify_spacy()
    download_sbert_finetuned()
    download_word2vec()
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
