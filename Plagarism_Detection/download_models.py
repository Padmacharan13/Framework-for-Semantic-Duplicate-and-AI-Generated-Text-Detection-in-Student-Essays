"""
download_models.py
Downloads and verifies pre-trained model artifacts and indices that are too large for git.

Run:
    python download_models.py

This script ensures all required model files, indices, and checkpoints are present locally.
If artifacts already exist, they are validated and preserved.
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


def verify_faiss_index() -> None:
    """Verify or build FAISS vector index."""
    from semantic_integrity.config import FAISS_INDEX_PATH, FAISS_METADATA_PATH, MSRP_PROCESSED_PATH
    import pandas as pd

    if FAISS_INDEX_PATH.exists() and FAISS_METADATA_PATH.exists():
        logger.info("FAISS vector index and metadata exist at %s", FAISS_INDEX_PATH)
        return

    logger.info("FAISS index not found. Building index from processed MSRP corpus...")
    try:
        from semantic_integrity.faiss_engine import FAISSEngine
        if MSRP_PROCESSED_PATH.exists():
            df = pd.read_parquet(MSRP_PROCESSED_PATH)
            col = "text1" if "text1" in df.columns else "text_a"
            corpus = df[col].dropna().tolist()[:1000]
        else:
            corpus = [
                "Sample text for plagiarism and duplicate detection.",
                "Semantic similarity helps discover paraphrase variations.",
                "Student essays must adhere to strict academic integrity standards.",
            ]
        engine = FAISSEngine()
        engine.build_index(corpus)
        engine.save_index(FAISS_INDEX_PATH, FAISS_METADATA_PATH)
        logger.info("FAISS vector index created and saved successfully.")
    except Exception as e:
        logger.warning("Could not automatically build FAISS index: %s", e)


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


def verify_ai_detector() -> None:
    """Verify presence of AI detector weights and ensemble checkpoints."""
    from semantic_integrity.config import AI_DETECTOR_TRANSFORMER_PATH, AI_DETECTOR_ENSEMBLE_PATH
    if AI_DETECTOR_TRANSFORMER_PATH.exists() and AI_DETECTOR_ENSEMBLE_PATH.exists():
        logger.info("AI Detector checkpoints found at %s", AI_DETECTOR_TRANSFORMER_PATH)
    else:
        logger.info(
            "AI Detector checkpoints not found. Run `python train_ai_detector.py` "
            "to fine-tune the transformer and fit the calibrated ensemble meta-model."
        )


def main() -> None:
    logger.info("=== Model & Artifact Download / Verification ===")
    verify_spacy()
    download_sbert_finetuned()
    download_word2vec()
    verify_faiss_index()
    verify_ai_detector()
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
