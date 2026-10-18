"""
semantic_integrity.ai_detection package

Provides:
  - LMFeatureExtractor (perplexity, burstiness, TTR)
  - TransformerAIDetector (RoBERTa sequence classification)
  - EnsembleAIDetector (fused meta-classifier)
  - predict(text) convenience API
"""

from semantic_integrity.ai_detection.classifier import TransformerAIDetector
from semantic_integrity.ai_detection.ensemble import (
    EnsembleAIDetector,
    get_detector,
    predict,
)
from semantic_integrity.ai_detection.features import LMFeatureExtractor

__all__ = [
    "LMFeatureExtractor",
    "TransformerAIDetector",
    "EnsembleAIDetector",
    "get_detector",
    "predict",
]
