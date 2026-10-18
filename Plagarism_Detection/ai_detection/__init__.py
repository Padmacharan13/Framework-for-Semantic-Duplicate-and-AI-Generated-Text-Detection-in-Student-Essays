"""
Top-level shim for semantic_integrity.ai_detection
Allows both `import ai_detection` and `from semantic_integrity import ai_detection`.
"""

import sys
from semantic_integrity.ai_detection import (
    LMFeatureExtractor,
    TransformerAIDetector,
    EnsembleAIDetector,
    get_detector,
    predict,
)

__all__ = [
    "LMFeatureExtractor",
    "TransformerAIDetector",
    "EnsembleAIDetector",
    "get_detector",
    "predict",
]
