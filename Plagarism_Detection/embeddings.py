"""
embeddings.py
Thin wrapper around sentence-transformers for batched encoding. Load the
model ONCE (it's expensive) and reuse the same Embedder instance everywhere.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from config import SBERT_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str = SBERT_MODEL) -> None:
        logger.info("Loading SBERT model '%s'...", model_name)
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Iterable[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        """
        Encode a list of texts into L2-normalized embeddings (so a plain dot
        product equals cosine similarity — cheaper than recomputing norms
        every time you compare vectors downstream).
        """
        texts = list(texts)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings


if __name__ == "__main__":
    embedder = Embedder()
    vecs = embedder.encode(["The automobile is rapid.", "The vehicle moves fast."])
    print("Shape:", vecs.shape)
    print("Cosine similarity (dot product of normalized vecs):", float(np.dot(vecs[0], vecs[1])))