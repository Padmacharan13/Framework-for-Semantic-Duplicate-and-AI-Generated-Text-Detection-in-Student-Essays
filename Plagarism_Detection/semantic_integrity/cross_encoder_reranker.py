"""
cross_encoder_reranker.py
Cross-Encoder Re-ranker for Secondary Precision Pass on Borderline Candidate Pairs.
Provides joint cross-attention scoring across text pairs to eliminate false positives from Bi-Encoders.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder

from .config import CROSS_ENCODER_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Cross-Encoder model wrapper for high-precision pair re-ranking.
    Jointly processes (Text A, Text B) through transformer cross-attention layers.
    """

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL) -> None:
        logger.info("Initializing Cross-Encoder model '%s'...", model_name)
        try:
            self.model_name = model_name
            self.model = CrossEncoder(
    model_name,
    tokenizer_args={"use_fast": False}
)
            self.available = True
            logger.info("Cross-Encoder model '%s' loaded successfully.", model_name)
        except Exception as e:
            logger.error("Failed to load Cross-Encoder model '%s': %s. Falling back to dummy re-ranker.", model_name, e)
            self.available = False
            self.model = None

    def _sigmoid(self, x: np.ndarray | float) -> np.ndarray | float:
        # Clip to avoid overflow in exp for large negative values
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))

    def predict_pairs(self, pairs: list[tuple[str, str]], batch_size: int = 32) -> np.ndarray:
        """
        Computes joint Cross-Encoder similarity scores for a list of (text_a, text_b) pairs.
        Returns normalized scores in [0.0, 1.0].
        """
        if not pairs:
            return np.array([], dtype=np.float32)

        if not self.available or self.model is None:
            raise RuntimeError("Cross-Encoder model is unavailable; cannot compute scores.")

        try:
            # Convert tuples to list of lists for compatibility with both v2 and v3
            pair_inputs = [list(p) for p in pairs]
            raw_scores = self.model.predict(pair_inputs, batch_size=batch_size, show_progress_bar=False)
            raw_scores = np.asarray(raw_scores, dtype=np.float32)

            # Flatten if multi-dimensional (some models return [N, num_labels])
            if raw_scores.ndim > 1:
                raw_scores = raw_scores[:, -1]  # Take last column (similarity/entailment)

            # If model outputs logits (e.g. MS-MARCO), convert via sigmoid to [0, 1]
            if raw_scores.ndim > 0 and (np.min(raw_scores) < -0.01 or np.max(raw_scores) > 1.01):
                norm_scores = self._sigmoid(raw_scores)
            else:
                norm_scores = np.clip(raw_scores, 0.0, 1.0)

            return norm_scores
        except Exception as e:
            logger.error("Error predicting Cross-Encoder scores: %s", e)
            raise RuntimeError(f"Cross-Encoder prediction failed: {e}")

    def predict_pair(self, text_a: str, text_b: str) -> float:
        scores = self.predict_pairs([(text_a, text_b)])
        return float(scores[0]) if len(scores) > 0 else 0.5

    def rerank_candidates(
        self,
        query_text: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Takes candidate documents (from FAISS Bi-Encoder retrieval), scores them with Cross-Encoder,
        and returns them re-ranked with updated ranks and precision scores.
        """
        if not candidates:
            return []

        pairs = [(query_text, cand.get("text", "")) for cand in candidates]
        cross_scores = self.predict_pairs(pairs)

        reranked = []
        for cand, ce_score in zip(candidates, cross_scores):
            ce_val = round(float(ce_score), 4)
            bi_val = float(cand.get("bi_encoder_score", 0.0))

            cand_copy = dict(cand)
            cand_copy["cross_encoder_score"] = ce_val
            cand_copy["score_diff"] = round(ce_val - bi_val, 4)
            # Two-stage combined confidence (0.6 * Cross-Encoder + 0.4 * Bi-Encoder)
            cand_copy["hybrid_confidence"] = round(0.6 * ce_val + 0.4 * bi_val, 4)
            reranked.append(cand_copy)

        # Sort descending by Cross-Encoder precision score
        reranked.sort(key=lambda x: x["cross_encoder_score"], reverse=True)

        # Update post-rerank ranks
        for idx, cand in enumerate(reranked, start=1):
            cand["reranked_rank"] = idx
            cand["rank_change"] = cand.get("rank", idx) - idx

        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked


if __name__ == "__main__":
    reranker = CrossEncoderReranker()
    q = "Renewable energy technologies like solar and wind are growing rapidly."
    t1 = "Solar panels and wind turbines are becoming vastly cheaper and replacing fossil fuels."
    t2 = "Quantum computing relies on qubits to execute complex algorithms."

    pairs = [(q, t1), (q, t2)]
    scores = reranker.predict_pairs(pairs)

    print(f"\nQuery: '{q}'")
    print(f"Pair 1 score (Renewable vs Solar/Wind): {scores[0]:.4f}")
    print(f"Pair 2 score (Renewable vs Quantum):    {scores[1]:.4f}")
