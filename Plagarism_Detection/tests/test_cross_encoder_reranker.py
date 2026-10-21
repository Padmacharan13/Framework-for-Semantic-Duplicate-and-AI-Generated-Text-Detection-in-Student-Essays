"""Regression tests for Cross-Encoder output normalization."""
import numpy as np

from semantic_integrity.cross_encoder_reranker import CrossEncoderReranker


class FakeRankingModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        return np.asarray(next(self.outputs), dtype=np.float32)


def test_ms_marco_score_is_independent_of_other_batch_candidates():
    # Same raw logit (0.5), first by itself and then beside a larger logit.
    # The model score for the first pair must not depend on batch composition.
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker.available = True
    reranker.model = FakeRankingModel([[0.5], [0.5, 2.0]])

    alone = reranker.predict_pairs([("query", "candidate")])[0]
    batched = reranker.predict_pairs([
        ("query", "candidate"),
        ("query", "another candidate"),
    ])[0]

    expected = 1.0 / (1.0 + np.exp(-0.5))
    assert np.isclose(alone, expected, atol=1e-6)
    assert np.isclose(batched, expected, atol=1e-6)
    assert np.isclose(alone, batched, atol=1e-6)


def test_empty_pair_list_returns_empty_score_array():
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker.available = True
    reranker.model = None

    scores = reranker.predict_pairs([])

    assert scores.dtype == np.float32
    assert scores.size == 0
