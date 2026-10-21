import numpy as np
import pytest

from semantic_integrity.similarity_engine import (
    PassageMatch,
    align_passages,
    document_level_summary,
)


class LookupEmbedder:
    """Small deterministic embedder for alignment unit tests."""

    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts):
        return np.asarray([self.vectors[text] for text in texts], dtype=float)


def test_alignment_does_not_reuse_document_b_sentence():
    embedder = LookupEmbedder({
        "a1": [1.0, 0.0],
        "a2": [0.7, 0.7],
        "b1": [1.0, 0.0],
        "b2": [0.0, 1.0],
    })

    matches = align_passages(
        embedder, ["a1", "a2"], ["b1", "b2"], match_threshold=0.5
    )

    assert len(matches) == 2
    assert len({m.sentence_a for m in matches}) == len(matches)
    assert len({m.sentence_b for m in matches}) == len(matches)


def test_document_score_is_symmetric_when_documents_are_swapped():
    match = PassageMatch(
        sentence_a="same sentence",
        sentence_b="same sentence",
        semantic_similarity=0.9,
        lexical_similarity=1.0,
        tier="verbatim",
    )

    forward = document_level_summary([match], total_sentences_a=2, total_sentences_b=4)
    reverse = document_level_summary([match], total_sentences_a=4, total_sentences_b=2)

    assert forward["overall_score"] == reverse["overall_score"]
    assert forward["matched_sentence_ratio"] == reverse["matched_sentence_ratio_b"]


def test_document_summary_handles_empty_documents():
    summary = document_level_summary([], total_sentences_a=0, total_sentences_b=3)
    assert summary["overall_score"] == 0.0
    assert summary["flagged"] is False
    assert summary["matched_sentence_ratio"] == 0.0


def test_alignment_rejects_out_of_range_threshold():
    embedder = LookupEmbedder({"a": [1.0, 0.0], "b": [1.0, 0.0]})
    with pytest.raises(ValueError):
        align_passages(embedder, ["a"], ["b"], match_threshold=1.1)
