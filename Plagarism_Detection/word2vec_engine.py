"""
word2vec_engine.py
Dense Word Embedding Engine using Gensim Word2Vec for baseline model comparison.
Computes IDF-weighted average Word2Vec sentence embeddings with L2 normalization.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

from config import WORD2VEC_MODEL_FILE, WORD2VEC_VECTOR_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_RE_TOKEN = re.compile(r"\b\w+\b")


def _tokenize(text: str) -> list[str]:
    """Basic lowercase tokenization for Word2Vec."""
    if not isinstance(text, str):
        return []
    return _RE_TOKEN.findall(text.lower())


class Word2VecEmbedder:
    def __init__(
        self,
        vector_size: int = WORD2VEC_VECTOR_SIZE,
        model_path: Path | str = WORD2VEC_MODEL_FILE,
        train_corpus: Iterable[str] | None = None,
    ) -> None:
        self.vector_size = vector_size
        self.model_path = Path(model_path)
        self.tfidf_vectorizer: TfidfVectorizer | None = None
        self.idf_dict: dict[str, float] = {}

        if self.model_path.exists():
            logger.info("Loading pre-trained Word2Vec model from '%s'...", self.model_path)
            self.model = Word2Vec.load(str(self.model_path))
        else:
            logger.info("Training fresh Word2Vec model (vector_size=%d)...", vector_size)
            corpus_texts = list(train_corpus) if train_corpus else self._default_corpus()
            tokenized_corpus = [_tokenize(t) for t in corpus_texts if t.strip()]

            self.model = Word2Vec(
                sentences=tokenized_corpus,
                vector_size=self.vector_size,
                window=5,
                min_count=1,
                workers=4,
                epochs=20,
                sg=1,  # Skip-gram architecture for richer semantic embeddings
            )
            self.model.save(str(self.model_path))
            logger.info("Saved Word2Vec model -> %s", self.model_path)

        # Fit IDF weights for IDF-weighted vector averaging
        if train_corpus:
            self._fit_idf(train_corpus)

    @staticmethod
    def _default_corpus() -> list[str]:
        return [
            "Academic integrity relies heavily on authentic assignment submissions.",
            "Instructors depend on students submitting original work for courses.",
            "Digital submissions make manual comparison for duplicates difficult.",
            "Automated pipelines scan student texts and index conceptual content.",
            "Traditional string matching methods fail when students restructure essays.",
            "Synonym substitution and syntax swapping preserve the original meaning.",
            "Artificial intelligence is reshaping higher education and learning.",
            "Quantum computing leverages quantum mechanics principles.",
            "Photosynthesis converts light energy into chemical energy in plants.",
            "Microservices architecture decomposes monolithic applications into services.",
        ]

    def _fit_idf(self, texts: Iterable[str]) -> None:
        """Fit TF-IDF vectorizer to extract inverse document frequency (IDF) weights."""
        self.tfidf_vectorizer = TfidfVectorizer(max_features=50_000, lowercase=True)
        self.tfidf_vectorizer.fit(texts)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        idfs = self.tfidf_vectorizer.idf_
        self.idf_dict = dict(zip(feature_names, idfs))

    def fit_corpus(self, texts: Iterable[str]) -> None:
        """Update/fit Word2Vec vocabulary and IDF weights on a new corpus."""
        texts = list(texts)
        tokenized = [_tokenize(t) for t in texts if t.strip()]
        if tokenized:
            self.model.build_vocab(tokenized, update=True)
            self.model.train(tokenized, total_examples=len(tokenized), epochs=self.model.epochs)
            self.model.save(str(self.model_path))
            self._fit_idf(texts)

    def encode_sentence(self, text: str) -> np.ndarray:
        """
        Computes IDF-weighted average Word2Vec vector for a single sentence
        and returns an L2-normalized vector.
        """
        tokens = _tokenize(text)
        vec = np.zeros(self.vector_size, dtype=np.float32)
        total_weight = 0.0

        for token in tokens:
            if token in self.model.wv:
                weight = self.idf_dict.get(token, 1.0)
                vec += self.model.wv[token] * weight
                total_weight += weight

        if total_weight > 0:
            vec /= total_weight

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        return vec

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        """
        Encode a list of text strings into L2-normalized Word2Vec sentence embeddings.
        Returns a 2D numpy array of shape (N, vector_size).
        """
        texts = list(texts)
        if not texts:
            return np.empty((0, self.vector_size), dtype=np.float32)

        embeddings = np.array([self.encode_sentence(t) for t in texts], dtype=np.float32)
        return embeddings


if __name__ == "__main__":
    embedder = Word2VecEmbedder()
    vecs = embedder.encode(["The automobile is rapid.", "The vehicle moves fast."])
    print("Word2Vec Encoded Shape:", vecs.shape)
    sim = float(np.dot(vecs[0], vecs[1]))
    print("Word2Vec Cosine Similarity:", round(sim, 4))
