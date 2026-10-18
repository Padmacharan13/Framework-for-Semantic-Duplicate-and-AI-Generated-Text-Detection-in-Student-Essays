import numpy as np
import logging
import spacy
from pathlib import Path
from gensim.models import Word2Vec
from .config import WORD2VEC_MODEL_PATH, W2V_VECTOR_SIZE, W2V_WINDOW, W2V_MIN_COUNT, W2V_EPOCHS

logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def tokenize(text: str) -> list[str]:
    return [token.lemma_.lower() for token in nlp(str(text)) if not token.is_punct and not token.is_space]

def train_word2vec(sentences_corpus: list[str], save_path=WORD2VEC_MODEL_PATH):
    valid_sentences = [str(s) for s in sentences_corpus if s]
    docs = nlp.pipe(valid_sentences, batch_size=1000)
    tokenized_corpus = [
        [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
        for doc in docs
    ]
    model = Word2Vec(
        sentences=tokenized_corpus,
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        workers=4,
        epochs=W2V_EPOCHS
    )
    model.save(str(save_path))
    return model

class Word2VecEmbedder:
    def __init__(self, model_path=WORD2VEC_MODEL_PATH):
        self.available = False
        self.dim = W2V_VECTOR_SIZE
        model_file = Path(model_path)
        if model_file.exists():
            try:
                self.model = Word2Vec.load(str(model_path))
                self.dim = self.model.vector_size
                self.available = True
            except Exception as e:
                logger.warning("Failed to load Word2Vec model from '%s': %s. Using fallback zero vectors.", model_path, e)
                self.model = None
        else:
            logger.warning("Word2Vec model not found at '%s'. Word2Vec features will return zero vectors.", model_path)
            self.model = None

    def get_sentence_vector(self, text: str) -> np.ndarray:
        if not self.available or self.model is None:
            raise RuntimeError("Word2Vec model is unavailable; cannot compute sentence vectors.")
        tokens = tokenize(text)
        vectors = [self.model.wv[t] for t in tokens if t in self.model.wv]
        if not vectors:
            return np.zeros(self.dim, dtype=np.float32)
        mean_vec = np.mean(vectors, axis=0)
        norm = np.linalg.norm(mean_vec)
        return (mean_vec / norm) if norm > 0 else mean_vec

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self.available or self.model is None:
            raise RuntimeError("Word2Vec model is unavailable; cannot encode texts.")
        docs = nlp.pipe([str(t) for t in texts], batch_size=2000)
        res = []
        for doc in docs:
            tokens = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
            vectors = [self.model.wv[t] for t in tokens if t in self.model.wv]
            if not vectors:
                res.append(np.zeros(self.dim, dtype=np.float32))
            else:
                mean_vec = np.mean(vectors, axis=0)
                norm = np.linalg.norm(mean_vec)
                res.append((mean_vec / norm) if norm > 0 else mean_vec)
        return np.array(res, dtype=np.float32)