import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from word2vec_engine import Word2VecEmbedder
from config import WORD2VEC_MODEL_PATH, SBERT_MODEL

class SimilarityBenchmarkEngine:
    def __init__(self):
        # 1. TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.tfidf_fitted = False

        # 2. Word2Vec
        self.w2v_embedder = Word2VecEmbedder(WORD2VEC_MODEL_PATH)

        # 3. SBERT model (base or fine‑tuned) as configured
        self.sbert_model = SentenceTransformer(SBERT_MODEL)

        # 4. Cross-Encoder Re-ranker
        from cross_encoder_reranker import CrossEncoderReranker
        self.cross_encoder = CrossEncoderReranker()

    def fit_tfidf(self, corpus: list[str]):
        self.tfidf_vectorizer.fit(corpus)
        self.tfidf_fitted = True

    def compute_tfidf_similarity(self, text1_list: list[str], text2_list: list[str]) -> np.ndarray:
        if not self.tfidf_fitted:
            self.fit_tfidf(text1_list + text2_list)
        vecs1 = self.tfidf_vectorizer.transform(text1_list)
        vecs2 = self.tfidf_vectorizer.transform(text2_list)
        # Compute row-wise cosine similarity (not element-wise dot product)
        sims = np.array([
            float(cosine_similarity(vecs1[i], vecs2[i])[0, 0])
            for i in range(vecs1.shape[0])
        ])
        return np.clip(sims, 0.0, 1.0)

    def compute_w2v_similarity(self, text1_list: list[str], text2_list: list[str]) -> np.ndarray:
        vecs1 = self.w2v_embedder.encode(text1_list)
        vecs2 = self.w2v_embedder.encode(text2_list)
        sims = np.sum(vecs1 * vecs2, axis=1)
        return np.clip(sims, 0.0, 1.0)

    def compute_sbert_similarity(self, text1_list: list[str], text2_list: list[str]) -> np.ndarray:
        vecs1 = self.sbert_model.encode(text1_list, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
        vecs2 = self.sbert_model.encode(text2_list, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
        return np.sum(vecs1 * vecs2, axis=1)

    def compute_hybrid_similarity(self, text1_list: list[str], text2_list: list[str]) -> np.ndarray:
        sbert_sim = self.compute_sbert_similarity(text1_list, text2_list)
        tfidf_sim = self.compute_tfidf_similarity(text1_list, text2_list)
        return 0.8 * sbert_sim + 0.2 * tfidf_sim

    def compute_cross_encoder_similarity(self, text1_list: list[str], text2_list: list[str]) -> np.ndarray:
        pairs = list(zip(text1_list, text2_list))
        scores = self.cross_encoder.predict_pairs(pairs)
        return np.clip(np.nan_to_num(scores, nan=0.0), 0.0, 1.0)


import re
from dataclasses import dataclass


@dataclass
class PassageMatch:
    sentence_a: str
    sentence_b: str
    semantic_similarity: float
    lexical_similarity: float
    tier: str
    cross_encoder_score: float | None = None
    hybrid_score: float | None = None


def compute_lexical_similarity(str1: str, str2: str) -> float:
    words1 = set(re.findall(r"\w+", str1.lower()))
    words2 = set(re.findall(r"\w+", str2.lower()))
    if not words1 or not words2:
        return 0.0
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union)


def align_passages(
    embedder,
    sentences_a: list[str],
    sentences_b: list[str],
    match_threshold: float = 0.55,
    cross_encoder=None
) -> list[PassageMatch]:
    if not sentences_a or not sentences_b:
        return []

    vecs_a = embedder.encode(sentences_a)
    vecs_b = embedder.encode(sentences_b)

    norms_a = np.linalg.norm(vecs_a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(vecs_b, axis=1, keepdims=True)
    norms_a[norms_a == 0] = 1.0
    norms_b[norms_b == 0] = 1.0

    normed_a = vecs_a / norms_a
    normed_b = vecs_b / norms_b

    sim_matrix = np.dot(normed_a, normed_b.T)

    matches: list[PassageMatch] = []
    pairs_to_rerank = []
    match_candidates = []

    for i, sent_a in enumerate(sentences_a):
        best_j = int(np.argmax(sim_matrix[i]))
        best_sim = float(sim_matrix[i, best_j])

        if best_sim >= match_threshold:
            sent_b = sentences_b[best_j]
            lex_sim = float(compute_lexical_similarity(sent_a, sent_b))
            match_candidates.append((sent_a, sent_b, best_sim, lex_sim))
            if cross_encoder is not None:
                pairs_to_rerank.append((sent_a, sent_b))

    ce_scores = []
    if cross_encoder is not None and pairs_to_rerank:
        try:
            ce_scores = cross_encoder.predict_pairs(pairs_to_rerank)
        except Exception as e:
            ce_scores = [None] * len(pairs_to_rerank)

    for idx, (sent_a, sent_b, best_sim, lex_sim) in enumerate(match_candidates):
        ce_score = float(ce_scores[idx]) if idx < len(ce_scores) and ce_scores[idx] is not None else None
        
        # Effective similarity score for tier classification
        if ce_score is not None:
            eff_sim = 0.6 * ce_score + 0.4 * best_sim
        else:
            eff_sim = best_sim

        norm_a = sent_a.strip().lower()
        norm_b = sent_b.strip().lower()

        if norm_a == norm_b or lex_sim >= 0.85 or (eff_sim >= 0.98 and lex_sim >= 0.75):
            tier = "verbatim"
        elif eff_sim >= 0.88 and lex_sim >= 0.45:
            tier = "light_paraphrase"
        elif eff_sim >= 0.80 and lex_sim < 0.45:
            tier = "structural_rewrite"
        elif eff_sim >= 0.60:
            tier = "heavy_paraphrase"
        else:
            tier = "no_match"

        matches.append(
            PassageMatch(
                sentence_a=sent_a,
                sentence_b=sent_b,
                semantic_similarity=best_sim,
                lexical_similarity=lex_sim,
                tier=tier,
                cross_encoder_score=ce_score,
                hybrid_score=round(eff_sim, 4) if ce_score is not None else round(best_sim, 4)
            )
        )

    return matches


def document_level_summary(matches: list[PassageMatch], total_sentences_a: int) -> dict:
    empty_tier_counts = {
        "verbatim": 0,
        "light_paraphrase": 0,
        "heavy_paraphrase": 0,
        "structural_rewrite": 0,
        "no_match": 0
    }
    if total_sentences_a == 0:
        return {
            "overall_score": 0.0,
            "matched_sentence_ratio": 0.0,
            "flagged": False,
            "tier_counts": empty_tier_counts
        }

    matched_count = len(matches)
    ratio = matched_count / total_sentences_a

    tier_counts = {
        "verbatim": 0,
        "light_paraphrase": 0,
        "heavy_paraphrase": 0,
        "structural_rewrite": 0,
        "no_match": 0
    }
    avg_sim = 0.0
    if matches:
        avg_sim = float(np.mean([m.semantic_similarity for m in matches]))
        for m in matches:
            tier_counts[m.tier] = tier_counts.get(m.tier, 0) + 1

    overall_score = float(round(avg_sim * ratio, 4))
    flagged = bool(overall_score >= 0.35 or (matched_count > 0 and ratio >= 0.5))

    return {
        "overall_score": overall_score,
        "matched_sentence_ratio": round(ratio, 4),
        "flagged": flagged,
        "tier_counts": tier_counts
    }