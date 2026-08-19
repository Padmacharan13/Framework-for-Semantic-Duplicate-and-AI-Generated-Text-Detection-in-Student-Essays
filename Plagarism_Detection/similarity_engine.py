"""
similarity_engine.py
Two things live here:

1. Batch scoring functions used by evaluate.py to benchmark TF-IDF vs SBERT
   on MSRP/QQP (whole-pair similarity, vectorized — no Python loops).
2. Sentence-level passage alignment + paraphrase-intensity tiering used by
   demo.py to produce a human-readable report for two full documents.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

from config import PARAPHRASE_TIERS
from embeddings import Embedder


# ---------------------------------------------------------------------------
# 1. Batch pair-scoring (for evaluate.py)
# ---------------------------------------------------------------------------

def tfidf_pair_similarity(text_a_list: list[str], text_b_list: list[str]) -> np.ndarray:
    """
    Vectorized TF-IDF cosine similarity for a list of (a, b) pairs.
    Fits ONE TF-IDF vocabulary across all texts (both sides, all rows) so
    similarity scores are comparable across the whole dataset.
    """
    vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2))
    all_texts = text_a_list + text_b_list
    vectorizer.fit(all_texts)

    mat_a = vectorizer.transform(text_a_list)
    mat_b = vectorizer.transform(text_b_list)

    # row-wise cosine similarity between mat_a[i] and mat_b[i], not a full
    # NxN matrix - we only need the diagonal for pairwise dataset scoring
    n = mat_a.shape[0]
    sims = np.zeros(n, dtype=np.float32)
    batch = 2000
    for start in range(0, n, batch):
        end = min(start + batch, n)
        sub_sims = sk_cosine_similarity(mat_a[start:end], mat_b[start:end])
        sims[start:end] = np.diagonal(sub_sims)
    return sims


def sbert_pair_similarity(embedder: Embedder, text_a_list: list[str], text_b_list: list[str]) -> np.ndarray:
    """
    Vectorized SBERT cosine similarity for a list of (a, b) pairs.
    Embeddings are pre-normalized, so cosine similarity is just a row-wise
    dot product - no sklearn pairwise call needed.
    """
    emb_a = embedder.encode(text_a_list, show_progress=True)
    emb_b = embedder.encode(text_b_list, show_progress=True)
    sims = np.sum(emb_a * emb_b, axis=1)  # row-wise dot product of normalized vectors
    return sims


def word2vec_pair_similarity(w2v_embedder, text_a_list: list[str], text_b_list: list[str]) -> np.ndarray:
    """
    Vectorized Word2Vec cosine similarity for a list of (a, b) pairs.
    Dense average word vectors are pre-normalized, so cosine similarity is
    row-wise dot product.
    """
    emb_a = w2v_embedder.encode(text_a_list)
    emb_b = w2v_embedder.encode(text_b_list)
    sims = np.sum(emb_a * emb_b, axis=1)
    return sims


# ---------------------------------------------------------------------------
# 2. Passage-level alignment + tiering (for demo.py)
# ---------------------------------------------------------------------------

@dataclass
class MatchedPassage:
    sentence_a: str
    sentence_b: str
    semantic_similarity: float
    lexical_similarity: float
    tier: str


def _lexical_similarity(a: str, b: str) -> float:
    """Cheap word-overlap (Jaccard) signal, used alongside SBERT for tiering."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def _classify_tier(semantic_sim: float, lexical_sim: float) -> str:
    """
    Combines the semantic (SBERT) and lexical (word-overlap) signals into
    one of the four paraphrase-intensity tiers defined in config.py.
    High lexical + high semantic -> verbatim/near-verbatim copying.
    Low lexical + high semantic -> heavy paraphrase / structural rewrite.
    """
    if semantic_sim < 0.55:
        return "not_matched"
    if lexical_sim >= 0.6:
        return "verbatim"
    if lexical_sim >= 0.3:
        return "light_paraphrase"
    if semantic_sim >= 0.75:
        return "heavy_paraphrase"
    return "structural_rewrite"


def align_passages(
    embedder: Embedder,
    sentences_a: list[str],
    sentences_b: list[str],
    match_threshold: float = 0.55,
) -> list[MatchedPassage]:
    """
    Greedy best-match sentence alignment between two documents:
    for every sentence in doc A, find its most similar sentence in doc B.
    Only keeps matches above `match_threshold`. This is the simplified
    version of PAN-style passage alignment - good enough to demo the
    concept without implementing the full contiguous-span-merging algorithm.
    """
    if not sentences_a or not sentences_b:
        return []

    emb_a = embedder.encode(sentences_a)
    emb_b = embedder.encode(sentences_b)
    sim_matrix = emb_a @ emb_b.T  # (len_a, len_b), both sides pre-normalized

    matches: list[MatchedPassage] = []
    for i, sent_a in enumerate(sentences_a):
        j = int(np.argmax(sim_matrix[i]))
        sem_sim = float(sim_matrix[i, j])
        if sem_sim < match_threshold:
            continue
        sent_b = sentences_b[j]
        lex_sim = _lexical_similarity(sent_a, sent_b)
        tier = _classify_tier(sem_sim, lex_sim)
        if tier == "not_matched":
            continue
        matches.append(MatchedPassage(
            sentence_a=sent_a,
            sentence_b=sent_b,
            semantic_similarity=sem_sim,
            lexical_similarity=lex_sim,
            tier=tier,
        ))
    return matches


def document_level_summary(matches: list[MatchedPassage], total_sentences_a: int) -> dict:
    """Rolls up passage-level matches into one overall document verdict."""
    if not matches or total_sentences_a == 0:
        return {"overall_score": 0.0, "flagged": False, "matched_sentence_ratio": 0.0, "tier_counts": {}}

    avg_sem = float(np.mean([m.semantic_similarity for m in matches]))
    ratio = len(matches) / total_sentences_a
    tier_counts = {t: sum(1 for m in matches if m.tier == t) for t in PARAPHRASE_TIERS}

    return {
        "overall_score": round(avg_sem, 4),
        "matched_sentence_ratio": round(ratio, 4),
        "flagged": avg_sem >= 0.75 and ratio >= 0.4,
        "tier_counts": tier_counts,
    }