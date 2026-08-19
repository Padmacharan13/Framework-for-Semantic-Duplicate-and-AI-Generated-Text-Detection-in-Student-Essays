"""
app.py
Flask web server for the Semantic Duplicate, Paraphrase, and AI-Generated Text Detection web app.
Loads SBERT (MPNet), Word2Vec, and spaCy models for multi-model baseline comparison.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from flask import Flask, jsonify, render_template, request

from config import SBERT_MODEL, WEB_HOST, WEB_PORT
from embeddings import Embedder
from preprocessing import TextPreprocessor
from similarity_engine import align_passages, document_level_summary
from word2vec_engine import Word2VecEmbedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")

# Model instances loaded globally once on startup
preprocessor: TextPreprocessor | None = None
embedder: Embedder | None = None
w2v_embedder: Word2VecEmbedder | None = None


def get_models() -> tuple[TextPreprocessor, Embedder, Word2VecEmbedder]:
    global preprocessor, embedder, w2v_embedder
    if preprocessor is None:
        logger.info("Initializing TextPreprocessor (spaCy)...")
        preprocessor = TextPreprocessor()
    if embedder is None:
        logger.info("Initializing Optimized SBERT Embedder ('%s')...", SBERT_MODEL)
        embedder = Embedder(model_name=SBERT_MODEL)
    if w2v_embedder is None:
        logger.info("Initializing Word2Vec Embedder...")
        w2v_embedder = Word2VecEmbedder()
    return preprocessor, embedder, w2v_embedder


PRESET_DOCUMENTS = {
    "paraphrased_essay": {
        "title": "Academic Paraphrasing (Heavy Rewrite)",
        "doc_a": """Academic integrity relies heavily on the authentic submission of assignments and essays. Digital submissions have scaled drastically, making manual comparison for duplicates virtually impossible for instructors. There is an urgent need for automated pipelines that can scan batches of student texts and index their underlying conceptual content. Traditional string-matching methods fail when a student restructures an essay using synonym substitution or syntax swapping while preserving the original meaning.""",
        "doc_b": """Instructors depend a great deal on students submitting their own original work for courses to remain fair. As the number of digital submissions has grown enormously, checking each one by hand for duplication is no longer realistic for teaching staff. Traditional string-matching methods fail when a student restructures an essay using synonym substitution or syntax swapping while preserving the original meaning. Renewable energy sources such as solar power are becoming more cost effective every year."""
    },
    "verbatim_copy": {
        "title": "Direct Verbatim Copying (Plagiarism)",
        "doc_a": """Artificial Intelligence is reshaping higher education by enabling personalized learning paths for students. Adaptive algorithms analyze historical student data to recommend targeted reading materials and tailored practice exercises. However, educational institutions must establish strict policies regarding academic honesty and proper citation.""",
        "doc_b": """Artificial Intelligence is reshaping higher education by enabling personalized learning paths for students. Adaptive algorithms analyze historical student data to recommend targeted reading materials and tailored practice exercises. Students can submit their assignments online through a centralized portal for automated feedback."""
    },
    "unrelated_topics": {
        "title": "Unrelated Topics (Low Similarity)",
        "doc_a": """Photosynthesis is the biological process used by plants, algae, and certain bacteria to convert light energy into chemical energy. Chlorophyll absorbs sunlight and drives the reaction between carbon dioxide and water to produce glucose and oxygen gas.""",
        "doc_b": """Quantum computing leverages quantum mechanics principles such as superposition and entanglement to perform complex computations. Quantum bits or qubits can represent multiple states simultaneously, offering exponential speedups for specialized mathematical algorithms."""
    },
    "technical_rewrite": {
        "title": "Technical / Code Documentation Rewrite",
        "doc_a": """The microservices architecture decomposes a large monolithic application into small, loosely coupled services. Each service runs in its own process, communicates via lightweight REST APIs, and can be deployed independently without affecting the broader system.""",
        "doc_b": """Monolithic software systems can be broken down into autonomous micro-services. These individual services execute as independent OS processes, exchange messages using REST APIs, and allow separate deployment pipelines that minimize system-wide blast radius."""
    }
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "models_ready": preprocessor is not None and embedder is not None and w2v_embedder is not None,
        "sbert_model": SBERT_MODEL
    })


@app.route("/api/presets", methods=["GET"])
def get_presets():
    return jsonify(PRESET_DOCUMENTS)


@app.route("/api/analyze", methods=["POST"])
def analyze_documents():
    try:
        prep, emb_sbert, emb_w2v = get_models()

        if request.is_json:
            data = request.get_json() or {}
            doc_a_raw = data.get("doc_a", "").strip()
            doc_b_raw = data.get("doc_b", "").strip()
            threshold = float(data.get("threshold", 0.55))
            model_type = data.get("model_type", "sbert").lower()
        else:
            doc_a_raw = request.form.get("doc_a", "").strip()
            doc_b_raw = request.form.get("doc_b", "").strip()
            threshold = float(request.form.get("threshold", 0.55))
            model_type = request.form.get("model_type", "sbert").lower()

            if "file_a" in request.files and request.files["file_a"].filename:
                doc_a_raw = request.files["file_a"].read().decode("utf-8", errors="ignore").strip()
            if "file_b" in request.files and request.files["file_b"].filename:
                doc_b_raw = request.files["file_b"].read().decode("utf-8", errors="ignore").strip()

        if not doc_a_raw or not doc_b_raw:
            return jsonify({"error": "Both Document A and Document B must contain text."}), 400

        # Preprocess texts using spaCy
        proc_results = prep.process_column([doc_a_raw, doc_b_raw])
        proc_a, proc_b = proc_results[0], proc_results[1]

        sentences_a = proc_a.sentences
        sentences_b = proc_b.sentences

        if not sentences_a or not sentences_b:
            return jsonify({"error": "Could not segment valid sentences from one or both documents."}), 400

        # Select model architecture for alignment
        if model_type == "word2vec":
            active_embedder = emb_w2v
            model_name_label = "Word2Vec Dense Average"
        else:
            active_embedder = emb_sbert
            model_name_label = f"SBERT ({SBERT_MODEL})"

        # Run passage alignment & tiering
        matches = align_passages(active_embedder, sentences_a, sentences_b, match_threshold=threshold)
        summary = document_level_summary(matches, total_sentences_a=len(sentences_a))

        # Build match mappings for frontend rendering
        matches_a_map = {}
        matches_b_map = {}
        match_list_json = []

        for idx, m in enumerate(matches):
            try:
                idx_a = sentences_a.index(m.sentence_a)
            except ValueError:
                idx_a = -1

            try:
                idx_b = sentences_b.index(m.sentence_b)
            except ValueError:
                idx_b = -1

            match_data = {
                "match_id": idx + 1,
                "sentence_a": m.sentence_a,
                "sentence_b": m.sentence_b,
                "index_a": idx_a,
                "index_b": idx_b,
                "semantic_similarity": round(float(m.semantic_similarity), 4),
                "lexical_similarity": round(float(m.lexical_similarity), 4),
                "tier": m.tier,
                "tier_label": m.tier.replace("_", " ").title()
            }
            match_list_json.append(match_data)

            if idx_a >= 0:
                matches_a_map[idx_a] = match_data
            if idx_b >= 0:
                if idx_b not in matches_b_map or match_data["semantic_similarity"] > matches_b_map[idx_b]["semantic_similarity"]:
                    matches_b_map[idx_b] = match_data

        # Construct structured sentence lists
        doc_a_sentences_structured = []
        for i, s in enumerate(sentences_a):
            m_info = matches_a_map.get(i)
            doc_a_sentences_structured.append({
                "index": i,
                "text": s,
                "is_matched": m_info is not None,
                "match_id": m_info["match_id"] if m_info else None,
                "paired_index": m_info["index_b"] if m_info else None,
                "tier": m_info["tier"] if m_info else "none",
                "semantic_similarity": m_info["semantic_similarity"] if m_info else 0.0,
                "lexical_similarity": m_info["lexical_similarity"] if m_info else 0.0
            })

        doc_b_sentences_structured = []
        for j, s in enumerate(sentences_b):
            m_info = matches_b_map.get(j)
            doc_b_sentences_structured.append({
                "index": j,
                "text": s,
                "is_matched": m_info is not None,
                "match_id": m_info["match_id"] if m_info else None,
                "paired_index": m_info["index_a"] if m_info else None,
                "tier": m_info["tier"] if m_info else "none",
                "semantic_similarity": m_info["semantic_similarity"] if m_info else 0.0,
                "lexical_similarity": m_info["lexical_similarity"] if m_info else 0.0
            })

        response = {
            "model_used": model_name_label,
            "summary": {
                "overall_score": summary["overall_score"],
                "matched_sentence_ratio": summary["matched_sentence_ratio"],
                "flagged": summary["flagged"],
                "total_sentences_a": len(sentences_a),
                "total_sentences_b": len(sentences_b),
                "matched_count": len(matches),
                "tier_counts": summary["tier_counts"]
            },
            "matches": match_list_json,
            "doc_a": {
                "raw": doc_a_raw,
                "paragraphs": proc_a.paragraphs,
                "sentences": doc_a_sentences_structured
            },
            "doc_b": {
                "raw": doc_b_raw,
                "paragraphs": proc_b.paragraphs,
                "sentences": doc_b_sentences_structured
            }
        }
        return jsonify(response)

    except Exception as e:
        logger.exception("Error analyzing documents")
        return jsonify({"error": str(e)}), 500


def main():
    logger.info("Pre-loading NLP, Word2Vec, and SBERT models...")
    get_models()
    logger.info("Starting Web Application server on http://%s:%s", WEB_HOST, WEB_PORT)
    app.run(host=WEB_HOST, port=WEB_PORT, debug=False)


if __name__ == "__main__":
    main()
