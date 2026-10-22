"""
app.py
Flask web server for the Semantic Duplicate, Paraphrase, and Text Similarity Detection web app.
Loads SBERT (MiniLM), Word2Vec, and spaCy models for multi-model baseline comparison.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from flask import Flask, jsonify, render_template, request

from semantic_integrity.config import SBERT_MODEL, WEB_HOST, WEB_PORT, CROSS_ENCODER_MODEL
from semantic_integrity.embeddings import Embedder
from semantic_integrity.preprocessing import TextPreprocessor
from semantic_integrity.similarity_engine import align_passages, document_level_summary
from semantic_integrity.word2vec_engine import Word2VecEmbedder
from semantic_integrity.faiss_engine import FAISSVectorIndex, SAMPLE_STUDENT_CORPUS
from semantic_integrity.cross_encoder_reranker import CrossEncoderReranker

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")

# Model instances loaded globally once on startup
preprocessor: TextPreprocessor | None = None
embedder: Embedder | None = None
w2v_embedder: Word2VecEmbedder | None = None
faiss_db: FAISSVectorIndex | None = None
cross_encoder: CrossEncoderReranker | None = None


def get_models():
    global preprocessor, embedder, w2v_embedder, faiss_db, cross_encoder
    if preprocessor is None:
        logger.info("Initializing TextPreprocessor (spaCy)...")
        preprocessor = TextPreprocessor()
    if embedder is None:
        logger.info("Initializing Optimized SBERT Embedder ('%s')...", SBERT_MODEL)
        embedder = Embedder(model_name=SBERT_MODEL)
    if w2v_embedder is None:
        logger.info("Initializing Word2Vec Embedder...")
        w2v_embedder = Word2VecEmbedder()
    if faiss_db is None:
        logger.info("Initializing FAISS Vector Index...")
        faiss_db = FAISSVectorIndex()
        if not faiss_db.load():
            logger.info("Building FAISS index with sample student corpus...")
            faiss_db.build_corpus_index(SAMPLE_STUDENT_CORPUS, embedder)
            try:
                faiss_db.save()
                logger.info("Saved FAISS index and metadata to disk.")
            except Exception as e:
                logger.warning("Could not persist FAISS index: %s", e)
    if cross_encoder is None:
        logger.info("Initializing Cross-Encoder Re-ranker ('%s')...", CROSS_ENCODER_MODEL)
        cross_encoder = CrossEncoderReranker(model_name=CROSS_ENCODER_MODEL)

    return preprocessor, embedder, w2v_embedder, faiss_db, cross_encoder


PRESET_DOCUMENTS = {
    "renewable_benchmark": {
        "title": "Renewable Energy Paraphrase Benchmark (5-Tier Test Suite)",
        "doc_a": """Renewable energy sources such as solar and wind power are becoming significantly more affordable with each passing year. As battery storage technology continues to improve, these renewable sources are expected to replace a large share of fossil fuel generation. Traditional power plants that burn coal and natural gas emit large amounts of carbon dioxide into the atmosphere every single day. Governments across the world are now offering tax incentives to homeowners who choose to install solar panels on their rooftops. Meanwhile, urban public transportation systems still rely heavily on diesel buses that consume large quantities of fossil fuel daily.""",
        "doc_b": """Renewable energy sources such as solar and wind power are becoming significantly more affordable with each passing year. As storage battery technology keeps improving, these renewable sources are projected to replace a significant portion of fossil fuel generation. Conventional facilities that rely on burning coal and natural gas release substantial volumes of harmful greenhouse gases into the atmosphere continuously. Tax incentives are increasingly being provided by governments worldwide to encourage residents to add solar panels on top of their homes. In addition, remote teams often use collaborative software platforms to manage weekly project timelines and track task deadlines efficiently."""
    },
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
    ready = preprocessor is not None and embedder is not None
    return jsonify({
        "status": "ready" if ready else "not_initialized",
        "models_ready": ready,
        "components_initialized": {
            "preprocessor": preprocessor is not None,
            "sbert": embedder is not None,
            "word2vec": w2v_embedder is not None,
            "faiss": faiss_db is not None,
            "cross_encoder": cross_encoder is not None,
        },
        "sbert_model": SBERT_MODEL
    }), (200 if ready else 503)


@app.route("/api/presets", methods=["GET"])
def get_presets():
    return jsonify(PRESET_DOCUMENTS)


@app.route("/api/analyze", methods=["POST"])
def analyze_documents():
    try:
        if request.is_json:
            data = request.get_json(silent=True) or {}
            doc_a_raw = data.get("doc_a", "")
            doc_b_raw = data.get("doc_b", "")
            threshold_value = data.get("threshold", 0.55)
            model_type = str(data.get("model_type", "sbert")).lower()
        else:
            doc_a_raw = request.form.get("doc_a", "")
            doc_b_raw = request.form.get("doc_b", "")
            threshold_value = request.form.get("threshold", 0.55)
            model_type = str(request.form.get("model_type", "sbert")).lower()
            if "file_a" in request.files and request.files["file_a"].filename:
                doc_a_raw = request.files["file_a"].read().decode("utf-8", errors="replace")
            if "file_b" in request.files and request.files["file_b"].filename:
                doc_b_raw = request.files["file_b"].read().decode("utf-8", errors="replace")

        if not isinstance(doc_a_raw, str) or not isinstance(doc_b_raw, str):
            return jsonify({"error": "Document inputs must be text strings."}), 400
        doc_a_raw = doc_a_raw.strip()
        doc_b_raw = doc_b_raw.strip()
        try:
            threshold = float(threshold_value)
        except (TypeError, ValueError):
            return jsonify({"error": "threshold must be a number between 0 and 1."}), 400
        if not 0.0 <= threshold <= 1.0:
            return jsonify({"error": "threshold must be between 0 and 1."}), 400

        # Bound request size before invoking expensive NLP models.
        max_chars = int(os.environ.get("VERITEXT_MAX_DOCUMENT_CHARS", "200000"))
        if len(doc_a_raw) > max_chars or len(doc_b_raw) > max_chars:
            return jsonify({"error": f"Each document must be at most {max_chars} characters."}), 413


        if not doc_a_raw or not doc_b_raw:
            return jsonify({"error": "Both Document A and Document B must contain text."}), 400

        # Defer expensive model initialization until request input has passed validation.
        prep, emb_sbert, emb_w2v, f_db, c_reranker = get_models()

        # Preprocess texts using spaCy
        proc_results = prep.process_column([doc_a_raw, doc_b_raw])
        proc_a, proc_b = proc_results[0], proc_results[1]

        sentences_a = proc_a.sentences
        sentences_b = proc_b.sentences

        if not sentences_a or not sentences_b:
            return jsonify({"error": "Could not segment valid sentences from one or both documents."}), 400

        # Select model architecture for alignment
        active_ce = None
        if model_type == "word2vec":
            active_embedder = emb_w2v
            model_name_label = "Word2Vec Dense Average"
        elif model_type in ["cross_encoder", "sbert_cross_encoder", "2stage"]:
            active_embedder = emb_sbert
            active_ce = c_reranker
            model_name_label = f"SBERT + Cross-Encoder 2-Stage Re-ranker"
        else:
            active_embedder = emb_sbert
            model_name_label = f"SBERT ({SBERT_MODEL})"

        # Run passage alignment & tiering
        matches = align_passages(active_embedder, sentences_a, sentences_b, match_threshold=threshold, cross_encoder=active_ce)
        summary = document_level_summary(
            matches,
            total_sentences_a=len(sentences_a),
            total_sentences_b=len(sentences_b),
        )

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
                "cross_encoder_score": round(float(m.cross_encoder_score), 4) if m.cross_encoder_score is not None else None,
                "hybrid_score": round(float(m.hybrid_score), 4) if hasattr(m, 'hybrid_score') and m.hybrid_score is not None else round(float(m.semantic_similarity), 4),
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
                "matched_sentence_ratio_b": summary["matched_sentence_ratio_b"],
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


@app.route("/api/ai-detect", methods=["POST"])
def detect_ai_authorship():
    """Run the independent AI-authorship detector on a single essay."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "A non-empty text string is required."}), 400

    max_chars = int(os.environ.get("VERITEXT_MAX_DOCUMENT_CHARS", "200000"))
    if len(text) > max_chars:
        return jsonify({"error": f"Text must be at most {max_chars} characters."}), 413

    try:
        from semantic_integrity.ai_detection.ensemble import get_detector
        result = get_detector().predict_essay(text.strip(), allow_untrained=False)
        return jsonify(result)
    except RuntimeError as exc:
        # A missing/untrained model is a setup issue, not a negative AI result.
        return jsonify({"error": str(exc), "code": "ai_detector_not_ready"}), 503
    except Exception:
        logger.exception("AI-authorship detection failed")
        return jsonify({"error": "AI-authorship detection failed. Check server logs."}), 500



@app.route("/api/faiss/search", methods=["POST"])
def faiss_search():
    try:
        data = request.get_json(silent=True) or {}

        query_text = data.get("query", "")
        if not isinstance(query_text, str) or not query_text.strip():
            return jsonify({"error": "Query text cannot be empty."}), 400

        query_text = query_text.strip()
        try:
            top_k = int(data.get("top_k", 5))
            min_score = float(data.get("min_score", 0.0))
        except (TypeError, ValueError):
            return jsonify({
                "error": "top_k must be an integer and min_score must be numeric."
            }), 400

        rerank_value = data.get("rerank", True)
        if isinstance(rerank_value, bool):
            rerank = rerank_value
        else:
            return jsonify({"error": "rerank must be true or false."}), 400

        if top_k < 1:
            return jsonify({"error": "top_k must be at least 1."}), 400

        if not -1.0 <= min_score <= 1.0:
            return jsonify({
                "error": "min_score must be between -1.0 and 1.0."
            }), 400

        # Load models only after all request parameters are validated.
        prep, emb_sbert, emb_w2v, f_db, c_reranker = get_models()

        candidates = f_db.search_candidates(
            query_text,
            emb_sbert,
            top_k=top_k,
            min_score=min_score
        )

        if rerank and candidates:
            candidates = c_reranker.rerank_candidates(query_text, candidates)

        return jsonify({
            "total_indexed": f_db.total_documents(),
            "candidates": candidates,
            "reranked_with_cross_encoder": rerank
        })

    except Exception as e:
        logger.exception("FAISS search error")
        return jsonify({"error": str(e)}), 500



@app.route("/api/corpus", methods=["GET"])
def get_corpus():
    try:
        prep, emb_sbert, emb_w2v, f_db, c_reranker = get_models()
        return jsonify({
            "total_documents": f_db.total_documents(),
            "corpus": f_db.metadata
        })
    except Exception as e:
        logger.exception("Get corpus error")
        return jsonify({"error": str(e)}), 500


def main():
    logger.info("Pre-loading NLP, Word2Vec, SBERT, FAISS, and Cross-Encoder models...")
    get_models()
    logger.info("Starting Web Application server on http://%s:%s", WEB_HOST, WEB_PORT)
    app.run(host=WEB_HOST, port=WEB_PORT, debug=False)


if __name__ == "__main__":
    main()
