"""
dashboard.py
Streamlit Interactive Instructor Dashboard for Semantic Duplicate, Paraphrase, and AI-Generated Text Detection.
Features FAISS Vector Index candidate retrieval, Cross-Encoder 2-stage re-ranking, single-pair passage alignment, and batch audit reports.
"""

from __future__ import annotations

import io
import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from config import SBERT_MODEL, CROSS_ENCODER_MODEL
from embeddings import Embedder
from preprocessing import TextPreprocessor
from similarity_engine import align_passages, document_level_summary
from word2vec_engine import Word2VecEmbedder
from faiss_engine import FAISSVectorIndex, SAMPLE_STUDENT_CORPUS
from cross_encoder_reranker import CrossEncoderReranker

# ==========================================
# Page Configuration & Custom Styling
# ==========================================
st.set_page_config(
    page_title="VeriText AI - Instructor Plagiarism & Duplicate Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and modern typography
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.8rem 2rem;
        border-radius: 12px;
        color: #ffffff;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        color: #f8fafc;
        font-weight: 700;
        margin: 0;
        font-size: 2.2rem;
    }

    .main-header p {
        color: #94a3b8;
        margin-top: 0.5rem;
        margin-bottom: 0;
        font-size: 1.05rem;
    }

    .metric-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
        transition: transform 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
    }

    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
    }

    .metric-lbl {
        color: #64748b;
        font-size: 0.88rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .tier-verbatim {
        background-color: #fee2e2;
        color: #991b1b;
        border-left: 4px solid #ef4444;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }

    .tier-light {
        background-color: #ffedd5;
        color: #9a3412;
        border-left: 4px solid #f97316;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }

    .tier-structural {
        background-color: #f3e8ff;
        color: #6b21a8;
        border-left: 4px solid #a855f7;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }

    .tier-heavy {
        background-color: #fef9c3;
        color: #854d0e;
        border-left: 4px solid #eab308;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }

    .unmatched-sent {
        background-color: #f8fafc;
        color: #334155;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        border-left: 4px solid #cbd5e1;
    }

    .badge-tag {
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        text-transform: uppercase;
        display: inline-block;
        margin-right: 0.5rem;
    }

    .badge-verbatim { background-color: #ef4444; color: white; }
    .badge-light { background-color: #f97316; color: white; }
    .badge-structural { background-color: #a855f7; color: white; }
    .badge-heavy { background-color: #eab308; color: white; }

</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==========================================
# Model Resource Caching
# ==========================================
@st.cache_resource(show_spinner=False)
def load_all_models():
    """
    Loads all core models ONCE into Streamlit memory.
    """
    preprocessor = TextPreprocessor()
    embedder = Embedder(model_name=SBERT_MODEL)
    w2v_embedder = Word2VecEmbedder()
    faiss_db = FAISSVectorIndex()
    if not faiss_db.load():
        faiss_db.build_corpus_index(SAMPLE_STUDENT_CORPUS, embedder)
    cross_encoder = CrossEncoderReranker(model_name=CROSS_ENCODER_MODEL)
    return preprocessor, embedder, w2v_embedder, faiss_db, cross_encoder


# ==========================================
# Application Header & Sidebar
# ==========================================
st.markdown(
    """
    <div class="main-header">
        <h1>🛡️ VeriText AI — Academic Duplicate & Paraphrase Detector</h1>
        <p>Two-Stage Semantic Intelligence: Fast FAISS Candidate Retrieval + Precision Cross-Encoder Re-ranking</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar Navigation
st.sidebar.markdown("### 🎓 VeriText AI")
st.sidebar.title("Navigation & Control")

navigation_mode = st.sidebar.radio(
    "Select Module:",
    [
        "📄 Document Pair Alignment",
        "🔍 FAISS Candidate Search",
        "⚡ Two-Stage Pipeline Precision Pass",
        "📊 Instructor Batch Audit Report"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ System Status")

# Initialize models
with st.spinner("Initializing SBERT, FAISS Index, and Cross-Encoder..."):
    preprocessor, embedder, w2v_embedder, faiss_db, cross_encoder = load_all_models()

st.sidebar.success(f"✅ SBERT ({SBERT_MODEL}) Ready")
st.sidebar.success(f"✅ FAISS Index Active ({faiss_db.total_documents()} docs)")
st.sidebar.success(f"✅ Cross-Encoder Active")

st.sidebar.markdown("---")
st.sidebar.info("🎓 **Framework for Semantic Duplicate & AI Text Detection in Student Essays**")


# ==========================================
# MODULE 1: Single Document Pair Alignment
# ==========================================
if navigation_mode == "📄 Document Pair Alignment":
    st.header("📄 Single Document Pair Alignment & Tier Classification")
    st.caption("Compare two student essays or assignments to detect verbatim copying, paraphrasing, or structural rewrites.")

    preset_options = {
        "Custom Input": ("Pasted / Custom Text", "", ""),
        "Renewable Energy Benchmark": (
            "Renewable Energy Benchmark (5-Tier Test)",
            "Renewable energy sources such as solar and wind power are becoming significantly more affordable with each passing year. As battery storage technology continues to improve, these renewable sources are expected to replace a large share of fossil fuel generation. Traditional power plants that burn coal and natural gas emit large amounts of carbon dioxide into the atmosphere every single day. Governments across the world are now offering tax incentives to homeowners who choose to install solar panels on their rooftops.",
            "Renewable energy sources such as solar and wind power are becoming significantly more affordable with each passing year. As storage battery technology keeps improving, these renewable sources are projected to replace a significant portion of fossil fuel generation. Conventional facilities that rely on burning coal and natural gas release substantial volumes of harmful greenhouse gases into the atmosphere continuously. Tax incentives are increasingly being provided by governments worldwide to encourage residents to add solar panels on top of their homes."
        ),
        "Academic Integrity Essay": (
            "Academic Paraphrasing (Heavy Rewrite)",
            "Academic integrity relies heavily on the authentic submission of assignments and essays. Digital submissions have scaled drastically, making manual comparison for duplicates virtually impossible for instructors. There is an urgent need for automated pipelines that can scan batches of student texts and index their underlying conceptual content. Traditional string-matching methods fail when a student restructures an essay using synonym substitution or syntax swapping while preserving the original meaning.",
            "Instructors depend a great deal on students submitting their own original work for courses to remain fair. As the number of digital submissions has grown enormously, checking each one by hand for duplication is no longer realistic for teaching staff. Traditional string-matching methods fail when a student restructures an essay using synonym substitution or syntax swapping while preserving the original meaning. Renewable energy sources such as solar power are becoming more cost effective every year."
        ),
        "Verbatim Copying": (
            "Direct Verbatim Copying (Plagiarism)",
            "Artificial Intelligence is reshaping higher education by enabling personalized learning paths for students. Adaptive algorithms analyze historical student data to recommend targeted reading materials and tailored practice exercises. However, educational institutions must establish strict policies regarding academic honesty and proper citation.",
            "Artificial Intelligence is reshaping higher education by enabling personalized learning paths for students. Adaptive algorithms analyze historical student data to recommend targeted reading materials and tailored practice exercises. Students can submit their assignments online through a centralized portal for automated feedback."
        ),
        "Unrelated Topics": (
            "Unrelated Topics (Low Similarity)",
            "Photosynthesis is the biological process used by plants, algae, and certain bacteria to convert light energy into chemical energy. Chlorophyll absorbs sunlight and drives the reaction between carbon dioxide and water to produce glucose and oxygen gas.",
            "Quantum computing leverages quantum mechanics principles such as superposition and entanglement to perform complex computations. Quantum bits or qubits can represent multiple states simultaneously, offering exponential speedups for specialized mathematical algorithms."
        )
    }

    selected_preset = st.selectbox("Load Test Preset:", list(preset_options.keys()))
    _, default_doc_a, default_doc_b = preset_options[selected_preset]

    col_model, col_thresh = st.columns([2, 1])
    with col_model:
        model_selection = st.selectbox(
            "Select Alignment Model Architecture:",
            [
                "SBERT + Cross-Encoder 2-Stage Re-ranker (Recommended)",
                "SBERT Bi-Encoder (Dense Vector Dot Product)",
                "Word2Vec Dense Average Embeddings",
            ]
        )
    with col_thresh:
        match_threshold = st.slider("Sentence Matching Threshold:", min_value=0.30, max_value=0.90, value=0.55, step=0.05)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Document A (Student Essay 1)")
        doc_a_input = st.text_area("Content for Document A:", value=default_doc_a, height=200, key="doc_a_area")
        file_a = st.file_uploader("Or upload Document A (.txt):", type=["txt"], key="file_a")
        if file_a:
            doc_a_input = file_a.getvalue().decode("utf-8", errors="ignore")

    with col_b:
        st.subheader("Document B (Student Essay 2)")
        doc_b_input = st.text_area("Content for Document B:", value=default_doc_b, height=200, key="doc_b_area")
        file_b = st.file_uploader("Or upload Document B (.txt):", type=["txt"], key="file_b")
        if file_b:
            doc_b_input = file_b.getvalue().decode("utf-8", errors="ignore")

    if st.button("🚀 Analyze Alignment & Perform Precision Pass", type="primary", use_container_width=True):
        if not doc_a_input.strip() or not doc_b_input.strip():
            st.error("⚠️ Please provide text for both Document A and Document B.")
        else:
            with st.spinner("Segmenting sentences, encoding embeddings, and executing cross-attention pass..."):
                start_t = time.time()
                proc_a, proc_b = preprocessor.process_column([doc_a_input, doc_b_input])

                sents_a, sents_b = proc_a.sentences, proc_b.sentences

                if "Cross-Encoder" in model_selection:
                    active_emb = embedder
                    active_ce = cross_encoder
                elif "Word2Vec" in model_selection:
                    active_emb = w2v_embedder
                    active_ce = None
                else:
                    active_emb = embedder
                    active_ce = None

                matches = align_passages(active_emb, sents_a, sents_b, match_threshold=match_threshold, cross_encoder=active_ce)
                summary = document_level_summary(matches, len(sents_a))
                proc_time = (time.time() - start_t) * 1000.0

            # Metrics Summary Bar
            st.markdown("---")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Overall Plagiarism Score", f"{summary['overall_score'] * 100:.1f}%")
            m2.metric("Matched Sentence Ratio", f"{summary['matched_sentence_ratio'] * 100:.1f}%")
            m3.metric("Total Sentences (A / B)", f"{len(sents_a)} / {len(sents_b)}")
            m4.metric("Flagged Status", "🚨 FLAGGED" if summary['flagged'] else "✅ CLEAN")
            m5.metric("Latency", f"{proc_time:.1f} ms")

            st.markdown("### 📊 Sentence Alignment Tier Breakdown")
            tier_cnts = summary["tier_counts"]
            tc1, tc2, tc3, tc4 = st.columns(4)
            tc1.info(f"🔴 Verbatim Copies: **{tier_cnts.get('verbatim', 0)}**")
            tc2.warning(f"🟠 Light Paraphrase: **{tier_cnts.get('light_paraphrase', 0)}**")
            tc3.error(f"🟣 Structural Rewrite: **{tier_cnts.get('structural_rewrite', 0)}**")
            tc4.success(f"🟡 Heavy Paraphrase: **{tier_cnts.get('heavy_paraphrase', 0)}**")

            # Side-by-side Highlighted Sentence View
            st.markdown("---")
            st.subheader("🔍 Side-by-Side Sentence Passage Comparison")

            matched_a_dict = {m.sentence_a: m for m in matches}

            col_view_a, col_view_b = st.columns(2)
            with col_view_a:
                st.markdown("#### Document A (Highlighted)")
                for idx, sent in enumerate(sents_a, start=1):
                    if sent in matched_a_dict:
                        m = matched_a_dict[sent]
                        tier_cls = f"tier-{m.tier.split('_')[0]}"
                        ce_info = f" | Cross-Encoder: {m.cross_encoder_score:.4f}" if m.cross_encoder_score is not None else ""
                        st.markdown(
                            f"""<div class="{tier_cls}">
                                <span class="badge-tag badge-{m.tier.split('_')[0]}">{m.tier.replace('_', ' ')}</span>
                                <strong>#{idx}</strong>: {sent}<br/>
                                <small style="opacity: 0.8;">SBERT Sim: {m.semantic_similarity:.4f} | Lexical: {m.lexical_similarity*100:.1f}%{ce_info}</small>
                            </div>""",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(f'<div class="unmatched-sent"><strong>#{idx}</strong>: {sent}</div>', unsafe_allow_html=True)

            with col_view_b:
                st.markdown("#### Document B (Paired Alignment)")
                matched_b_dict = {m.sentence_b: m for m in matches}
                for idx, sent in enumerate(sents_b, start=1):
                    if sent in matched_b_dict:
                        m = matched_b_dict[sent]
                        tier_cls = f"tier-{m.tier.split('_')[0]}"
                        ce_info = f" | Cross-Encoder: {m.cross_encoder_score:.4f}" if m.cross_encoder_score is not None else ""
                        st.markdown(
                            f"""<div class="{tier_cls}">
                                <span class="badge-tag badge-{m.tier.split('_')[0]}">{m.tier.replace('_', ' ')}</span>
                                <strong>#{idx}</strong>: {sent}<br/>
                                <small style="opacity: 0.8;">SBERT Sim: {m.semantic_similarity:.4f} | Lexical: {m.lexical_similarity*100:.1f}%{ce_info}</small>
                            </div>""",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(f'<div class="unmatched-sent"><strong>#{idx}</strong>: {sent}</div>', unsafe_allow_html=True)

            # Matched Pairs Detailed Table
            if matches:
                st.markdown("---")
                st.subheader("📋 Matched Sentence Pair Records")
                table_data = []
                for idx, m in enumerate(matches, start=1):
                    table_data.append({
                        "Pair #": idx,
                        "Document A Sentence": m.sentence_a,
                        "Document B Sentence": m.sentence_b,
                        "Bi-Encoder Sim": round(m.semantic_similarity, 4),
                        "Cross-Encoder Score": round(m.cross_encoder_score, 4) if m.cross_encoder_score is not None else "N/A",
                        "Hybrid Score": round(m.hybrid_score, 4) if m.hybrid_score is not None else round(m.semantic_similarity, 4),
                        "Lexical Overlap": f"{m.lexical_similarity * 100:.1f}%",
                        "Tier Tag": m.tier.replace("_", " ").title()
                    })
                df_matches = pd.DataFrame(table_data)
                st.dataframe(df_matches, use_container_width=True)


# ==========================================
# MODULE 2: FAISS Candidate Search
# ==========================================
elif navigation_mode == "🔍 FAISS Candidate Search":
    st.header("🔍 FAISS Vector Database: Fast Corpus-Wide Candidate Retrieval")
    st.caption("Instantly query student essays against an indexed vector corpus using L2-normalized SBERT inner products.")

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Indexed Documents in Vector DB", faiss_db.total_documents())
    col_info2.metric("Vector Dimension", "384 (SBERT MiniLM-L6)")
    col_info3.metric("Index Architecture", "FAISS IndexFlatIP (L2 Normed)")

    st.markdown("---")
    st.subheader("🔎 Query Corpus Candidates")

    query_essay = st.text_area(
        "Enter Query Essay or Text Snippet to Search:",
        value="Renewable power generation with solar panels and wind turbines is replacing conventional fossil fuel plants quickly as battery storage improves.",
        height=120
    )

    col_k, col_rerank = st.columns([1, 1])
    top_k = col_k.slider("Top K Candidates to Retrieve:", min_value=1, max_value=10, value=5)
    enable_ce_rerank = col_rerank.checkbox("Enable Secondary Cross-Encoder Re-ranking", value=True)

    if st.button("⚡ Execute FAISS Candidate Search", type="primary", use_container_width=True):
        if not query_essay.strip():
            st.error("Please enter query text to search.")
        else:
            with st.spinner("Querying FAISS vector index & re-ranking candidates..."):
                t0 = time.time()
                candidates = faiss_db.search_candidates(query_essay, embedder, top_k=top_k)
                retrieval_ms = (time.time() - t0) * 1000.0

                if enable_ce_rerank and candidates:
                    t1 = time.time()
                    candidates = cross_encoder.rerank_candidates(query_essay, candidates)
                    rerank_ms = (time.time() - t1) * 1000.0
                else:
                    rerank_ms = 0.0

            st.success(f"⚡ FAISS Retrieval completed in **{retrieval_ms:.2f} ms** | Cross-Encoder Re-rank in **{rerank_ms:.2f} ms**")

            if not candidates:
                st.warning("No candidate matches found in FAISS index above threshold.")
            else:
                c_df_list = []
                for cand in candidates:
                    bi_score = cand.get("bi_encoder_score", 0.0)
                    ce_score = cand.get("cross_encoder_score", None)
                    orig_rank = cand.get("rank", 1)
                    new_rank = cand.get("reranked_rank", orig_rank)
                    rank_shift = orig_rank - new_rank
                    rank_str = f"#{new_rank}" if enable_ce_rerank else f"#{orig_rank}"
                    shift_str = f"▲ +{rank_shift}" if rank_shift > 0 else (f"▼ {rank_shift}" if rank_shift < 0 else "➖ 0")

                    c_df_list.append({
                        "Rank": rank_str,
                        "Doc ID": cand["doc_id"],
                        "Title": cand["title"],
                        "Author": cand["author"],
                        "Bi-Encoder Score (FAISS)": f"{bi_score:.4f}",
                        "Cross-Encoder Precision": f"{ce_score:.4f}" if ce_score is not None else "N/A",
                        "Rank Shift": shift_str if enable_ce_rerank else "N/A",
                        "Essay Snippet": cand["text"][:120] + "..."
                    })

                st.markdown("### 🏆 Retrieved Candidate Rankings")
                st.dataframe(pd.DataFrame(c_df_list), use_container_width=True)

                st.markdown("---")
                st.subheader("📖 Candidate Essay Details & Comparison")
                for cand in candidates:
                    with st.expander(f"📌 [{cand['doc_id']}] {cand['title']} - Author: {cand['author']}"):
                        col_d1, col_d2 = st.columns(2)
                        col_d1.metric("FAISS Bi-Encoder Score", f"{cand['bi_encoder_score']:.4f}")
                        if "cross_encoder_score" in cand:
                            col_d2.metric("Cross-Encoder Score", f"{cand['cross_encoder_score']:.4f}", delta=f"{cand['score_diff']:+.4f}")
                        st.markdown("**Full Essay Text:**")
                        st.write(cand["text"])

    st.markdown("---")
    st.subheader("📚 Current Indexed Essay Corpus")
    with st.expander("View All Documents Currently Indexed in FAISS Vector DB"):
        corpus_df = pd.DataFrame(faiss_db.metadata)[["doc_id", "title", "author", "char_length", "text"]]
        st.dataframe(corpus_df, use_container_width=True)


# ==========================================
# MODULE 3: Two-Stage Pipeline Precision Pass
# ==========================================
elif navigation_mode == "⚡ Two-Stage Pipeline Precision Pass":
    st.header("⚡ Two-Stage Retrieval Pipeline: Bi-Encoder + Cross-Encoder Re-ranker")
    st.caption("Demonstrates why candidate retrieval requires a 2-stage architecture: Bi-Encoder scales fast for candidate filtering, and Cross-Encoder provides secondary precision pass for borderline pairs.")

    st.markdown(
        """
        ```
        [ Query Essay ] ──► [ SBERT Vector Encoding ] ──► [ FAISS Index (Top-K Candidates) ] (Stage 1: Fast Candidate Retrieval)
                                                                       │
                                                                       ▼
        [ Flagged Borderline Pairs ] ◄── [ Cross-Encoder Joint Attention Pass ] (Stage 2: Precision Re-ranking)
        ```
        """
    )

    st.subheader("🧪 Live Borderline Candidate Pair Benchmark")

    sample_test_pairs = [
        (
            "Renewable energy technologies such as solar and wind power are becoming more affordable.",
            "As battery storage technology improves, renewable energy sources will replace fossil fuels.",
            "High Similarity / Paraphrase"
        ),
        (
            "Traditional power plants burning coal and gas emit large quantities of carbon dioxide daily.",
            "Conventional facilities that rely on coal release substantial volumes of greenhouse gases continuously.",
            "Heavy Paraphrase"
        ),
        (
            "Microservices architecture decomposes a monolithic application into lightweight independent services.",
            "Monolithic software systems can be broken down into autonomous micro-services.",
            "Structural Rewrite"
        ),
        (
            "Academic integrity relies heavily on authentic submissions of assignments.",
            "Photosynthesis is the process used by plants to convert sunlight into energy.",
            "Unrelated Borderline False Positive Check"
        )
    ]

    benchmark_data = []
    with st.spinner("Computing Bi-Encoder vs Cross-Encoder joint predictions..."):
        for p1, p2, label in sample_test_pairs:
            # Stage 1: Bi-encoder cosine sim
            vec1 = embedder.encode([p1])[0]
            vec2 = embedder.encode([p2])[0]
            bi_sim = float(np.dot(vec1, vec2))

            # Stage 2: Cross-encoder precision score
            ce_sim = float(cross_encoder.predict_pair(p1, p2))

            benchmark_data.append({
                "Pair Description": label,
                "Text A": p1,
                "Text B": p2,
                "Stage 1 Bi-Encoder (SBERT)": round(bi_sim, 4),
                "Stage 2 Cross-Encoder": round(ce_sim, 4),
                "Score Shift (CE - BE)": round(ce_sim - bi_sim, 4)
            })

    df_bench = pd.DataFrame(benchmark_data)
    st.dataframe(df_bench, use_container_width=True)

    # Plot Comparison Chart
    st.markdown("### 📈 Bi-Encoder vs Cross-Encoder Score Comparison Chart")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_bench["Pair Description"],
        y=df_bench["Stage 1 Bi-Encoder (SBERT)"],
        name="Stage 1: Bi-Encoder (SBERT)",
        marker_color="#3b82f6"
    ))
    fig.add_trace(go.Bar(
        x=df_bench["Pair Description"],
        y=df_bench["Stage 2 Cross-Encoder"],
        name="Stage 2: Cross-Encoder Re-ranker",
        marker_color="#10b981"
    ))
    fig.update_layout(
        barmode='group',
        title="Bi-Encoder vs Cross-Encoder Precision Pass",
        yaxis_title="Similarity Score [0 - 1]",
        xaxis_title="Pair Category",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# MODULE 4: Instructor Batch Audit Report
# ==========================================
elif navigation_mode == "📊 Instructor Batch Audit Report":
    st.header("📊 Instructor Batch Assignment Audit & Pairwise Matrix")
    st.caption("Scan an entire batch of student essays to generate a pairwise similarity matrix, flag high-risk duplicate pairs, and export audit reports.")

    st.subheader("📁 Batch Corpus Selection")
    st.info(f"Currently loaded batch: **{faiss_db.total_documents()} Student Essays** in database.")

    flag_threshold = st.slider("Set Plagiarism Flagging Threshold:", min_value=0.30, max_value=0.90, value=0.50, step=0.05)

    if st.button("⚙️ Run Full Batch Pairwise Scan", type="primary", use_container_width=True):
        metadata = faiss_db.metadata
        n = len(metadata)

        with st.spinner(f"Computing pairwise similarity matrix for {n} student essays ({n*(n-1)//2} pairs)..."):
            texts = [m["text"] for m in metadata]
            proc_list = preprocessor.process_column(texts)

            sim_matrix = np.zeros((n, n))
            pair_records = []

            for i in range(n):
                for j in range(i + 1, n):
                    sents_i = proc_list[i].sentences
                    sents_j = proc_list[j].sentences

                    matches = align_passages(embedder, sents_i, sents_j, match_threshold=0.50)
                    summary = document_level_summary(matches, len(sents_i))
                    score = summary["overall_score"]
                    sim_matrix[i, j] = score
                    sim_matrix[j, i] = score

                    if score >= flag_threshold:
                        pair_records.append({
                            "Student A (Doc ID)": f"{metadata[i]['author']} ({metadata[i]['doc_id']})",
                            "Student B (Doc ID)": f"{metadata[j]['author']} ({metadata[j]['doc_id']})",
                            "Title A": metadata[i]["title"],
                            "Title B": metadata[j]["title"],
                            "Overall Score": f"{score*100:.1f}%",
                            "Matched Sentences": int(summary["matched_sentence_ratio"] * len(sents_i)),
                            "Status": "🚨 FLAGGED FOR REVIEW"
                        })

        st.markdown("---")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Total Essay Pairs Evaluated", f"{n*(n-1)//2}")
        m_col2.metric("Flagged High-Risk Pairs", len(pair_records))
        m_col3.metric("Flagging Threshold", f"{flag_threshold*100:.0f}%")

        # Heatmap Visualization
        st.markdown("### 🗺️ Student Essay Pairwise Similarity Heatmap")
        doc_labels = [f"{m['author']} ({m['doc_id']})" for m in metadata]
        fig_map = px.imshow(
            sim_matrix,
            x=doc_labels,
            y=doc_labels,
            color_continuous_scale="Reds",
            title="Batch Similarity Matrix Heatmap"
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # Flagged Pairs Table
        st.markdown("### 🚨 Flagged Duplicate / Paraphrased Assignments")
        if not pair_records:
            st.success("🎉 No assignment pairs exceeded the plagiarism threshold!")
        else:
            df_flagged = pd.DataFrame(pair_records)
            st.dataframe(df_flagged, use_container_width=True)

            # Export CSV
            csv_buffer = io.StringIO()
            df_flagged.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Instructor Audit Report (CSV)",
                data=csv_buffer.getvalue(),
                file_name="plagiarism_audit_report.csv",
                mime="text/csv"
            )
