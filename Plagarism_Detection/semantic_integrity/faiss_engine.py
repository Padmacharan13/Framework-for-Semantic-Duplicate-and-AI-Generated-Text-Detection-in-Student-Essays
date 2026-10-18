"""
faiss_engine.py
FAISS Vector Indexing & Fast Candidate Retrieval Engine.
Provides high-performance vector search over large essay corpora using L2-normalized SBERT embeddings.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from .config import FAISS_INDEX_PATH, FAISS_METADATA_PATH, FAISS_TOP_K

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


SAMPLE_STUDENT_CORPUS = [
    {
        "doc_id": "ESSAY_001",
        "title": "Impact of Renewable Energy Technologies",
        "author": "Alice Chen",
        "text": (
            "Renewable energy sources such as solar and wind power are becoming significantly more affordable with each passing year. "
            "As battery storage technology continues to improve, these renewable sources are expected to replace a large share of fossil fuel generation. "
            "Traditional power plants that burn coal and natural gas emit large amounts of carbon dioxide into the atmosphere every single day. "
            "Governments across the world are now offering tax incentives to homeowners who choose to install solar panels on their rooftops."
        )
    },
    {
        "doc_id": "ESSAY_002",
        "title": "Clean Power & Energy Transition (Paraphrased Variant)",
        "author": "Bob Smith",
        "text": (
            "Renewable energy sources such as solar and wind power are becoming significantly more affordable with each passing year. "
            "As storage battery technology keeps improving, these renewable sources are projected to replace a significant portion of fossil fuel generation. "
            "Conventional facilities that rely on burning coal and natural gas release substantial volumes of harmful greenhouse gases into the atmosphere continuously. "
            "Tax incentives are increasingly being provided by governments worldwide to encourage residents to add solar panels on top of their homes."
        )
    },
    {
        "doc_id": "ESSAY_003",
        "title": "Artificial Intelligence in Higher Education",
        "author": "Charlie Davis",
        "text": (
            "Artificial Intelligence is reshaping higher education by enabling personalized learning paths for students. "
            "Adaptive algorithms analyze historical student data to recommend targeted reading materials and tailored practice exercises. "
            "However, educational institutions must establish strict policies regarding academic honesty and proper citation."
        )
    },
    {
        "doc_id": "ESSAY_004",
        "title": "AI in Academia & Academic Integrity (Partial Copy)",
        "author": "Diana Prince",
        "text": (
            "Artificial Intelligence is reshaping higher education by enabling personalized learning paths for students. "
            "Adaptive algorithms analyze historical student data to recommend targeted reading materials and tailored practice exercises. "
            "Students can submit their assignments online through a centralized portal for automated feedback."
        )
    },
    {
        "doc_id": "ESSAY_005",
        "title": "Microservices Architecture in Distributed Systems",
        "author": "Evan Wright",
        "text": (
            "The microservices architecture decomposes a large monolithic application into small, loosely coupled services. "
            "Each service runs in its own process, communicates via lightweight REST APIs, and can be deployed independently without affecting the broader system."
        )
    },
    {
        "doc_id": "ESSAY_006",
        "title": "Modern Microservices Design Patterns (Structural Rewrite)",
        "author": "Fiona Gallagher",
        "text": (
            "Monolithic software systems can be broken down into autonomous micro-services. "
            "These individual services execute as independent OS processes, exchange messages using REST APIs, and allow separate deployment pipelines that minimize system-wide blast radius."
        )
    },
    {
        "doc_id": "ESSAY_007",
        "title": "Quantum Computing & Quantum Entanglement",
        "author": "George Clark",
        "text": (
            "Quantum computing leverages quantum mechanics principles such as superposition and entanglement to perform complex computations. "
            "Quantum bits or qubits can represent multiple states simultaneously, offering exponential speedups for specialized mathematical algorithms."
        )
    },
    {
        "doc_id": "ESSAY_008",
        "title": "Plant Photosynthesis & Biological Processes",
        "author": "Hannah Abbott",
        "text": (
            "Photosynthesis is the biological process used by plants, algae, and certain bacteria to convert light energy into chemical energy. "
            "Chlorophyll absorbs sunlight and drives the reaction between carbon dioxide and water to produce glucose and oxygen gas."
        )
    },
    {
        "doc_id": "ESSAY_009",
        "title": "Database Indexing & Query Optimization",
        "author": "Ian Malcolm",
        "text": (
            "B-Tree indexes and Hash indexes accelerate search operations in relational databases. "
            "By creating appropriate index structures on frequently filtered columns, execution engine costs for complex JOIN queries are reduced significantly."
        )
    },
    {
        "doc_id": "ESSAY_010",
        "title": "Ethics & Governance in Machine Learning",
        "author": "Julia Roberts",
        "text": (
            "Automated decision-making systems must adhere to strict principles of fairness, accountability, and transparency. "
            "Algorithmic bias can emerge when training datasets contain historical disparities or non-representative demographic samples."
        )
    }
]


class FAISSVectorIndex:
    """
    FAISS-backed Dense Vector Index for fast candidate retrieval across a corpus.
    Uses Inner Product (IndexFlatIP) on L2-normalized embeddings, equivalent to Cosine Similarity.
    """

    def __init__(self, embedding_dim: int = 384) -> None:
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.metadata: list[dict[str, Any]] = []

    def is_built(self) -> bool:
        return self.index is not None and self.index.ntotal > 0

    def total_documents(self) -> int:
        return self.index.ntotal if self.index else 0

    def build_corpus_index(
        self,
        documents: list[dict[str, Any]],
        embedder: Any,
        batch_size: int = 64
    ) -> int:
        """
        Encodes a collection of documents and indexes them in FAISS.
        Auto-detects embedding dimension from the model's output.
        """
        if not documents:
            logger.warning("Empty document list provided to build_corpus_index.")
            return 0

        logger.info("Building FAISS vector index for %d documents...", len(documents))
        texts = [doc.get("text", "") for doc in documents]
        embeddings = embedder.encode(texts, batch_size=batch_size, show_progress=False)

        embeddings_np = np.asarray(embeddings, dtype=np.float32)

        # Auto-detect embedding dimension from actual model output
        actual_dim = embeddings_np.shape[1]
        if actual_dim != self.embedding_dim:
            logger.info("Auto-adjusting FAISS index dimension from %d to %d.", self.embedding_dim, actual_dim)
            self.embedding_dim = actual_dim

        # Ensure L2 normalization for Inner Product cosine similarity
        faiss.normalize_L2(embeddings_np)

        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings_np)
        self.metadata = [
            {
                "doc_id": doc.get("doc_id", f"DOC_{i:03d}"),
                "title": doc.get("title", f"Document {i+1}"),
                "author": doc.get("author", "Unknown"),
                "text": doc.get("text", ""),
                "char_length": len(doc.get("text", "")),
            }
            for i, doc in enumerate(documents)
        ]
        logger.info("Successfully indexed %d documents into FAISS (dim=%d).", self.index.ntotal, self.embedding_dim)
        return self.index.ntotal

    def search_candidates(
        self,
        query_text: str,
        embedder: Any,
        top_k: int = FAISS_TOP_K,
        min_score: float = 0.0
    ) -> list[dict[str, Any]]:
        """
        Searches FAISS index for top_k documents most similar to query_text.
        Returns a list of candidate dictionaries containing similarity score, rank, and doc metadata.
        """
        if not self.is_built():
            logger.warning("FAISS index is empty or not built.")
            return []

        query_vec = embedder.encode([query_text], show_progress=False)
        query_np = np.asarray(query_vec, dtype=np.float32)
        faiss.normalize_L2(query_np)

        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_np, k)

        candidates = []
        for rank, (score, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self.metadata):
                continue
            sim_score = float(score)
            if sim_score < min_score:
                continue

            meta = self.metadata[idx]
            candidates.append({
                "rank": rank,
                "doc_id": meta["doc_id"],
                "title": meta["title"],
                "author": meta["author"],
                "text": meta["text"],
                "bi_encoder_score": round(sim_score, 4),
                "bi_encoder_similarity": round(sim_score, 4),
                "index": int(idx)
            })

        return candidates

    def save(self, index_path: str | Path = FAISS_INDEX_PATH, metadata_path: str | Path = FAISS_METADATA_PATH) -> None:
        """
        Saves the FAISS index binary and metadata JSON to disk.
        """
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info("Saved FAISS index to '%s' and metadata to '%s'.", index_path, metadata_path)

    def load(self, index_path: str | Path = FAISS_INDEX_PATH, metadata_path: str | Path = FAISS_METADATA_PATH) -> bool:
        """
        Loads FAISS index binary and metadata JSON from disk if present.
        """
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        if not index_path.exists() or not metadata_path.exists():
            logger.info("FAISS index or metadata file not found at '%s'.", index_path)
            return False

        try:
            self.index = faiss.read_index(str(index_path))
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            logger.info("Successfully loaded FAISS index (%d items).", self.index.ntotal)
            return True
        except Exception as e:
            logger.error("Failed to load FAISS index: %s", e)
            return False


if __name__ == "__main__":
    from embeddings import Embedder

    embedder = Embedder()
    faiss_db = FAISSVectorIndex()
    faiss_db.build_corpus_index(SAMPLE_STUDENT_CORPUS, embedder)

    query = "Renewable power generation with solar panels is getting much cheaper."
    results = faiss_db.search_candidates(query, embedder, top_k=3)

    print(f"\nQuery: '{query}'")
    print("FAISS Candidate Retrieval Results:")
    for res in results:
        print(f"  Rank {res['rank']} | ID: {res['doc_id']} | Score: {res['bi_encoder_score']:.4f} | Title: {res['title']}")
