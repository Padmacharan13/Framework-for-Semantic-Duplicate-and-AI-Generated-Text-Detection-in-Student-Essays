"""
evaluate.py
Benchmarks 3 models for duplicate/paraphrase detection on MSRP+QQP:
  1. TF-IDF Cosine Similarity (Lexical Baseline)
  2. Word2Vec Dense Averaged Embeddings (Dense Word Vector Baseline)
  3. SBERT (Optimized MPNet Transformer) (Proposed Contextual Model)

For each model: sweeps similarity thresholds, picks the threshold maximizing F1,
and computes Precision, Recall, F1, and Accuracy.

Outputs:
  - results/evaluation_results.csv   (3-model metrics summary)
  - results/model_comparison_f1.png  (3-bar comparison chart)

Run:
    python evaluate.py
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")  # headless-safe backend, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from config import EVAL_PLOT_FILE, EVAL_RESULTS_CSV
from data_loader import load_all_benchmark_data
from embeddings import Embedder
from similarity_engine import sbert_pair_similarity, tfidf_pair_similarity, word2vec_pair_similarity
from word2vec_engine import Word2VecEmbedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def best_threshold_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Sweeps thresholds 0.05-0.95 and returns metrics at the F1-maximizing cutoff."""
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0}
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best["f1"]:
            best = {
                "threshold": round(float(t), 2),
                "f1": round(float(f1), 4),
                "precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
                "recall": round(float(recall_score(labels, preds, zero_division=0)), 4),
                "accuracy": round(float(accuracy_score(labels, preds)), 4),
            }
    return best


def run() -> None:
    df = load_all_benchmark_data()
    text_a = df["text_a"].tolist()
    text_b = df["text_b"].tolist()
    labels = df["label"].to_numpy()

    logger.info("1/3 Scoring %d pairs with TF-IDF baseline...", len(df))
    tfidf_scores = tfidf_pair_similarity(text_a, text_b)

    logger.info("2/3 Scoring %d pairs with Word2Vec dense embedding model...", len(df))
    all_texts = text_a + text_b
    w2v_embedder = Word2VecEmbedder(train_corpus=all_texts)
    w2v_scores = word2vec_pair_similarity(w2v_embedder, text_a, text_b)

    logger.info("3/3 Scoring %d pairs with Optimized SBERT (MPNet)...", len(df))
    sbert_embedder = Embedder()
    sbert_scores = sbert_pair_similarity(sbert_embedder, text_a, text_b)

    logger.info("Sweeping thresholds and calculating performance metrics...")
    tfidf_metrics = best_threshold_metrics(tfidf_scores, labels)
    w2v_metrics = best_threshold_metrics(w2v_scores, labels)
    sbert_metrics = best_threshold_metrics(sbert_scores, labels)

    results = pd.DataFrame([
        {"method": "TF-IDF (Lexical Baseline)", **tfidf_metrics},
        {"method": "Word2Vec (Dense Baseline)", **w2v_metrics},
        {"method": "SBERT (Optimized MPNet)", **sbert_metrics},
    ])
    results.to_csv(EVAL_RESULTS_CSV, index=False)
    logger.info("Saved 3-way evaluation metrics table -> %s", EVAL_RESULTS_CSV)
    print("\n=== MODEL PERFORMANCE COMPARISON ===")
    print(results.to_string(index=False))

    # ---- 3-Way Model Comparison Plot ----
    fig, ax = plt.subplots(figsize=(9, 6))
    metrics_to_plot = ["precision", "recall", "f1", "accuracy"]
    x = np.arange(len(metrics_to_plot))
    width = 0.25

    tfidf_vals = [tfidf_metrics[m] for m in metrics_to_plot]
    w2v_vals = [w2v_metrics[m] for m in metrics_to_plot]
    sbert_vals = [sbert_metrics[m] for m in metrics_to_plot]

    ax.bar(x - width, tfidf_vals, width, label="TF-IDF (Lexical Baseline)", color="#64748b")
    ax.bar(x, w2v_vals, width, label="Word2Vec (Dense Baseline)", color="#f59e0b")
    ax.bar(x + width, sbert_vals, width, label="SBERT (Optimized MPNet)", color="#6366f1")

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics_to_plot], fontweight="bold", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title(f"Model Comparison on Benchmark Dataset (MSRP+QQP, n={len(df)})", fontweight="bold", fontsize=12)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Add value annotations above bars
    for i in range(len(metrics_to_plot)):
        ax.text(i - width, tfidf_vals[i] + 0.01, f"{tfidf_vals[i]:.2f}", ha="center", fontsize=8)
        ax.text(i, w2v_vals[i] + 0.01, f"{w2v_vals[i]:.2f}", ha="center", fontsize=8)
        ax.text(i + width, sbert_vals[i] + 0.01, f"{sbert_vals[i]:.2f}", ha="center", fontsize=8, fontweight="bold")

    fig.tight_layout()
    fig.savefig(EVAL_PLOT_FILE, dpi=150)
    logger.info("Saved 3-way model comparison chart -> %s", EVAL_PLOT_FILE)


if __name__ == "__main__":
    run()