import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from config import MSRP_PROCESSED_PATH, QQP_PROCESSED_PATH, EVAL_RESULTS_CSV, EVAL_PLOT_PNG
from similarity_engine import SimilarityBenchmarkEngine

def optimize_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    best_thresh, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.95, 0.005):
        pred = (scores >= t).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1, best_thresh = score, t
    return best_thresh

def evaluate_predictions(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(int)
    return {
        "Threshold": round(threshold, 3),
        "Accuracy": round(accuracy_score(y_true, preds), 4),
        "Precision": round(precision_score(y_true, preds, zero_division=0), 4),
        "Recall": round(recall_score(y_true, preds, zero_division=0), 4),
        "F1-Score": round(f1_score(y_true, preds, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_true, scores), 4)
    }

def main():
    print("[1/4] Loading Validation and Test sets...")
    has_split = False
    val_dfs, test_dfs = [], []
    
    for path in [MSRP_PROCESSED_PATH, QQP_PROCESSED_PATH]:
        if path.exists():
            df = pd.read_parquet(path)
            if "split" in df.columns:
                has_split = True
                val_dfs.append(df[df["split"] == "validation"])
                test_dfs.append(df[df["split"] == "test"])

    if not has_split or len(val_dfs) == 0:
        print("  -> Loading fresh splits directly via data_loader...")
        from data_loader import load_msrp, load_qqp
        msrp_df = load_msrp()
        qqp_df = load_qqp()
        val_df = pd.concat([msrp_df[msrp_df["split"] == "validation"], qqp_df[qqp_df["split"] == "validation"]], ignore_index=True)
        test_df = pd.concat([msrp_df[msrp_df["split"] == "test"], qqp_df[qqp_df["split"] == "test"]], ignore_index=True)
    else:
        val_df = pd.concat(val_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)

    print(f"  -> Validation pairs: {len(val_df)}, Test pairs: {len(test_df)}")

    col1_v = "text1" if "text1" in val_df.columns else "text_a"
    col2_v = "text2" if "text2" in val_df.columns else "text_b"
    col1_t = "text1" if "text1" in test_df.columns else "text_a"
    col2_t = "text2" if "text2" in test_df.columns else "text_b"

    print("[2/4] Initializing Benchmark Engine (TF-IDF, Word2Vec, Hybrid SBERT, Cross-Encoder)...")
    engine = SimilarityBenchmarkEngine()
    engine.fit_tfidf(val_df[col1_v].tolist() + val_df[col2_v].tolist())

    models = {
        "TF-IDF": (engine.compute_tfidf_similarity, "Baseline 1"),
        "Word2Vec": (engine.compute_w2v_similarity, "Baseline 2"),
        "Hybrid SBERT": (engine.compute_hybrid_similarity, "Proposed Model"),
        "Cross-Encoder": (engine.compute_cross_encoder_similarity, "2-Stage Re-ranker")
    }

    results = []
    print("[3/4] Tuning thresholds on Validation Split & Evaluating on Held-Out Test Split...")
    for model_name, (scorer, model_type) in models.items():
        try:
            print(f"\n--- Evaluating {model_name} ({model_type}) ---")
            val_scores = scorer(val_df[col1_v].tolist(), val_df[col2_v].tolist())
            val_scores = np.clip(np.nan_to_num(val_scores, nan=0.0), 0.0, 1.0)
            best_t = optimize_threshold(val_df["label"].to_numpy(), val_scores)

            test_scores = scorer(test_df[col1_t].tolist(), test_df[col2_t].tolist())
            test_scores = np.clip(np.nan_to_num(test_scores, nan=0.0), 0.0, 1.0)
            metrics = evaluate_predictions(test_df["label"].to_numpy(), test_scores, best_t)
            metrics["Model"] = model_name
            metrics["Type"] = model_type
            results.append(metrics)
            print(f"  -> {model_name} evaluation complete. F1={metrics['F1-Score']:.4f}")
        except Exception as e:
            print(f"  -> WARNING: {model_name} evaluation failed: {e}")
            results.append({
                "Type": model_type,
                "Model": model_name,
                "Threshold": 0.0,
                "Accuracy": 0.0,
                "Precision": 0.0,
                "Recall": 0.0,
                "F1-Score": 0.0,
                "ROC-AUC": 0.0,
            })

    res_df = pd.DataFrame(results)[["Type", "Model", "Threshold", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]]
    res_df.to_csv(EVAL_RESULTS_CSV, index=False)
    print("\n" + res_df.to_string(index=False) + "\n")

    print(f"[4/4] Generating comparative visualization at {EVAL_PLOT_PNG}...")
    plt.figure(figsize=(10, 5))
    bars = plt.bar(res_df["Model"], res_df["F1-Score"], color=["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"], width=0.55)
    plt.ylabel("F1 Score")
    plt.title("Performance Comparison: Multi-Tier Model Architecture")
    plt.ylim(0, 1.05)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(EVAL_PLOT_PNG, dpi=300)
    print(f"Artifacts successfully saved to {EVAL_RESULTS_CSV} and {EVAL_PLOT_PNG}")

if __name__ == "__main__":
    main()