"""
train_ai_detector.py
End-to-end training and calibration script for the AI Detection System:
  1. Train Transformer on Split A (ai_detection_train.parquet)
  2. Fit and Calibrate Ensemble Meta-Classifier on Split B (ai_detection_validation.parquet)
  3. Learn Data-Driven Sentence & Perplexity Thresholds on Split B
  4. Evaluate on Split C (ai_detection_test.parquet) and Save Real Metrics & Calibration Curves
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from semantic_integrity.ai_detection.classifier import TransformerAIDetector
from semantic_integrity.ai_detection.ensemble import EnsembleAIDetector
from semantic_integrity.ai_detection.features import fit_and_save_thresholds
from semantic_integrity.config import (
    AI_DETECTOR_DIR,
    AI_DETECTOR_ENSEMBLE_PATH,
    AI_DETECTION_TEST_PATH,
    AI_DETECTION_TRAIN_PATH,
    AI_DETECTOR_TRANSFORMER_PATH,
    AI_DETECTION_VAL_PATH,
    RANDOM_SEED,
    RESULTS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    AI_DETECTOR_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STAGE 3: AI DETECTOR TRAINING, CALIBRATION & EVALUATION")
    print("=" * 70)

    # 1. Load Grouped Splits
    print("\n[1/5] Loading Grouped Dataset Splits...")
    train_df = pd.read_parquet(AI_DETECTION_TRAIN_PATH)
    val_df = pd.read_parquet(AI_DETECTION_VAL_PATH)
    test_df = pd.read_parquet(AI_DETECTION_TEST_PATH)

    print(f"  -> Split A (Train):      {len(train_df)} essays ({train_df['group_id'].nunique()} groups)")
    print(f"  -> Split B (Validation): {len(val_df)} essays ({val_df['group_id'].nunique()} groups)")
    print(f"  -> Split C (Test):       {len(test_df)} essays ({test_df['group_id'].nunique()} groups)")

    # Assert no group leakage between any pair of splits
    train_grps = set(train_df["group_id"].unique())
    val_grps = set(val_df["group_id"].unique())
    test_grps = set(test_df["group_id"].unique())

    assert len(train_grps.intersection(val_grps)) == 0, "LEAKAGE: Train & Validation overlap!"
    assert len(train_grps.intersection(test_grps)) == 0, "LEAKAGE: Train & Test overlap!"
    assert len(val_grps.intersection(test_grps)) == 0, "LEAKAGE: Validation & Test overlap!"
    print("  -> Verified: 0 group leakage across Splits A, B, and C.")

    # 2. Fine-tune Transformer on Split A
    print("\n[2/5] Fine-Tuning Transformer on Split A...")
    # For reliable, responsive training on CPU, sample balanced split A
    n_per_class = min((train_df["label"] == 0).sum(), (train_df["label"] == 1).sum(), 250)
    h_df = train_df[train_df["label"] == 0].sample(n_per_class, random_state=RANDOM_SEED)
    ai_df = train_df[train_df["label"] == 1].sample(n_per_class, random_state=RANDOM_SEED)
    train_sample = pd.concat([h_df, ai_df], ignore_index=True).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    transformer = TransformerAIDetector(
        model_name_or_path="distilroberta-base",
        max_length=512,
    )

    val_sample_size = min(len(val_df), 150)
    val_sample = val_df.sample(val_sample_size, random_state=RANDOM_SEED).reset_index(drop=True)

    transformer.train_model(
        train_texts=train_sample["text"].tolist(),
        train_labels=train_sample["label"].tolist(),
        val_texts=val_sample["text"].tolist(),
        val_labels=val_sample["label"].tolist(),
        epochs=2,
        batch_size=16,
        lr=3e-5,
        output_dir=AI_DETECTOR_TRANSFORMER_PATH,
    )
    print(f"  -> Transformer fine-tuning complete. Saved to: {AI_DETECTOR_TRANSFORMER_PATH}")

    # 3. Learn Thresholds on Split B
    print("\n[3/5] Learning Data-Driven Thresholds on Split B (Validation)...")
    ensemble = EnsembleAIDetector(meta_model_path=AI_DETECTOR_ENSEMBLE_PATH)
    # Ensure detector points to freshly trained transformer
    ensemble._transformer = TransformerAIDetector(model_name_or_path=AI_DETECTOR_TRANSFORMER_PATH)

    threshold_sample = val_df.sample(min(len(val_df), 120), random_state=RANDOM_SEED).reset_index(drop=True)
    learned_thresholds = fit_and_save_thresholds(
        val_texts=threshold_sample["text"].tolist(),
        val_labels=threshold_sample["label"].tolist(),
        feature_extractor=ensemble.feature_extractor,
    )
    print(f"  -> Learned dynamic thresholds: {learned_thresholds}")

    # 4. Fit & Calibrate Meta-Classifier on Split B
    print("\n[4/5] Fitting & Calibrating Meta-Classifier on Split B...")
    val_fit_df = val_df.sample(min(len(val_df), 250), random_state=RANDOM_SEED).reset_index(drop=True)
    ensemble.fit(
        val_texts=val_fit_df["text"].tolist(),
        val_labels=val_fit_df["label"].tolist(),
        save=True,
        calibration_method="sigmoid",
    )

    # 5. Evaluate on Split C (Held-Out Test Split)
    print("\n[5/5] Evaluating Calibrated Ensemble on Split C (Held-Out Test Set)...")
    # Evaluate the full held-out test split; do not report a small random
    # subset as the test-set result.
    test_eval_df = test_df.reset_index(drop=True)

    y_test = test_eval_df["label"].to_numpy()
    probs = ensemble.predict_proba(test_eval_df["text"].tolist())
    ai_probs = probs[:, 1]
    preds = (ai_probs >= 0.5).astype(int)

    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, ai_probs))
    brier = float(brier_score_loss(y_test, ai_probs))

    metrics = {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4),
        "Brier_Score": round(brier, 4),
        "Threshold": 0.5,
        "Test_Samples": len(test_eval_df),
        "Human_Samples": int((y_test == 0).sum()),
        "AI_Samples": int((y_test == 1).sum()),
        "Learned_Thresholds": learned_thresholds,
    }

    metrics_path = RESULTS_DIR / "ai_detector_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Reliability Diagram (Calibration Curve)
    prob_true, prob_pred = calibration_curve(y_test, ai_probs, n_bins=10)
    calib_plot_path = RESULTS_DIR / "calibration_curve.png"

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.plot(prob_pred, prob_true, marker="o", color="#2563eb", linewidth=2, label=f"Ensemble (Brier={brier:.4f})")
    plt.xlabel("Mean Predicted Probability", fontsize=11)
    plt.ylabel("Empirical Fraction of Positives", fontsize=11)
    plt.title("Reliability Diagram: Calibrated AI Detector", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(calib_plot_path, dpi=300)
    plt.close()

    print("\n" + "=" * 70)
    print("REAL TEST SET METRICS (CALIBRATED ENSEMBLE AI DETECTOR)")
    print("=" * 70)
    for k, v in metrics.items():
        if k != "Learned_Thresholds":
            print(f"  {k:15s}: {v}")
    print(f"\nArtifacts saved to:\n  - {metrics_path}\n  - {calib_plot_path}")


if __name__ == "__main__":
    main()
