"""
semantic_integrity/ai_detection/ensemble.py

Ensemble AI Detection Module.
Combines Language Model perplexity/burstiness features with Transformer
deep representations via a calibrated meta-classifier, providing high-accuracy
AI-generated text detection and sentence-level interpretability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from semantic_integrity.ai_detection.classifier import TransformerAIDetector
from semantic_integrity.ai_detection.features import (
    LMFeatureExtractor,
    load_learned_thresholds,
    split_into_sentences,
)
from semantic_integrity.config import (
    AI_DETECTOR_DIR,
    AI_DETECTOR_ENSEMBLE_PATH,
    AI_DETECTOR_THRESHOLD,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)


def get_verdict(prob: float) -> Tuple[str, str]:
    """
    Map AI probability to human-readable verdict and risk level.
    """
    if prob >= 0.85:
        return "Highly Confident AI-Generated", "HIGH"
    elif prob >= 0.60:
        return "Likely AI-Generated", "HIGH"
    elif prob >= 0.40:
        return "Uncertain / Mixed Signals", "MEDIUM"
    elif prob >= 0.20:
        return "Likely Human-Authored", "LOW"
    else:
        return "Confident Human-Authored", "LOW"


class EnsembleAIDetector:
    """
    Calibrated Ensemble classifier combining:
      1. GPT-2 Perplexity and Burstiness Features
      2. RoBERTa Sequence Classification Probabilities (Chunk-Averaged)
      3. Stylometric & Lexical Diversity Metrics
    """

    FEATURE_NAMES = [
        "transformer_ai_prob",
        "perplexity",
        "log_perplexity",
        "burstiness_ppl_var",
        "burstiness_ppl_std",
        "burstiness_cv",
        "sentence_len_mean",
        "sentence_len_std",
        "sentence_count",
        "ttr",
        "avg_word_len",
    ]

    def __init__(
        self,
        meta_model_path: Optional[Union[str, Path]] = None,
        threshold: float = AI_DETECTOR_THRESHOLD,
    ):
        self.meta_model_path = Path(meta_model_path or AI_DETECTOR_ENSEMBLE_PATH)
        self.threshold = threshold

        # Lazy components
        self._feature_extractor: Optional[LMFeatureExtractor] = None
        self._transformer: Optional[TransformerAIDetector] = None
        self.meta_pipeline: Optional[Any] = None

        if self.meta_model_path.exists():
            try:
                self.load(self.meta_model_path)
            except Exception as e:
                logger.warning("Failed to load existing ensemble meta-model: %s", e)

    @property
    def feature_extractor(self) -> LMFeatureExtractor:
        if self._feature_extractor is None:
            self._feature_extractor = LMFeatureExtractor()
        return self._feature_extractor

    @property
    def transformer(self) -> TransformerAIDetector:
        if self._transformer is None:
            self._transformer = TransformerAIDetector()
        return self._transformer

    def is_trained(self) -> bool:
        """Check if both the fine-tuned transformer and calibrated meta-model exist."""
        return self.meta_pipeline is not None and self.transformer.is_checkpoint_trained()

    def extract_combined_features(
        self,
        texts: Union[str, List[str]],
        allow_untrained: bool = False,
    ) -> np.ndarray:
        """
        Extract feature matrix of shape (N, num_features) combining transformer
        probabilities and LM perplexity/burstiness features.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty((0, len(self.FEATURE_NAMES)), dtype=np.float32)

        # 1. Transformer probabilities (sliding-window chunk scored)
        try:
            trans_probs = self.transformer.predict_proba(texts, allow_untrained=allow_untrained)[:, 1]
        except Exception as e:
            if not allow_untrained:
                raise RuntimeError(f"Cannot extract ensemble features without trained transformer: {e}") from e
            logger.warning("Transformer feature fallback: %s", e)
            trans_probs = np.full(len(texts), 0.5, dtype=np.float32)

        # 2. LM Features
        rows: List[List[float]] = []
        for i, text in enumerate(texts):
            feat_dict = self.feature_extractor.extract_features(text)
            row = [
                float(trans_probs[i]),
                feat_dict["perplexity"],
                feat_dict["log_perplexity"],
                feat_dict["burstiness_ppl_var"],
                feat_dict["burstiness_ppl_std"],
                feat_dict["burstiness_cv"],
                feat_dict["sentence_len_mean"],
                feat_dict["sentence_len_std"],
                feat_dict["sentence_count"],
                feat_dict["ttr"],
                feat_dict["avg_word_len"],
            ]
            rows.append(row)

        return np.array(rows, dtype=np.float32)

    def fit(
        self,
        val_texts: List[str],
        val_labels: List[int],
        save: bool = True,
        calibration_method: str = "sigmoid",
    ) -> "EnsembleAIDetector":
        """
        Fit the ensemble meta-classifier on Split B (validation texts & labels)
        using Platt calibration (CalibratedClassifierCV with sigmoid).
        """
        logger.info("Extracting ensemble features for %d Split B samples...", len(val_texts))
        X = self.extract_combined_features(val_texts, allow_untrained=False)
        y = np.array(val_labels, dtype=int)

        base_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=RANDOM_SEED,
                class_weight="balanced",
            )),
        ])

        # Platt scaling / Isotonic calibration
        calibrated = CalibratedClassifierCV(
            estimator=base_pipeline,
            method=calibration_method,
            cv=3,
        )
        calibrated.fit(X, y)
        self.meta_pipeline = calibrated

        # Calculate Brier calibration score on training split
        probs = calibrated.predict_proba(X)[:, 1]
        brier = float(brier_score_loss(y, probs))
        logger.info("Ensemble meta-classifier successfully fitted. Brier score on validation: %.4f", brier)

        if save:
            self.save(self.meta_model_path)

        return self

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Serialize meta pipeline to disk."""
        target = Path(path or self.meta_model_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.meta_pipeline, target)
        logger.info("Saved ensemble meta-model to %s", target)
        return target

    def load(self, path: Optional[Union[str, Path]] = None) -> None:
        """Load meta pipeline from disk."""
        target = Path(path or self.meta_model_path)
        if not target.exists():
            raise FileNotFoundError(f"Meta model not found at {target}")
        self.meta_pipeline = joblib.load(target)
        logger.info("Loaded ensemble meta-model from %s", target)

    def predict_proba(
        self,
        texts: Union[str, List[str]],
        allow_untrained: bool = False,
    ) -> np.ndarray:
        """
        Compute calibrated AI probabilities.
        Returns array of shape (N, 2).
        """
        if isinstance(texts, str):
            texts = [texts]

        if not self.is_trained():
            if not allow_untrained:
                raise RuntimeError(
                    "EnsembleAIDetector is untrained or missing checkpoints. "
                    "Run `python train_ai_detector.py` to train and calibrate the ensemble before predicting."
                )
            return np.full((len(texts), 2), 0.5, dtype=np.float32)

        X = self.extract_combined_features(texts, allow_untrained=allow_untrained)
        return self.meta_pipeline.predict_proba(X)

    def predict_essay(
        self,
        text: str,
        allow_untrained: bool = True,
    ) -> Dict[str, Any]:
        """
        Main end-to-end prediction API for a single essay.
        Returns comprehensive analysis with dynamically learned sentence-level highlights.
        """
        text = text.strip()
        if not text:
            return {
                "mode": "empty",
                "is_ai": False,
                "ai_probability": 0.0,
                "human_probability": 1.0,
                "confidence": 0.0,
                "verdict": "Empty Input",
                "risk_level": "LOW",
                "metrics": {},
                "sentence_analysis": [],
                "explanation": "No text provided for analysis.",
            }

        if not self.is_trained():
            if not allow_untrained:
                raise RuntimeError(
                    "EnsembleAIDetector has no fine-tuned transformer or meta-model checkpoint. "
                    "Run `python train_ai_detector.py` to train the models."
                )
            # Return explicit mode="untrained" without silent arbitrary scores
            return {
                "mode": "untrained",
                "is_ai": False,
                "ai_probability": None,
                "human_probability": None,
                "confidence": 0.0,
                "verdict": "Model Untrained",
                "risk_level": "UNKNOWN",
                "metrics": {},
                "sentence_analysis": [],
                "explanation": (
                    "AI Detection models have not yet been fine-tuned on this environment. "
                    "Please run `python train_ai_detector.py` to initialize weights and calibrate thresholds."
                ),
            }

        # 1. Ensemble calibrated probability
        probs = self.predict_proba(text, allow_untrained=False)[0]
        human_prob = float(probs[0])
        ai_prob = float(probs[1])
        is_ai = bool(ai_prob >= self.threshold)
        confidence = float(abs(ai_prob - 0.5) * 2.0)
        verdict, risk_level = get_verdict(ai_prob)

        # 2. Extract metrics & dynamic threshold tagging
        lm_features = self.feature_extractor.extract_features(text)
        sentence_details = self.feature_extractor.compute_sentence_perplexities(text)
        thresholds = load_learned_thresholds()

        sent_ai_thresh = thresholds.get("sentence_ai_ppl", 25.0)
        sent_human_thresh = thresholds.get("sentence_human_ppl", 60.0)
        doc_ppl_cutoff = thresholds.get("doc_ppl_cutoff", 35.0)
        doc_burst_cutoff = thresholds.get("doc_burst_cv_cutoff", 0.45)

        # Sentence tagging for UI visualization
        sentence_analysis = []
        for item in sentence_details:
            s_ppl = item["perplexity"]
            if s_ppl < sent_ai_thresh:
                s_status = "Likely AI"
                s_color = "#f87171"  # red
            elif s_ppl > sent_human_thresh:
                s_status = "Likely Human"
                s_color = "#4ade80"  # green
            else:
                s_status = "Neutral / Mixed"
                s_color = "#facc15"  # yellow

            sentence_analysis.append({
                "sentence": item["sentence"],
                "perplexity": round(s_ppl, 2),
                "word_count": item["word_count"],
                "status": s_status,
                "color": s_color,
            })

        # 3. Generate data-grounded rationale
        explanation_parts = []
        if ai_prob >= self.threshold:
            explanation_parts.append(
                f"The essay exhibits stylistic patterns strongly characteristic of machine generation (calibrated AI score: {ai_prob:.1%})."
            )
            if lm_features["perplexity"] < doc_ppl_cutoff:
                explanation_parts.append(
                    f"Perplexity ({lm_features['perplexity']:.1f}) is below the validation threshold ({doc_ppl_cutoff:.1f}), showing high word predictability."
                )
            if lm_features["burstiness_cv"] < doc_burst_cutoff:
                explanation_parts.append(
                    f"Sentence burstiness ({lm_features['burstiness_cv']:.2f}) indicates synthetic structural uniformity."
                )
        else:
            explanation_parts.append(
                f"The text exhibits variance consistent with human authorship (human confidence: {human_prob:.1%})."
            )
            if lm_features["perplexity"] >= doc_ppl_cutoff:
                explanation_parts.append(
                    f"Perplexity ({lm_features['perplexity']:.1f}) shows natural human vocabulary diversity."
                )
            if lm_features["burstiness_cv"] >= doc_burst_cutoff:
                explanation_parts.append(
                    f"Burstiness variance ({lm_features['burstiness_cv']:.2f}) reflects authentic cadence and rhythm."
                )

        return {
            "mode": "trained",
            "is_ai": is_ai,
            "ai_probability": round(ai_prob, 4),
            "human_probability": round(human_prob, 4),
            "confidence": round(confidence, 4),
            "verdict": verdict,
            "risk_level": risk_level,
            "metrics": {
                "perplexity": round(lm_features["perplexity"], 2),
                "burstiness_std": round(lm_features["burstiness_ppl_std"], 2),
                "burstiness_cv": round(lm_features["burstiness_cv"], 2),
                "lexical_diversity_ttr": round(lm_features["ttr"], 3),
                "sentence_count": int(lm_features["sentence_count"]),
                "avg_sentence_len": round(lm_features["sentence_len_mean"], 1),
                "learned_thresholds": thresholds,
            },
            "sentence_analysis": sentence_analysis,
            "explanation": " ".join(explanation_parts),
        }


# Global singleton instance for easy import and call
_global_ensemble: Optional[EnsembleAIDetector] = None


def get_detector() -> EnsembleAIDetector:
    """Return singleton instance of EnsembleAIDetector."""
    global _global_ensemble
    if _global_ensemble is None:
        _global_ensemble = EnsembleAIDetector()
    return _global_ensemble


def predict(text: str) -> Dict[str, Any]:
    """
    Convenience function: Run AI detection on a given essay text.
    """
    return get_detector().predict_essay(text)
