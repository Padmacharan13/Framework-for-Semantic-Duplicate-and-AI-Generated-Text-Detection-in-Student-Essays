"""
semantic_integrity/ai_detection/features.py

Extracts language model perplexity, burstiness, and stylometric features
used for AI-generated text detection.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from semantic_integrity.config import AI_DETECTOR_LM_NAME

logger = logging.getLogger(__name__)


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into clean, non-empty sentences using regex.
    """
    # Split on punctuation followed by whitespace or end of string
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 10]
    if not sentences and text.strip():
        sentences = [text.strip()]
    return sentences


def compute_ttr(text: str) -> float:
    """
    Compute Type-Token Ratio (lexical diversity: unique tokens / total tokens).
    """
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


class LMFeatureExtractor:
    """
    Extracts perplexity and burstiness features using a causal language model (GPT-2).
    Uses sliding window for long texts to avoid truncation artifacts.
    """

    def __init__(
        self,
        model_name_or_path: str = AI_DETECTOR_LM_NAME,
        device: Optional[str] = None,
        max_length: int = 1024,
        stride: int = 512,
    ):
        self.model_name = model_name_or_path
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.max_length = max_length
        self.stride = stride

        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForCausalLM] = None

    def _load_model(self) -> None:
        """Lazy load tokenizer and model on first use."""
        if self._model is None or self._tokenizer is None:
            logger.info("Loading LM feature extractor model: %s on %s", self.model_name, self.device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()

    @property
    def tokenizer(self) -> AutoTokenizer:
        self._load_model()
        assert self._tokenizer is not None
        return self._tokenizer

    @property
    def model(self) -> AutoModelForCausalLM:
        self._load_model()
        assert self._model is not None
        return self._model

    def compute_perplexity(self, text: str) -> float:
        """
        Compute perplexity of text using sliding-window cross-entropy.
        Returns a positive float value; if text is empty, returns float('nan').
        """
        text = text.strip()
        if not text:
            return float("nan")

        encodings = self.tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.device)
        seq_len = input_ids.size(1)

        if seq_len < 2:
            return float("nan")

        nlls: List[torch.Tensor] = []
        prev_end_loc = 0

        with torch.no_grad():
            for begin_loc in range(0, seq_len, self.stride):
                end_loc = min(begin_loc + self.max_length, seq_len)
                trg_len = end_loc - prev_end_loc
                input_chunk = input_ids[:, begin_loc:end_loc]
                target_chunk = input_chunk.clone()
                target_chunk[:, :-trg_len] = -100

                outputs = self.model(input_chunk, labels=target_chunk)
                neg_log_likelihood = outputs.loss

                if not torch.isnan(neg_log_likelihood):
                    nlls.append(neg_log_likelihood * trg_len)

                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break

        if not nlls:
            return float("nan")

        total_nll = torch.stack(nlls).sum()
        ppl = torch.exp(total_nll / seq_len).item()
        return float(min(ppl, 1e5))  # clip extreme spikes

    def compute_sentence_perplexities(self, text: str) -> List[Dict[str, Any]]:
        """
        Compute individual perplexities for each sentence in the text.
        Returns a list of dicts with: sentence, perplexity, word_count.
        """
        sentences = split_into_sentences(text)
        results = []

        for sent in sentences:
            ppl = self.compute_perplexity(sent)
            word_count = len(re.findall(r"\b\w+\b", sent))
            results.append({
                "sentence": sent,
                "perplexity": float(ppl) if not math.isnan(ppl) else 0.0,
                "word_count": word_count,
            })

        return results

    def compute_burstiness(self, text: str) -> Dict[str, float]:
        """
        Burstiness measures the variation and irregularity in perplexity and sentence
        lengths across an essay. Human writing exhibits high variance in sentence
        structure and predictability, whereas AI text tends to be uniform.
        """
        sent_data = self.compute_sentence_perplexities(text)
        if not sent_data:
            return {
                "burstiness_ppl_var": 0.0,
                "burstiness_ppl_std": 0.0,
                "burstiness_cv": 0.0,
                "sentence_len_mean": 0.0,
                "sentence_len_std": 0.0,
                "sentence_count": 0.0,
            }

        ppls = [d["perplexity"] for d in sent_data if not math.isnan(d["perplexity"]) and d["perplexity"] > 0]
        lengths = [d["word_count"] for d in sent_data]

        if len(ppls) >= 2:
            ppl_var = float(np.var(ppls))
            ppl_std = float(np.std(ppls))
            ppl_mean = float(np.mean(ppls))
            ppl_cv = float(ppl_std / (ppl_mean + 1e-6))
        elif len(ppls) == 1:
            ppl_var = 0.0
            ppl_std = 0.0
            ppl_cv = 0.0
        else:
            ppl_var = 0.0
            ppl_std = 0.0
            ppl_cv = 0.0

        len_mean = float(np.mean(lengths)) if lengths else 0.0
        len_std = float(np.std(lengths)) if len(lengths) >= 2 else 0.0

        return {
            "burstiness_ppl_var": ppl_var,
            "burstiness_ppl_std": ppl_std,
            "burstiness_cv": ppl_cv,
            "sentence_len_mean": len_mean,
            "sentence_len_std": len_std,
            "sentence_count": float(len(sent_data)),
        }

    def extract_features(self, text: str) -> Dict[str, float]:
        """
        Extract complete dictionary of LM-based features for text classification.
        """
        ppl = self.compute_perplexity(text)
        if math.isnan(ppl) or ppl <= 0:
            ppl = 50.0  # reasonable fallback baseline
        log_ppl = float(np.log(max(ppl, 1.0)))

        burst = self.compute_burstiness(text)
        ttr = compute_ttr(text)

        # Token length and uppercase heuristics
        chars = len(text)
        words = len(re.findall(r"\b\w+\b", text))
        avg_word_len = chars / max(words, 1)

        features = {
            "perplexity": float(ppl),
            "log_perplexity": log_ppl,
            "burstiness_ppl_var": burst["burstiness_ppl_var"],
            "burstiness_ppl_std": burst["burstiness_ppl_std"],
            "burstiness_cv": burst["burstiness_cv"],
            "sentence_len_mean": burst["sentence_len_mean"],
            "sentence_len_std": burst["sentence_len_std"],
            "sentence_count": burst["sentence_count"],
            "ttr": float(ttr),
            "avg_word_len": float(avg_word_len),
        }
        return features

    def extract_feature_vector(self, text: str) -> np.ndarray:
        """
        Extract numerical feature vector ordered consistently for classifier training.
        """
        feat_dict = self.extract_features(text)
        keys = [
            "perplexity", "log_perplexity", "burstiness_ppl_var",
            "burstiness_ppl_std", "burstiness_cv", "sentence_len_mean",
            "sentence_len_std", "sentence_count", "ttr", "avg_word_len",
        ]
        return np.array([feat_dict[k] for k in keys], dtype=np.float32)
