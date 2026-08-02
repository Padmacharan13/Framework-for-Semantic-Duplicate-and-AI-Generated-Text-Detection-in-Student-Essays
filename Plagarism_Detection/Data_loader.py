"""
data_loader.py
Loads MSRP and QQP DIRECTLY from the Hugging Face Hub (via `datasets`) into
a standardized schema - no manual downloads, no local raw files needed.

Standardized schema for both:

    id | text_a | text_b | label | source

label: 1 = paraphrase/duplicate, 0 = not
source: "msrp" or "qqp"

First call will download and cache the dataset under ~/.cache/huggingface
(only happens once - subsequent runs load from local cache, no network).
"""

from __future__ import annotations

import logging

import pandas as pd
from datasets import concatenate_datasets, load_dataset

from config import (
    HF_MSRP_CONFIG,
    HF_MSRP_DATASET,
    HF_MSRP_SPLITS,
    HF_QQP_CONFIG,
    HF_QQP_DATASET,
    HF_QQP_SPLITS,
    QQP_SAMPLE_SIZE,
    RANDOM_SEED,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MSRP  (Hugging Face: glue/mrpc)
# ---------------------------------------------------------------------------

def load_msrp() -> pd.DataFrame:
    """
    Pulls GLUE MRPC (the Hugging Face hosted version of MSRP) directly from
    the Hub and concatenates train+validation+test into one DataFrame.
    """
    logger.info("Loading MSRP (glue/mrpc) from Hugging Face Hub...")
    splits = []
    for split in HF_MSRP_SPLITS:
        ds = load_dataset(HF_MSRP_DATASET, HF_MSRP_CONFIG, split=split)
        splits.append(ds)
    full = concatenate_datasets(splits)

    df = full.to_pandas()
    standardized = pd.DataFrame({
        "id": df.index.astype(str).map(lambda i: f"msrp_{i}"),
        "text_a": df["sentence1"].astype(str),
        "text_b": df["sentence2"].astype(str),
        "label": df["label"].astype(int),
        "source": "msrp",
    })
    standardized = standardized.dropna(subset=["text_a", "text_b"]).reset_index(drop=True)
    logger.info("Loaded MSRP: %d pairs", len(standardized))
    return standardized


# ---------------------------------------------------------------------------
# QQP  (Hugging Face: glue/qqp)
# ---------------------------------------------------------------------------

def load_qqp(sample_size: int | None = QQP_SAMPLE_SIZE) -> pd.DataFrame:
    """
    Pulls GLUE QQP directly from the Hub. Only train+validation are used
    since GLUE's official qqp "test" split ships without labels (label=-1,
    reserved for the leaderboard). Subsamples for dev speed since QQP has
    400k+ rows in the train split alone.
    """
    logger.info("Loading QQP (glue/qqp) from Hugging Face Hub...")
    splits = []
    for split in HF_QQP_SPLITS:
        ds = load_dataset(HF_QQP_DATASET, HF_QQP_CONFIG, split=split)
        splits.append(ds)
    full = concatenate_datasets(splits)

    df = full.to_pandas()
    df = df[df["label"].isin([0, 1])]  # safety: drop any unlabeled rows
    df = df.dropna(subset=["question1", "question2", "label"])

    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=RANDOM_SEED).reset_index(drop=True)

    standardized = pd.DataFrame({
        "id": df.index.astype(str).map(lambda i: f"qqp_{i}"),
        "text_a": df["question1"].astype(str),
        "text_b": df["question2"].astype(str),
        "label": df["label"].astype(int),
        "source": "qqp",
    })
    logger.info("Loaded QQP: %d pairs (sampled to %s)", len(standardized), sample_size)
    return standardized


# ---------------------------------------------------------------------------
# Combined loader
# ---------------------------------------------------------------------------

def load_all_benchmark_data() -> pd.DataFrame:
    """Loads and concatenates MSRP + QQP into one standardized DataFrame."""
    dfs = [load_msrp(), load_qqp()]
    combined = pd.concat(dfs, ignore_index=True)
    logger.info("Combined benchmark dataset: %d total pairs", len(combined))
    return combined


if __name__ == "__main__":
    df = load_all_benchmark_data()
    print(df["source"].value_counts())
    print(df.head())