"""
data_loader.py
Loads MSRP and QQP directly from the Hugging Face Hub into a standardized
schema:  id | text_a | text_b | label | source
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


def load_msrp() -> pd.DataFrame:
    logger.info("Loading MSRP (nyu-mll/glue, mrpc) from Hugging Face Hub...")
    splits = [load_dataset(HF_MSRP_DATASET, HF_MSRP_CONFIG, split=s) for s in HF_MSRP_SPLITS]
    df = concatenate_datasets(splits).to_pandas()
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


def load_qqp(sample_size: int | None = QQP_SAMPLE_SIZE) -> pd.DataFrame:
    logger.info("Loading QQP (nyu-mll/glue, qqp) from Hugging Face Hub...")
    splits = [load_dataset(HF_QQP_DATASET, HF_QQP_CONFIG, split=s) for s in HF_QQP_SPLITS]
    df = concatenate_datasets(splits).to_pandas()
    df = df[df["label"].isin([0, 1])]
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


def load_all_benchmark_data() -> pd.DataFrame:
    combined = pd.concat([load_msrp(), load_qqp()], ignore_index=True)
    logger.info("Combined benchmark dataset: %d total pairs", len(combined))
    return combined


if __name__ == "__main__":
    df = load_all_benchmark_data()
    print(df["source"].value_counts())
    print(df.head())