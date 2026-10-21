"""
data_loader.py
Loads MSRP and QQP directly from the Hugging Face Hub into a standardized
schema:  id | text_a | text_b | label | source
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from .config import (
    HF_MSRP_CONFIG,
    HF_MSRP_DATASET,
    HF_MSRP_SPLITS,
    HF_QQP_CONFIG,
    HF_QQP_DATASET,
    HF_QQP_SPLITS,
    MSRP_PROCESSED_PATH,
    QQP_PROCESSED_PATH,
    QQP_SAMPLE_SIZE,
    RANDOM_SEED,
    PROCESSED_DATA_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SPLIT_IDS_PATH = PROCESSED_DATA_DIR / "split_ids.json"


def load_msrp(force_reload: bool = False) -> pd.DataFrame:
    """
    Load MSRP (MRPC) from Hugging Face Hub using the official GLUE MRPC labeled splits:
      - train: 3,668 rows
      - validation: 408 rows
      - test: 1,725 rows
    All official splits have genuine binary labels (0 or 1).
    """
    logger.info("Loading MSRP (nyu-mll/glue, mrpc) from Hugging Face Hub...")
    dfs = []
    for s in HF_MSRP_SPLITS:
        ds_df = load_dataset(HF_MSRP_DATASET, HF_MSRP_CONFIG, split=s).to_pandas()
        ds_df["split"] = s
        dfs.append(ds_df)
    df = pd.concat(dfs, ignore_index=True)

    # Filter out any non-binary labels if present
    df = df[df["label"].isin([0, 1])]
    df = df.dropna(subset=["sentence1", "sentence2"]).reset_index(drop=True)

    standardized = pd.DataFrame({
        "id": [f"msrp_{row.split}_{idx}" for idx, row in df.iterrows()],
        "split": df["split"].astype(str),
        "text1": df["sentence1"].astype(str),
        "text2": df["sentence2"].astype(str),
        "text_a": df["sentence1"].astype(str),
        "text_b": df["sentence2"].astype(str),
        "label": df["label"].astype(int),
        "source": "msrp",
    })
    standardized = standardized.dropna(subset=["text1", "text2"]).reset_index(drop=True)
    logger.info("Loaded MSRP: %d pairs (train=%d, val=%d, test=%d)",
                len(standardized),
                (standardized["split"] == "train").sum(),
                (standardized["split"] == "validation").sum(),
                (standardized["split"] == "test").sum())
    return standardized


def load_qqp(sample_size: int | None = QQP_SAMPLE_SIZE) -> pd.DataFrame:
    """
    Load QQP from Hugging Face Hub using labeled train and validation splits.
    Carve a fixed held-out test split using RANDOM_SEED to prevent data leakage:
      - 80% train
      - 10% validation
      - 10% test
    """
    logger.info("Loading QQP (nyu-mll/glue, qqp) from Hugging Face Hub...")
    dfs = []
    # Only load labeled splits (GLUE test split has dummy -1 labels)
    for s in ["train", "validation"]:
        ds_df = load_dataset(HF_QQP_DATASET, HF_QQP_CONFIG, split=s).to_pandas()
        dfs.append(ds_df)
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["label"].isin([0, 1])]
    df = df.dropna(subset=["question1", "question2", "label"]).reset_index(drop=True)

    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=RANDOM_SEED).reset_index(drop=True)
    else:
        df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    n = len(df)
    n_val = max(1, int(n * 0.10))
    n_test = max(1, int(n * 0.10))
    n_train = n - n_val - n_test

    splits = ["train"] * n_train + ["validation"] * n_val + ["test"] * n_test
    df["split"] = splits

    standardized = pd.DataFrame({
        "id": [f"qqp_{idx}" for idx in range(len(df))],
        "split": df["split"].astype(str),
        "text1": df["question1"].astype(str),
        "text2": df["question2"].astype(str),
        "text_a": df["question1"].astype(str),
        "text_b": df["question2"].astype(str),
        "label": df["label"].astype(int),
        "source": "qqp",
    })
    logger.info("Loaded QQP: %d pairs (train=%d, val=%d, test=%d)",
                len(standardized),
                (standardized["split"] == "train").sum(),
                (standardized["split"] == "validation").sum(),
                (standardized["split"] == "test").sum())
    return standardized


def save_split_ids(msrp_df: pd.DataFrame, qqp_df: pd.DataFrame) -> None:
    """Save explicit split IDs to disk to ensure identical partition usage across scripts."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    split_info = {
        "msrp": {
            "train": msrp_df[msrp_df["split"] == "train"]["id"].tolist(),
            "validation": msrp_df[msrp_df["split"] == "validation"]["id"].tolist(),
            "test": msrp_df[msrp_df["split"] == "test"]["id"].tolist(),
        },
        "qqp": {
            "train": qqp_df[qqp_df["split"] == "train"]["id"].tolist(),
            "validation": qqp_df[qqp_df["split"] == "validation"]["id"].tolist(),
            "test": qqp_df[qqp_df["split"] == "test"]["id"].tolist(),
        }
    }
    with open(SPLIT_IDS_PATH, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
    logger.info("Saved split IDs to %s", SPLIT_IDS_PATH)


def load_all_benchmark_data() -> pd.DataFrame:
    msrp = load_msrp()
    qqp = load_qqp()
    save_split_ids(msrp, qqp)
    combined = pd.concat([msrp, qqp], ignore_index=True)
    logger.info("Combined benchmark dataset: %d total pairs", len(combined))
    return combined


if __name__ == "__main__":
    df = load_all_benchmark_data()
    print(df["source"].value_counts())
    print(df.groupby(["source", "split"]).size())