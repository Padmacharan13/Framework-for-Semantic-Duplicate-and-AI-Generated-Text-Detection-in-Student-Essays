"""
main_pipeline.py
Runs preprocessing on MSRP + QQP and saves the results to parquet.
This is the Week 1 half of the checkpoint - run this BEFORE evaluate.py
if you want the saved parquet files available for later inspection
(evaluate.py itself re-loads data fresh from Hugging Face and doesn't
require this step, but saving processed output is good practice /
useful evidence to show staff that the pipeline is end-to-end).

Run:
    python main_pipeline.py
"""

from __future__ import annotations

import logging

import pandas as pd

from config import PROCESSED_MSRP_FILE, PROCESSED_QQP_FILE
from data_loader import load_msrp, load_qqp
from preprocessing import TextPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def preprocess_pair_dataframe(df: pd.DataFrame, preprocessor: TextPreprocessor) -> pd.DataFrame:
    processed_a = preprocessor.process_column(df["text_a"].tolist())
    processed_b = preprocessor.process_column(df["text_b"].tolist())

    df = df.copy()
    df["text_a_clean"] = [p.cleaned for p in processed_a]
    df["text_a_sentences"] = [p.sentences for p in processed_a]
    df["text_b_clean"] = [p.cleaned for p in processed_b]
    df["text_b_sentences"] = [p.sentences for p in processed_b]
    return df


def run() -> None:
    preprocessor = TextPreprocessor()

    msrp_df = load_msrp()
    msrp_processed = preprocess_pair_dataframe(msrp_df, preprocessor)
    msrp_processed.to_parquet(PROCESSED_MSRP_FILE, index=False)
    logger.info("Saved processed MSRP -> %s (%d rows)", PROCESSED_MSRP_FILE, len(msrp_processed))

    qqp_df = load_qqp()
    qqp_processed = preprocess_pair_dataframe(qqp_df, preprocessor)
    qqp_processed.to_parquet(PROCESSED_QQP_FILE, index=False)
    logger.info("Saved processed QQP -> %s (%d rows)", PROCESSED_QQP_FILE, len(qqp_processed))

    logger.info("Preprocessing pipeline complete.")


if __name__ == "__main__":
    run()