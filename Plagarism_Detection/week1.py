"""
main_week1.py
Runs the full Week 1 pipeline end-to-end:

  1. Load MSRP + QQP (standardized schema)
  2. Load your domain paraphrase mini-dataset (if already built/validated)
  3. Batch-preprocess every text column (clean, segment, lemmatize)
  4. Save everything to data/processed/*.parquet for Week 2 to consume

Run:
    python main_week1.py
"""

from __future__ import annotations

import logging

import pandas as pd

from config import (
    DOMAIN_LABELED_FILE,
    PROCESSED_DOMAIN_FILE,
    PROCESSED_MSRP_FILE,
    PROCESSED_QQP_FILE,
)
from data_loader import load_msrp, load_qqp
from preprocessing import TextPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def preprocess_pair_dataframe(df: pd.DataFrame, preprocessor: TextPreprocessor) -> pd.DataFrame:
    """
    Given a DataFrame with text_a/text_b columns, adds cleaned/sentence/lemma
    columns for both sides using ONE batched pass per column (not per row).
    """
    processed_a = preprocessor.process_column(df["text_a"].tolist())
    processed_b = preprocessor.process_column(df["text_b"].tolist())

    df = df.copy()
    df["text_a_clean"] = [p.cleaned for p in processed_a]
    df["text_a_sentences"] = [p.sentences for p in processed_a]
    df["text_a_lemmas"] = [p.lemmatized_sentences for p in processed_a]

    df["text_b_clean"] = [p.cleaned for p in processed_b]
    df["text_b_sentences"] = [p.sentences for p in processed_b]
    df["text_b_lemmas"] = [p.lemmatized_sentences for p in processed_b]

    return df


def run() -> None:
    preprocessor = TextPreprocessor()  # load spaCy once, reuse for everything below

    # ---- MSRP ----
    try:
        msrp_df = load_msrp()
        msrp_processed = preprocess_pair_dataframe(msrp_df, preprocessor)
        msrp_processed.to_parquet(PROCESSED_MSRP_FILE, index=False)
        logger.info("Saved processed MSRP -> %s (%d rows)", PROCESSED_MSRP_FILE, len(msrp_processed))
    except Exception as e:
        logger.warning("Skipping MSRP (Hugging Face load failed): %s", e)

    # ---- QQP ----
    try:
        qqp_df = load_qqp()
        qqp_processed = preprocess_pair_dataframe(qqp_df, preprocessor)
        qqp_processed.to_parquet(PROCESSED_QQP_FILE, index=False)
        logger.info("Saved processed QQP -> %s (%d rows)", PROCESSED_QQP_FILE, len(qqp_processed))
    except Exception as e:
        logger.warning("Skipping QQP (Hugging Face load failed): %s", e)

    # ---- Domain mini-dataset (only if you've already filled + validated it) ----
    if DOMAIN_LABELED_FILE.exists():
        domain_df = pd.read_csv(DOMAIN_LABELED_FILE)
        domain_df = domain_df.rename(columns={"source_text": "text_a", "rewrite_text": "text_b"})
        domain_df["label"] = (domain_df["target_tier"] != "not_paraphrase").astype(int)
        domain_df["source"] = "domain"
        domain_processed = preprocess_pair_dataframe(domain_df, preprocessor)
        domain_processed.to_parquet(PROCESSED_DOMAIN_FILE, index=False)
        logger.info(
            "Saved processed domain dataset -> %s (%d rows)",
            PROCESSED_DOMAIN_FILE, len(domain_processed),
        )
    else:
        logger.info(
            "Domain dataset not found at %s yet - run build_domain_dataset.py "
            "(generate, then validate) before this step.",
            DOMAIN_LABELED_FILE,
        )

    logger.info("Week 1 pipeline complete.")


if __name__ == "__main__":
    run()