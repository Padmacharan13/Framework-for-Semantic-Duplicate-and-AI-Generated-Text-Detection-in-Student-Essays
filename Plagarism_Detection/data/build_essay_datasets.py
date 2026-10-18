"""
build_essay_datasets.py
Creates essay-level datasets for both AI-detection and duplicate-detection tasks.

Datasets produced:
  (a) AI Detection: human essays vs AI-generated essays — uses the
      Hello-SimpleAI/HC3 dataset (Human ChatGPT Comparison Corpus) from
      Hugging Face, which contains human answers and ChatGPT answers to
      the same questions across multiple domains.
  (b) Duplicate/Paraphrase Detection: essay pairs with different attack types
      (verbatim copy, synonym swap, sentence reorder, unrelated) built from
      the human essays in the AI detection set.

Outputs (Parquet):
  data/essays/ai_detection_{split}.parquet   — columns: id, text, label (0=human, 1=ai), source_domain
  data/essays/duplicate_pairs_{split}.parquet — columns: id, text_a, text_b, label (0/1), attack_type

All random operations use RANDOM_SEED=42 for reproducibility.
Splits: train (70%), validation (15%), test (15%).

Dataset Licenses:
  - HC3: CC BY-SA 4.0 (https://huggingface.co/datasets/Hello-SimpleAI/HC3)

Run:
    python -m data.build_essay_datasets
    # or
    python data/build_essay_datasets.py
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

# Add project root to path so we can import config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from semantic_integrity.config import RANDOM_SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "data" / "essays"

# ── Reproducibility ──
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Minimum text length to keep (characters)
MIN_TEXT_LEN = 120


# ====================================================================
# Part (a): AI Detection Dataset from HC3
# ====================================================================

def build_ai_detection_dataset() -> pd.DataFrame:
    """
    Load HC3 (Human ChatGPT Comparison Corpus) and create a binary
    classification dataset: label 0 = human, label 1 = AI-generated.
    """
    logger.info("Loading HC3 dataset from Hugging Face Hub...")

    records = []
    # HC3 has multiple configs: finance, medicine, open_qa, wiki_csai, etc.
    configs = ["finance", "medicine", "open_qa"]
    base_url = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main"

    for config_name in configs:
        try:
            ds = load_dataset(
                "json",
                data_files=f"{base_url}/{config_name}.jsonl",
                split="train",
            )
            logger.info("  Loaded HC3 config '%s': %d rows", config_name, len(ds))

            for row in ds:
                question = str(row.get("question", ""))

                # Human answers
                human_answers = row.get("human_answers", [])
                for ans in human_answers:
                    text = str(ans).strip()
                    if len(text) >= MIN_TEXT_LEN:
                        records.append({
                            "text": text,
                            "label": 0,  # human
                            "source_domain": config_name,
                            "question": question[:200],
                        })

                # ChatGPT answers
                chatgpt_answers = row.get("chatgpt_answers", [])
                for ans in chatgpt_answers:
                    text = str(ans).strip()
                    if len(text) >= MIN_TEXT_LEN:
                        records.append({
                            "text": text,
                            "label": 1,  # AI-generated
                            "source_domain": config_name,
                            "question": question[:200],
                        })

        except Exception as e:
            logger.warning("Failed to load HC3 config '%s': %s", config_name, e)

    if not records:
        logger.error("No records loaded from HC3. Check network connection.")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Deduplicate by text hash
    df["text_hash"] = df["text"].apply(lambda t: hashlib.md5(t.encode()).hexdigest())
    df = df.drop_duplicates(subset=["text_hash"]).drop(columns=["text_hash"])

    # Balance classes by downsampling the larger class
    df_human = df[df["label"] == 0]
    df_ai = df[df["label"] == 1]
    min_count = min(len(df_human), len(df_ai))
    df = pd.concat([
        df_human.sample(n=min_count, random_state=RANDOM_SEED),
        df_ai.sample(n=min_count, random_state=RANDOM_SEED),
    ], ignore_index=True)

    # Add IDs
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    df["id"] = [f"ai_det_{i:05d}" for i in range(len(df))]

    logger.info("AI Detection dataset: %d samples (human=%d, ai=%d)",
                len(df), (df["label"] == 0).sum(), (df["label"] == 1).sum())
    return df


# ====================================================================
# Part (b): Duplicate/Paraphrase Pair Dataset
# ====================================================================

def _synonym_swap(text: str) -> str:
    """Simple word-level synonym swap using a small lookup table."""
    synonyms = {
        "important": "crucial", "crucial": "vital", "significant": "notable",
        "increase": "rise", "decrease": "decline", "method": "approach",
        "approach": "technique", "study": "research", "research": "investigation",
        "show": "demonstrate", "demonstrate": "illustrate", "large": "substantial",
        "small": "minor", "help": "assist", "problem": "issue",
        "issue": "challenge", "result": "outcome", "provide": "offer",
        "however": "nevertheless", "therefore": "consequently",
        "also": "additionally", "because": "since", "although": "though",
        "many": "numerous", "few": "several", "very": "extremely",
        "good": "excellent", "bad": "poor", "new": "novel",
        "use": "utilize", "make": "create", "find": "discover",
        "think": "believe", "want": "desire", "need": "require",
    }
    words = text.split()
    swapped = []
    for w in words:
        lower = w.lower().strip(".,!?;:")
        if lower in synonyms and random.random() < 0.3:
            replacement = synonyms[lower]
            # Preserve original casing of first letter
            if w[0].isupper():
                replacement = replacement.capitalize()
            swapped.append(replacement)
        else:
            swapped.append(w)
    return " ".join(swapped)


def _sentence_reorder(text: str) -> str:
    """Shuffle sentence order (keeps first and last sentence in place for coherence)."""
    # Split on sentence-ending punctuation followed by space
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= 2:
        return text
    middle = sentences[1:-1]
    random.shuffle(middle)
    return " ".join([sentences[0]] + middle + [sentences[-1]])


def build_duplicate_pairs_dataset(human_essays: pd.DataFrame) -> pd.DataFrame:
    """
    Create essay pairs with different attack types from human essays.

    Attack types:
      - verbatim: identical copy (label=1)
      - synonym_swap: word-level synonym replacement (label=1)
      - sentence_reorder: shuffle sentence order (label=1)
      - unrelated: pair with a random different essay (label=0)
    """
    logger.info("Building duplicate/paraphrase pair dataset...")

    texts = human_essays["text"].tolist()
    if len(texts) < 20:
        logger.warning("Too few human essays (%d) for pair generation", len(texts))
        return pd.DataFrame()

    records = []
    pair_id = 0

    for i, original in enumerate(texts):
        if len(original.strip()) < MIN_TEXT_LEN:
            continue

        # 1. Verbatim copy
        records.append({
            "text_a": original,
            "text_b": original,
            "label": 1,
            "attack_type": "verbatim",
        })
        pair_id += 1

        # 2. Synonym swap
        swapped = _synonym_swap(original)
        records.append({
            "text_a": original,
            "text_b": swapped,
            "label": 1,
            "attack_type": "synonym_swap",
        })
        pair_id += 1

        # 3. Sentence reorder
        reordered = _sentence_reorder(original)
        records.append({
            "text_a": original,
            "text_b": reordered,
            "label": 1,
            "attack_type": "sentence_reorder",
        })
        pair_id += 1

        # 4. Unrelated pairs (3 unrelated pairs to balance the 3 positive attack pairs)
        candidates = [j for j in range(len(texts)) if j != i]
        if len(candidates) >= 3:
            neg_samples = random.sample(candidates, 3)
            for j in neg_samples:
                records.append({
                    "text_a": original,
                    "text_b": texts[j],
                    "label": 0,
                    "attack_type": "unrelated",
                })
                pair_id += 1
        elif candidates:
            j = random.choice(candidates)
            records.append({
                "text_a": original,
                "text_b": texts[j],
                "label": 0,
                "attack_type": "unrelated",
            })
            pair_id += 1

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    df["id"] = [f"dup_pair_{i:05d}" for i in range(len(df))]

    logger.info("Duplicate pairs dataset: %d pairs", len(df))
    logger.info("  Attack type distribution:\n%s", df["attack_type"].value_counts().to_string())
    return df


# ====================================================================
# Splitting & Saving
# ====================================================================

def split_and_save(df: pd.DataFrame, name: str) -> None:
    """Split into train/val/test (70/15/15) and save as Parquet."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n = len(df)
    indices = np.random.permutation(n)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    splits = {
        "train": indices[:n_train],
        "validation": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }

    for split_name, split_indices in splits.items():
        split_df = df.iloc[split_indices].copy()
        split_df["split"] = split_name
        out_path = OUTPUT_DIR / f"{name}_{split_name}.parquet"
        split_df.to_parquet(out_path, index=False)
        logger.info("  Saved %s -> %s (%d rows)", split_name, out_path.name, len(split_df))


def main() -> None:
    logger.info("=" * 60)
    logger.info("Building Essay-Level Datasets")
    logger.info("=" * 60)

    # (a) AI Detection dataset
    logger.info("\n--- Part (a): AI Detection Dataset ---")
    ai_df = build_ai_detection_dataset()
    if len(ai_df) > 0:
        split_and_save(ai_df, "ai_detection")
    else:
        logger.error("AI detection dataset is empty; skipping save.")

    # (b) Duplicate pairs dataset (uses human essays from part a)
    logger.info("\n--- Part (b): Duplicate Pairs Dataset ---")
    if len(ai_df) > 0:
        human_essays = ai_df[ai_df["label"] == 0].copy()
        dup_df = build_duplicate_pairs_dataset(human_essays)
        if len(dup_df) > 0:
            split_and_save(dup_df, "duplicate_pairs")
        else:
            logger.error("Duplicate pairs dataset is empty; skipping save.")
    else:
        logger.error("No human essays available for duplicate pair generation.")

    logger.info("\n=== Dataset build complete ===")
    logger.info("Output directory: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
