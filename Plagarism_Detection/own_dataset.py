"""
build_domain_dataset.py
Generates a fill-in-the-blank CSV template for the domain-specific
academic-paraphrase mini-dataset described in the Week 1 plan, and
validates it once your group has filled it in.

Workflow:
  1. Run `python build_domain_dataset.py generate` once -> writes
     data/domain/domain_paraphrase_seed.csv with source paragraphs and
     empty columns for each group member to fill in.
  2. Each group member fills in their assigned "rewrite" columns by hand
     (this is manual writing, not something to automate - the point is
     that a human produced the paraphrase, like a student would).
  3. Run `python build_domain_dataset.py validate` -> checks the filled
     file for missing cells, duplicate rewrites, and prints row counts
     per paraphrase tier so you know your label balance before Week 3.
"""

from __future__ import annotations

import sys

import pandas as pd

from config import DOMAIN_SEED_FILE, DOMAIN_LABELED_FILE, PARAPHRASE_TIERS

# Replace these with real paragraphs from your own writing or public-domain
# text (do NOT use copyrighted essays verbatim as your "source" paragraphs -
# write your own short source paragraphs, 3-5 sentences each).
SAMPLE_SOURCE_PARAGRAPHS = [
    "The rapid growth of digital assignment submissions has made manual "
    "plagiarism checking impractical for most instructors. Automated tools "
    "are now essential for maintaining academic integrity at scale.",
    "Renewable energy sources such as solar and wind power are becoming "
    "more cost effective every year. As battery storage improves, these "
    "sources are expected to replace a large share of fossil fuel generation.",
    "Effective time management involves prioritizing tasks based on their "
    "urgency and importance. Students who plan their week in advance tend "
    "to experience less stress before deadlines.",
]


def generate_template(n_copies_per_source: int = 1) -> None:
    """Writes an empty-to-fill CSV: one row per (source paragraph, tier)."""
    rows = []
    for src_id, source_text in enumerate(SAMPLE_SOURCE_PARAGRAPHS):
        for tier in PARAPHRASE_TIERS[1:]:  # skip "verbatim" - that's just the source itself
            rows.append({
                "source_id": src_id,
                "source_text": source_text,
                "target_tier": tier,
                "rewrite_text": "",       # <-- group member fills this in by hand
                "written_by": "",          # <-- group member's name, for accountability
            })
        # also add an explicit verbatim row (exact copy) as the trivial positive case
        rows.append({
            "source_id": src_id,
            "source_text": source_text,
            "target_tier": "verbatim",
            "rewrite_text": source_text,
            "written_by": "system_generated",
        })

    df = pd.DataFrame(rows)
    df.to_csv(DOMAIN_SEED_FILE, index=False)
    print(f"Template written to {DOMAIN_SEED_FILE}")
    print(f"Rows to fill in by hand: {(df['rewrite_text'] == '').sum()}")
    print("Open this CSV, have each group member write their assigned rewrites, "
          "save it as the same filename, then run: python build_domain_dataset.py validate")


def validate_filled_dataset() -> None:
    """Checks the hand-filled CSV for completeness before it's used in Week 3."""
    if not DOMAIN_SEED_FILE.exists():
        print(f"No file found at {DOMAIN_SEED_FILE}. Run 'generate' first.")
        sys.exit(1)

    df = pd.read_csv(DOMAIN_SEED_FILE)

    empty_mask = df["rewrite_text"].isna() | (df["rewrite_text"].astype(str).str.strip() == "")
    n_empty = empty_mask.sum()
    if n_empty > 0:
        print(f"WARNING: {n_empty} rows still have an empty rewrite_text. "
              f"Rows: {df[empty_mask].index.tolist()}")

    # flag rewrites that are suspiciously identical to the source (likely someone
    # just copy-pasted instead of paraphrasing)
    same_as_source = (
        df["rewrite_text"].astype(str).str.strip().str.lower()
        == df["source_text"].astype(str).str.strip().str.lower()
    ) & (df["target_tier"] != "verbatim")
    if same_as_source.any():
        print(f"WARNING: {same_as_source.sum()} non-verbatim rows are identical "
              f"to the source text - these need to actually be rewritten. "
              f"Rows: {df[same_as_source].index.tolist()}")

    print("\nRow counts per tier:")
    print(df["target_tier"].value_counts())

    if n_empty == 0 and not same_as_source.any():
        df.to_csv(DOMAIN_LABELED_FILE, index=False)
        print(f"\nAll rows valid. Final labeled dataset saved to {DOMAIN_LABELED_FILE}")
    else:
        print("\nFix the warnings above, then re-run validate.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "generate"
    if mode == "generate":
        generate_template()
    elif mode == "validate":
        validate_filled_dataset()
    else:
        print("Usage: python build_domain_dataset.py [generate|validate]")