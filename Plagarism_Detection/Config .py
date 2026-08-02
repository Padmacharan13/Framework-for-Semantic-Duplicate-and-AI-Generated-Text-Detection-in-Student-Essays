"""
config.py
Central place for paths and constants.

MSRP and QQP are now pulled directly from the Hugging Face Hub via the
`datasets` library - no manual downloads or local raw files needed.
Only your own domain-specific paraphrase mini-set still lives on disk
(since it's hand-written by your group, not a public dataset).

    data/
      processed/                 <- Week 1 scripts write cleaned output here
      domain/
        domain_paraphrase_seed.csv   <- template this project generates for you
"""

from pathlib import Path

# ---- Root paths -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DOMAIN_DATA_DIR = DATA_DIR / "domain"

for _d in (PROCESSED_DATA_DIR, DOMAIN_DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- MSRP (Hugging Face: GLUE benchmark, "mrpc" task) ---------------------
# https://huggingface.co/datasets/nyu-mll/glue  (config: "mrpc")
# Columns: sentence1, sentence2, label (1 = paraphrase, 0 = not), idx
HF_MSRP_DATASET = "glue"
HF_MSRP_CONFIG = "mrpc"
HF_MSRP_SPLITS = ["train", "validation", "test"]  # mrpc test split DOES have labels

# ---- QQP (Hugging Face: GLUE benchmark, "qqp" task) ------------------------
# https://huggingface.co/datasets/nyu-mll/glue  (config: "qqp")
# Columns: question1, question2, label (1 = duplicate, 0 = not), idx
# NOTE: GLUE's qqp "test" split has no labels (label = -1) since it's used
# for the official leaderboard - only train/validation are labeled.
HF_QQP_DATASET = "glue"
HF_QQP_CONFIG = "qqp"
HF_QQP_SPLITS = ["train", "validation"]

# ---- Domain (your own academic-paraphrase) dataset -----------------------
DOMAIN_SEED_FILE = DOMAIN_DATA_DIR / "domain_paraphrase_seed.csv"
DOMAIN_LABELED_FILE = DOMAIN_DATA_DIR / "domain_paraphrase_labeled.csv"

# ---- Output of Week 1 preprocessing --------------------------------------
PROCESSED_MSRP_FILE = PROCESSED_DATA_DIR / "msrp_processed.parquet"
PROCESSED_QQP_FILE = PROCESSED_DATA_DIR / "qqp_processed.parquet"
PROCESSED_DOMAIN_FILE = PROCESSED_DATA_DIR / "domain_processed.parquet"

# ---- spaCy model ----------------------------------------------------------
SPACY_MODEL = "en_core_web_sm"

# ---- Paraphrase intensity tiers (used from Week 1 onward for labeling) ---
PARAPHRASE_TIERS = ["verbatim", "light_paraphrase", "heavy_paraphrase", "structural_rewrite"]

# ---- Misc -------------------------------------------------------------
RANDOM_SEED = 42
QQP_SAMPLE_SIZE = 50_000  # QQP has 400k+ pairs; subsample for speed during dev