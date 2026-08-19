"""
config.py
Central place for paths and constants. Flat project layout — this file
lives directly in the project root alongside every other .py file, so
PROJECT_ROOT is just this file's own folder (ONE .parent, not two —
that mismatch is what put data/ in the wrong place last time).
"""

from pathlib import Path

# ---- Root paths -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DOMAIN_DATA_DIR = DATA_DIR / "domain"
RESULTS_DIR = PROJECT_ROOT / "results"

for _d in (PROCESSED_DATA_DIR, DOMAIN_DATA_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- MSRP (Hugging Face: GLUE benchmark, "mrpc" task) ---------------------
# https://huggingface.co/datasets/nyu-mll/glue  (config: "mrpc")
# Columns: sentence1, sentence2, label (1 = paraphrase, 0 = not), idx
HF_MSRP_DATASET = "nyu-mll/glue"
HF_MSRP_CONFIG = "mrpc"
HF_MSRP_SPLITS = ["train", "validation", "test"]

# ---- QQP (Hugging Face: GLUE benchmark, "qqp" task) ------------------------
HF_QQP_DATASET = "nyu-mll/glue"
HF_QQP_CONFIG = "qqp"
HF_QQP_SPLITS = ["train", "validation"]

# ---- Domain (your own academic-paraphrase) dataset -----------------------
DOMAIN_SEED_FILE = DOMAIN_DATA_DIR / "domain_paraphrase_seed.csv"
DOMAIN_LABELED_FILE = DOMAIN_DATA_DIR / "domain_paraphrase_labeled.csv"

# ---- Output of preprocessing --------------------------------------------
PROCESSED_MSRP_FILE = PROCESSED_DATA_DIR / "msrp_processed.parquet"
PROCESSED_QQP_FILE = PROCESSED_DATA_DIR / "qqp_processed.parquet"
PROCESSED_DOMAIN_FILE = PROCESSED_DATA_DIR / "domain_processed.parquet"

# ---- spaCy model ----------------------------------------------------------
SPACY_MODEL = "en_core_web_sm"

# ---- Embedding models (SBERT & Word2Vec) -----------------------------------
SBERT_MODEL = "all-mpnet-base-v2"   # State-of-the-art 110M parameter SBERT model for top semantic quality
SBERT_BASELINE_MODEL = "all-MiniLM-L6-v2"
WORD2VEC_MODEL_FILE = DATA_DIR / "word2vec_paraphrase.model"
WORD2VEC_VECTOR_SIZE = 100

# ---- Similarity thresholds / tiers ----------------------------------------
SIMILARITY_THRESHOLD = 0.80           # duplicate/flagged cutoff on cosine similarity
PARAPHRASE_TIERS = ["verbatim", "light_paraphrase", "heavy_paraphrase", "structural_rewrite"]

# ---- Results outputs --------------------------------------------------
EVAL_RESULTS_CSV = RESULTS_DIR / "evaluation_results.csv"
EVAL_PLOT_FILE = RESULTS_DIR / "model_comparison_f1.png"
DEMO_REPORT_HTML = RESULTS_DIR / "demo_report.html"

# ---- Web Application Server -------------------------------------------
WEB_HOST = "127.0.0.1"
WEB_PORT = 5000

# ---- Misc -------------------------------------------------------------
RANDOM_SEED = 42
QQP_SAMPLE_SIZE = 20_000  # subsampled for speed; raise later for final numbers