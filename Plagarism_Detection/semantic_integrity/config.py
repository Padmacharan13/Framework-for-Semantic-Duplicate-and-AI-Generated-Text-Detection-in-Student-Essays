import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directory Paths
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Directories are created lazily by entry-point scripts, not at import time.
# To create them explicitly, call ensure_directories() below.
def ensure_directories():
    for directory in [DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

# Processed Files
MSRP_PROCESSED_PATH = PROCESSED_DATA_DIR / "msrp_processed.parquet"
QQP_PROCESSED_PATH = PROCESSED_DATA_DIR / "qqp_processed.parquet"

# Model Paths
WORD2VEC_MODEL_PATH = MODELS_DIR / "word2vec.model"
SBERT_FINETUNED_PATH = MODELS_DIR / "sbert_finetuned"
FAISS_INDEX_PATH = MODELS_DIR / "faiss_index.bin"
FAISS_METADATA_PATH = MODELS_DIR / "faiss_metadata.json"

# Model Settings
SBERT_BASE_MODEL = "all-MiniLM-L6-v2"
# Prefer the fine‑tuned SBERT model if it has been saved under the models directory.
# The fine‑tuned model is stored as a directory; if it exists we load it directly.

if Path(SBERT_FINETUNED_PATH).exists():
    SBERT_MODEL = str(SBERT_FINETUNED_PATH)
else:
    SBERT_MODEL = SBERT_BASE_MODEL
CROSS_ENCODER_MODEL = "cross-encoder/quora-distilroberta-base"
SPACY_MODEL = "en_core_web_sm"
W2V_VECTOR_SIZE = 300
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
W2V_EPOCHS = 10

# Vector DB & Re-ranking Settings
FAISS_TOP_K = 10
BORDERLINE_LOWER_THRESHOLD = 0.35
BORDERLINE_UPPER_THRESHOLD = 0.85

# Fine-Tuning Hyperparameters
TRAIN_BATCH_SIZE = 32
TRAIN_EPOCHS = 4
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1

# Dataset loading configs
HF_MSRP_DATASET = "nyu-mll/glue"
HF_MSRP_CONFIG = "mrpc"
HF_MSRP_SPLITS = ["train", "validation", "test"]

HF_QQP_DATASET = "nyu-mll/glue"
HF_QQP_CONFIG = "qqp"
HF_QQP_SPLITS = ["train", "validation", "test"]

QQP_SAMPLE_SIZE = 20000
RANDOM_SEED = 42

# Web Application & Demo Configs
WEB_HOST = os.getenv("HOST", "127.0.0.1")
WEB_PORT = int(os.getenv("PORT", 5000))
DEMO_REPORT_HTML = RESULTS_DIR / "demo_report.html"

# Evaluation Output
EVAL_RESULTS_CSV = RESULTS_DIR / "evaluation_results.csv"
EVAL_PLOT_PNG = RESULTS_DIR / "model_comparison_f1.png"

# Essay-Level Data Paths (Stage 2)
ESSAYS_DATA_DIR = DATA_DIR / "essays"
AI_DETECTION_TRAIN_PATH = ESSAYS_DATA_DIR / "ai_detection_train.parquet"
AI_DETECTION_VAL_PATH = ESSAYS_DATA_DIR / "ai_detection_validation.parquet"
AI_DETECTION_TEST_PATH = ESSAYS_DATA_DIR / "ai_detection_test.parquet"
DUPLICATE_PAIRS_TRAIN_PATH = ESSAYS_DATA_DIR / "duplicate_pairs_train.parquet"
DUPLICATE_PAIRS_VAL_PATH = ESSAYS_DATA_DIR / "duplicate_pairs_validation.parquet"
DUPLICATE_PAIRS_TEST_PATH = ESSAYS_DATA_DIR / "duplicate_pairs_test.parquet"

# AI Detection Module (Stage 3)
AI_DETECTOR_DIR = MODELS_DIR / "ai_detector"
AI_DETECTOR_TRANSFORMER_PATH = AI_DETECTOR_DIR / "transformer"
AI_DETECTOR_ENSEMBLE_PATH = AI_DETECTOR_DIR / "ensemble_meta_model.joblib"
AI_DETECTOR_LM_NAME = "gpt2"
AI_DETECTOR_TRANSFORMER_NAME = "distilroberta-base"
AI_DETECTOR_FALLBACK_TRANSFORMER = "roberta-base"
AI_DETECTOR_THRESHOLD = 0.5

# Unified Framework (Stage 4)
UNIFIED_FRAMEWORK_WEIGHTS = {
    "semantic_duplicate": 0.5,
    "ai_generated": 0.5,
}