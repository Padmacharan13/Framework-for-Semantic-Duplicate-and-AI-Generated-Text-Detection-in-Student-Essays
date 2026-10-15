import pandas as pd
from config import MSRP_PROCESSED_PATH, QQP_PROCESSED_PATH, WORD2VEC_MODEL_PATH
from word2vec_engine import train_word2vec

def main():
    print("[1/3] Loading training texts...", flush=True)
    texts = []
    for path in [MSRP_PROCESSED_PATH, QQP_PROCESSED_PATH]:
        if path.exists():
            df = pd.read_parquet(path)
            if "split" in df.columns:
                train_df = df[df["split"] == "train"]
            else:
                train_df = df

            col1 = "text1" if "text1" in train_df.columns else "text_a"
            col2 = "text2" if "text2" in train_df.columns else "text_b"

            texts.extend(train_df[col1].dropna().tolist())
            texts.extend(train_df[col2].dropna().tolist())

    print(f"[2/3] Training Word2Vec model on {len(texts):,} sentences...", flush=True)
    train_word2vec(texts, save_path=WORD2VEC_MODEL_PATH)
    print(f"[3/3] Word2Vec successfully trained and saved to {WORD2VEC_MODEL_PATH}", flush=True)

if __name__ == "__main__":
    main()