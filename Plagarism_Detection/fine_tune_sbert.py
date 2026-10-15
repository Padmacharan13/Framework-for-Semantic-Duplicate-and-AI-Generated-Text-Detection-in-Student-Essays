import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.sentence_transformer import losses, evaluation
from config import (
    MSRP_PROCESSED_PATH,
    QQP_PROCESSED_PATH,
    SBERT_BASE_MODEL,
    SBERT_FINETUNED_PATH,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    LEARNING_RATE,
    WARMUP_RATIO
)

def load_data_split(split_name: str, positive_only: bool = False):
    examples = []
    val_s1, val_s2, val_scores = [], [], []

    for path in [MSRP_PROCESSED_PATH, QQP_PROCESSED_PATH]:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "split" in df.columns:
            sub_df = df[df["split"] == split_name]
        else:
            sub_df = df

        col1 = "text1" if "text1" in sub_df.columns else "text_a"
        col2 = "text2" if "text2" in sub_df.columns else "text_b"

        if positive_only:
            sub_df = sub_df[sub_df["label"] == 1]
            for _, row in sub_df.iterrows():
                examples.append(InputExample(texts=[str(row[col1]), str(row[col2])]))
        else:
            for _, row in sub_df.iterrows():
                val_s1.append(str(row[col1]))
                val_s2.append(str(row[col2]))
                val_scores.append(float(row["label"]))

    return examples, (val_s1, val_s2, val_scores)

def main():
    print("[1/4] Preparing training data with MultipleNegativesRankingLoss...")
    train_examples, _ = load_data_split("train", positive_only=True)
    _, (val_s1, val_s2, val_scores) = load_data_split("validation", positive_only=False)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=TRAIN_BATCH_SIZE)
    model = SentenceTransformer(SBERT_BASE_MODEL)

    train_loss = losses.MultipleNegativesRankingLoss(model)
    evaluator = evaluation.EmbeddingSimilarityEvaluator(
        val_s1, val_s2, val_scores, batch_size=TRAIN_BATCH_SIZE, name="val_similarity"
    )

    warmup_steps = int(len(train_dataloader) * TRAIN_EPOCHS * WARMUP_RATIO)
    print(f"[2/4] Fine-tuning {SBERT_BASE_MODEL} for {TRAIN_EPOCHS} epochs ({warmup_steps} warmup steps)...")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=TRAIN_EPOCHS,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": LEARNING_RATE},
        output_path=str(SBERT_FINETUNED_PATH),
        evaluation_steps=250,
        save_best_model=True
    )
    print(f"[3/4] Fine-tuning complete. Model saved to: {SBERT_FINETUNED_PATH}")

if __name__ == "__main__":
    main()