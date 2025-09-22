from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd
import os

def load_quora_data(path, limit=10000):
    """
    Loads and formats the Quora Question Pairs dataset.

    Each row is turned into a training pair with a similarity label:
    - 1.0 for duplicate (semantically similar)
    - 0.0 for non-duplicate (not similar)

    Args:
        path (str): Path to the dataset CSV file.
        limit (int): Max number of samples to load (for faster training/testing).

    Returns:
        List[InputExample]: SentenceTransformer-compatible training examples.
    """
    df = pd.read_csv(path)
    df = df.dropna(subset=["question1", "question2", "is_duplicate"]).head(limit)

    examples = [
        InputExample(
            texts=[row["question1"], row["question2"]],
            label=float(row["is_duplicate"])
        )
        for _, row in df.iterrows()
    ]
    return examples

def fine_tune_model(
    data_path="data/raw/quora_duplicate_questions.csv",
    output_path="models/fine_tuned_mpnet",
    epochs=1,
    batch_size=16
):
    """
    Fine-tunes 'all-mpnet-base-v2' on Quora sentence pairs to learn semantic similarity.

    Uses CosineSimilarityLoss to make similar sentences closer in vector space.

    Args:
        data_path (str): CSV file path of training data.
        output_path (str): Directory to save the trained model.
        epochs (int): Number of training passes over the dataset.
        batch_size (int): Number of pairs processed together per training step.
    """
    print("[INFO] Loading high-accuracy MPNet model...")
    model = SentenceTransformer('all-mpnet-base-v2')

    print(f"[INFO] Reading dataset from: {data_path}")
    train_examples = load_quora_data(data_path)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    print("[INFO] Setting up cosine similarity loss...")
    train_loss = losses.CosineSimilarityLoss(model=model)

    print("[INFO] Starting training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100
    )

    os.makedirs(output_path, exist_ok=True)
    model.save(output_path)
    print(f"[INFO] Training complete. Model saved to: {output_path}")

if __name__ == "__main__":
    fine_tune_model(
        data_path="data/raw/quora_duplicate_questions.csv",
        output_path="models/fine_tuned_mpnet",
        epochs=1,
        batch_size=16
    )
