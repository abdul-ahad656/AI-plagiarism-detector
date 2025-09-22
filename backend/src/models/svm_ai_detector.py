import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import joblib
import os

def train_svm_model():
    # === File Paths ===
    data_path = os.path.join("data", "raw", "ai_human_text.csv")
    output_path = os.path.join("models", "svm", "svm_ai_detector.pkl")

    # === Load and Prepare Data ===
    print(f"[INFO] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")

    X = df["text"]
    y = df["label"]

    # === Build Training Pipeline ===
    print("[INFO] Creating pipeline: TF-IDF → SVM")
    pipeline = make_pipeline(
        TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
        SVC(kernel='linear', probability=True)
    )

    # === Train the Model ===
    print("[INFO] Training SVM model...")
    pipeline.fit(X, y)
    print("[INFO] Training complete.")

    # === Save the Model ===
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"[INFO] Model saved to: {output_path}")

# === Run the training only if executed directly ===
if __name__ == "__main__":
    train_svm_model()
