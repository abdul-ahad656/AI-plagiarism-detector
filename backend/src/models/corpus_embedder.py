import os
import fitz
from docx import Document
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

# Location of all documents to embed
CORPUS_DIR = "data/corpus"

# Where to save the generated embeddings
OUTPUT_PATH = "models/embeddings/corpus_embeddings.pkl"

# Load the same model you're using for comparison/search
model = SentenceTransformer("all-mpnet-base-v2")  # Consistent with your backend

def extract_text_from_file(filepath):
    """
    Extracts readable text from supported document formats.
    Supported: .txt, .pdf, .docx
    """
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == ".txt":
            with open(filepath, 'r', encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif ext == ".pdf":
            text = ""
            doc = fitz.open(filepath)
            for page in doc:
                text += page.get_text()
            return text
        elif ext == ".docx":
            doc = Document(filepath)
            return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    
    return ""  # Return empty string for unsupported or failed files

def create_corpus_embeddings():
    """
    Iterates over all documents in the corpus folder,
    extracts text from each, generates its embedding,
    and saves the full embedding list + filenames to a .pkl file.
    """
    embeddings = []
    filenames = []

    print(f"[INFO] Scanning documents in: {CORPUS_DIR}")
    for fname in os.listdir(CORPUS_DIR):
        full_path = os.path.join(CORPUS_DIR, fname)

        # Ensure it's a file (skip subfolders)
        if os.path.isfile(full_path):
            text = extract_text_from_file(full_path)

            if text.strip():  # Skip empty files
                emb = model.encode(text)
                embeddings.append(emb)
                filenames.append(fname)
                print(f"Embedded → {fname}")
            else:
                print(f"Skipped (empty/unsupported): {fname}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Save filename list and embedding array together
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump((filenames, np.array(embeddings)), f)

    print(f"\nDone! {len(embeddings)} documents embedded.")
    print(f"Embeddings saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    create_corpus_embeddings()
