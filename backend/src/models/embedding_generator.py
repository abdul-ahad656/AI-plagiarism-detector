from sentence_transformers import SentenceTransformer
import numpy as np

# Load the high-performance sentence embedding model
model = SentenceTransformer('all-mpnet-base-v2')  # Ensure consistency across your app

def get_embeddings(text, as_tensor=False):
    """
    Generates a semantic embedding for the given text input.

    Args:
        text (str): Input string or paragraph to embed.
        as_tensor (bool): If True, returns PyTorch tensor. Otherwise, NumPy array.

    Returns:
        torch.Tensor or np.ndarray: The text embedding.
    """
    if not text.strip():
        raise ValueError("Input text is empty or blank.")

    embedding = model.encode(text, convert_to_tensor=as_tensor)
    return embedding
