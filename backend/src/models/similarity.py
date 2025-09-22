import torch
import numpy as np
from torch.nn.functional import cosine_similarity as torch_cosine_similarity

def cosine_similarity(emb1, emb2):
    """
    Computes the cosine similarity between two text embeddings.

    Args:
        emb1, emb2: Either PyTorch tensors or NumPy arrays representing sentence embeddings.

    Returns:
        float: Cosine similarity score between -1.0 (opposite) and 1.0 (identical).
    """
    # Convert NumPy arrays to PyTorch tensors if needed
    if isinstance(emb1, np.ndarray):
        emb1 = torch.tensor(emb1, dtype=torch.float32)
    if isinstance(emb2, np.ndarray):
        emb2 = torch.tensor(emb2, dtype=torch.float32)

    # Ensure both are 1D tensors of the same shape
    if emb1.shape != emb2.shape:
        raise ValueError(f"Shape mismatch: {emb1.shape} vs {emb2.shape}")

    # Compute cosine similarity (dot product normalized)
    return torch_cosine_similarity(emb1, emb2, dim=0).item()
