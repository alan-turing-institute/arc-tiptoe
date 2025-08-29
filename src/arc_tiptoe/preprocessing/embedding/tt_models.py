"""
Load the models for embedding.
"""

from sentence_transformers import SentenceTransformer
from torch import device as torch_device


def load_sentence_transformer(
    model_name: str, device: torch_device
) -> SentenceTransformer:
    """Load a SentenceTransformer model."""
    return SentenceTransformer(model_name, device=device)
