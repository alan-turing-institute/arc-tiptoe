"""
Load the models for embedding.
"""

import os

from sentence_transformers import SentenceTransformer
from torch import device as torch_device


def load_sentence_transformer(
    model_name: str, device: torch_device
) -> SentenceTransformer:
    """Load a SentenceTransformer model."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logical_cpus = os.cpu_count()
    print(f"Number of cpus: {logical_cpus}")
    os.environ["OMP_NUM_THREADS"] = str(logical_cpus)
    return SentenceTransformer(model_name, device=device)


# ---- Preprocessing for models -------
def distilbert_preprocess(text: str, max_length: int = 512) -> str:
    """
    For the distilbert msmarco model the text needs to be capped at a smaller length
    """
    if not text:
        return ""
    return " ".join(text.split()[:max_length])


def modernbert_preprocess(text: str) -> str:
    """
    For the modernbert embed model this the text needs to be prepended with,
    'search_document: '
    """
    return "search_document: " + text


PREPROCESSING_METHODS = {
    "msmarco-distilbert-base-tas-b": distilbert_preprocess,
    "nomic-ai/modernbert-embed-base": modernbert_preprocess,
    "google/embeddinggemma-300m": distilbert_preprocess,
}
