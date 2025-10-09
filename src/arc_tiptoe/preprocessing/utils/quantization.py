import numpy as np


def quantize_query_embedding(embedding: np.ndarray) -> np.ndarray:
    # Quantize embedding
    data_min = np.min(embedding)
    data_max = np.max(embedding)
    data_range = max(abs(data_min), abs(data_max))
    scale = 127.0 / data_range
    return np.clip(np.round(embedding * scale), -127, 127).astype(np.int8)
