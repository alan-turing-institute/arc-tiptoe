"""
Clustering methods. Includes:
- Kmeans

TODO:
- HDBSCAN
- Agglomerative Clustering
"""

import faiss
import numpy as np


def kmeans_clustering(embeddings: np.ndarray, num_clusters: int) -> np.ndarray:
    """Perform KMeans clustering on the given embeddings."""
    dim = embeddings.shape[1]
    kmeans = faiss.Kmeans(dim, num_clusters, verbose=True, nredo=3)
    kmeans.train(embeddings.astype(np.float32))
    return kmeans.centroids
