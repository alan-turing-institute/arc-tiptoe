"""
Clustering methods. Includes:
- Kmeans

TODO:
- HDBSCAN
- Agglomerative Clustering
"""

import faiss
import numpy as np


# KMeans clustering
def kmeans_centroids(embeddings: np.ndarray, num_clusters: int) -> np.ndarray:
    """Perform KMeans clustering on the given embeddings."""
    dim = embeddings.shape[1]
    kmeans = faiss.Kmeans(dim, num_clusters, verbose=True, nredo=3)
    kmeans.train(embeddings.astype(np.float32))
    return kmeans.centroids


def kmeans_sub_cluster(embed_contents: np.ndarray, num_bundles: int, n_redo=1):
    """Sub-clustering the embedded contents using kmeans"""
    dim = embed_contents.shape[1]
    kmeans = faiss.Kmeans(dim, num_bundles, verbose=True, nredo=n_redo)
    kmeans.train(embed_contents.astype(np.float32))
    centroids = kmeans.centroids
    _, assignments = kmeans.index.search(embed_contents.astype(np.float32), 1)
    return centroids, assignments
