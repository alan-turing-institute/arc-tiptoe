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


def kmeans_embed_contents(contents: list[str], num_bundles: int, logger=None):
    """Embed the contents using FAISS."""
    if logger:
        logger.info("Embedding contents")
    embed_contents = [elem[1] for elem in contents]
    data = np.loadtxt(embed_contents, delimiter=",")
    kmeans = faiss.Kmeans(data.shape[1], num_bundles, nredo=3)
    if len(data) >= 1 and len(np.shape(data)) == 2:
        kmeans.train(data.astype(np.float32))
        centroids = kmeans.centroids
        _, assignments = kmeans.index.search(data.astype(np.float32), 1)
    else:
        centroids = np.zeros((1, data.shape[1]))
        assignments = [[0]]
    return centroids, assignments
