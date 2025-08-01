# from sklearn.cluster import MiniBatchKMeans
# import pickle
# import os
# import sys
# import glob
# import re
import logging

import faiss
import numpy as np

# import concurrent

# NUM_CLUSTERS = 35
# NUM_CLUSTERS = 100000
NUM_CLUSTERS = 4 * 32 * 10
DIM = 768
# DIM = 192
MULTI_ASSIGN = 2


def main():
    """Main function to compute centroids for MS MARCO embeddings."""
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting KMeans clustering for MS MARCO embeddings...")
    logging.info("Number of clusters: %d, Dimension: %d", NUM_CLUSTERS, DIM)

    # Load the embeddings from the file
    # url_file = "/work/edauterman/private-search/code/embedding/embeddings_msmarco/msmarco_url.npy"
    embed_file = "/work/edauterman/private-search/code/embedding/embeddings_msmarco/msmarco_embeddings.npy"

    centroids_file = "clustering/centroids/msmarco_centroids.npy"

    data = np.load(embed_file)
    print("Loaded")
    print(data)
    kmeans = faiss.Kmeans(DIM, NUM_CLUSTERS, verbose=True, nredo=3)
    kmeans.train(data.astype(np.float32))
    centroids = kmeans.centroids
    print(centroids)
    np.savetxt(centroids_file, centroids)

    print("Finished kmeans find centroids")


if __name__ == "__main__":
    main()
