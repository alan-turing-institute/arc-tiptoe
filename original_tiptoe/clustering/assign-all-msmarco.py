# from sklearn.cluster import MiniBatchKMeans
# import os
# import sys
# import glob
# import re
import logging

import faiss
import numpy as np
from tqdm import tqdm

# import pickle


# import concurrent

NUM_CLUSTERS = 4 * 32 * 10
DIM = 768
# DIM = 192
MULTI_ASSIGN = 2


def main():
    """Main function to assign clusters to MS MARCO embeddings."""
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting KMeans cluster assignment for MS MARCO embeddings...")
    logging.info("Number of clusters: %d, Dimension: %d", NUM_CLUSTERS, DIM)

    # Load the embeddings and centroids
    url_file = "embedding/embeddings/msmarco_combined_docids.npy"
    embed_file = "embedding/embeddings/msmarco_combined_embeddings.npy"
    centroids_file = "clustering/centroids/msmarco_centroids.npy"

    logging.info("Loading embeddings from %s", embed_file)
    logging.info("Loading centroids from %s", centroids_file)

    data = np.load(embed_file)

    try:
        centroids = np.load(centroids_file)  # Try .npy format first
        logging.info("Loaded centroids from .npy file")
    except ValueError:
        # If that fails, try text format
        centroids = np.loadtxt(centroids_file)
        logging.info("Loaded centroids from text file")

    logging.info("Loaded centroids shape: %s", centroids.shape)
    logging.info("Data shape: %s", data.shape)
    logging.info("Number of embeddings: %d", data.shape[0])

    logging.info("starting kmeans assignment")
    cluster_files = [
        (f"clustering/assignments/msmarco_cluster_{i}.txt") for i in range(NUM_CLUSTERS)
    ]
    assignment_dict = {}
    logging.info("Initialized cluster files and assignment dictionary.")

    # Create a FAISS index for the centroids
    index = faiss.IndexFlatL2(DIM)
    if centroids.ndim == 1:
        centroids = centroids.reshape(1, -1)
    index.add(centroids.astype(np.float32))
    distances, assignments = index.search(data.astype(np.float32), MULTI_ASSIGN)

    logging.info("Finished kmeans assignment")

    urls = np.load(url_file)

    percentiles = []
    for i in range(1, MULTI_ASSIGN):
        percentiles.append(
            np.percentile([(dist[i] - dist[0]) for dist in distances], 20)
        )

    over_assign_count = 0
    for i in range(len(assignments)):
        for k in range(MULTI_ASSIGN):
            if (k == 0) or (
                k > 0 and (distances[i][k] - distances[i][0]) < percentiles[k - 1]
            ):
                cluster = assignments[i][k]
                if cluster not in assignment_dict:
                    assignment_dict[cluster] = [i]
                else:
                    assignment_dict[cluster].append(i)
                if k > 0:
                    over_assign_count += 1

    for i in tqdm(range(NUM_CLUSTERS), desc="Writing clusters", unit="cluster"):
        with open(cluster_files[i], "w") as f:
            if i in assignment_dict:
                for idx in assignment_dict[i]:
                    embed = data[idx]
                    url = urls[idx]
                    embstr = ",".join(["%f" % ch for ch in embed])
                    doc_id = idx
                    data_str = "%d | %s | %s\n" % (doc_id, embstr, url)
                    f.write(data_str + "\n")
            else:
                print("Not in assignment dict %d" % i)

    print("Over assigning for param %d = %d" % (MULTI_ASSIGN, over_assign_count))


if __name__ == "__main__":
    main()
