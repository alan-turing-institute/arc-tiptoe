"""
Build a FAISS index for the given embeddings.
"""

import faiss
import numpy as np

CENTROIDS_FILE = "/home/azureuser/data/embeddings/centroids.txt"
FAISS_OUT_FILE = "/home/azureuser/data/artifact/dim192/index.faiss"


def main():
    """Main function to build FAISS index."""
    centroids = np.loadtxt(CENTROIDS_FILE)
    d = centroids.shape[1]
    print(f"Centroids shape: {d}")

    index = faiss.IndexFlatIP(d)
    index.add(centroids)
    faiss.write_index(index, FAISS_OUT_FILE)


if __name__ == "__main__":
    main()
