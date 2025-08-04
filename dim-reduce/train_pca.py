# from sentence_transformers import SentenceTransformer, LoggingHandler, util, evaluation, models, InputExample
# from sklearn.preprocessing import normalize
import logging

# import os
# import gzip
# import csv
# import random
# import numpy as np
# import torch
import numpy as np
from sklearn.decomposition import PCA

# import sys
# import glob
# import re
# import concurrent.futures

# New size for the embeddings
NEW_DIM = 192
NUM_CLUSTERS = 1280
PCA_COMPONENTS_FILE = f"/dim_reduce/dim_reduced/pca_{NEW_DIM}.npy"


def train_pca(train_vecs):
    """Train a PCA model on the provided vectors.

    Args:
        train_vecs: The training vectors to fit the PCA model.

    Returns:
        The trained PCA model.
    """
    pca = PCA(n_components=NEW_DIM, svd_solver="full")
    pca.fit(train_vecs)
    return pca


def adjust_precision(vec):
    """Adjust the precision of the vector to fit within the required dimensions."""
    return np.round(np.array(vec) * (1 << 5))


def main():
    """Main function to train PCA on the embeddings."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting PCA training")
    logger.info("Base directory: %s", PCA_COMPONENTS_FILE)

    # Load the training embeddings
    logger.info("Loading training embeddings")
    train_embeddings = np.load("embedding/embeddings/msmarco_combined_embeddings.npy")
    train_embeddings = [adjust_precision(embed) for embed in train_embeddings]
    logger.info("Loaded and adjusted precision")
    pca = train_pca(train_embeddings)
    logger.info("Ran PCA")
    np.save(PCA_COMPONENTS_FILE, np.transpose(pca.components_))


if __name__ == "__main__":
    main()
