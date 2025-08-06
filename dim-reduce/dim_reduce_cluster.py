"""Dimensionality reduction for clustering assignments using PCA."""

import logging
import os

import numpy as np
from pca import transform_embeddings

# New size for the embeddings
NUM_CLUSTERS = 1280
NEW_DIM = 192
PCA_COMPONENTS_FILE = f"dim_reduce/dim_reduced/pca_{NEW_DIM}.npy"


def main():
    """Main function to transform embeddings using PCA components."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting PCA transformation")
    logger.info("Base directory: %s", PCA_COMPONENTS_FILE)

    # Load the PCA components
    logger.info("Loading PCA components from %s", PCA_COMPONENTS_FILE)
    pca_components = np.load(PCA_COMPONENTS_FILE)

    # Transform each cluster's embeddings
    for i in range(NUM_CLUSTERS):
        input_file = f"clustering/assignments/msmarco_cluster_{i}.txt"
        output_file = f"clustering/dim_red_assigments/pca_192/cluster_{i}.txt"

        if not os.path.exists(output_file):
            logger.info("Transforming embeddings for cluster %d", i)
            transform_embeddings(pca_components, input_file, output_file)
            logger.info("Finished processing cluster %d", i)
        else:
            logger.info("Output file for cluster %d already exists, skipping", i)


if __name__ == "__main__":
    main()
