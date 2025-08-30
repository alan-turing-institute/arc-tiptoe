"""Dimensionality Reduction without Clustering using PCA."""

import logging

import numpy as np
from pca import adjust_precision

# New size for the embeddings
NEW_DIM = 192
PCA_COMPONENTS_FILE = f"dim_reduce/dim_reduced/pca_{NEW_DIM}.npy"
PCA_EMBEDDINGS_FILE = f"dim_reduce/dim_reduced/pca_embeddings_{NEW_DIM}.npy"
EMBEDDINGS_FILES = "embedding/embeddings/msmarco_combined_embeddings.npy"


def main():
    """Main function to run PCA on the embeddings."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting PCA transformation")
    logger.info("Base directory: %s", PCA_COMPONENTS_FILE)

    # Load the PCA components
    logger.info("Loading PCA components from %s", PCA_COMPONENTS_FILE)
    pca_components = np.load(PCA_COMPONENTS_FILE)

    # Load and transform the embeddings
    logger.info("Loading embeddings from %s", EMBEDDINGS_FILES)
    embeds = np.load(EMBEDDINGS_FILES)
    logger.info("Loaded %d embeddings", len(embeds))
    embeds = [adjust_precision(embed) for embed in embeds]
    logger.info("Adjusted precision of embeddings")
    out_embeddings = np.clip(np.round(np.matmul(embeds, pca_components) / 10), -16, 15)

    logger.info("Transformed embeddings using PCA")
    logger.info("Saving transformed embeddings to %s", PCA_EMBEDDINGS_FILE)
    np.save(PCA_EMBEDDINGS_FILE, out_embeddings)
    logger.info("Transformed embeddings saved successfully to %s", PCA_EMBEDDINGS_FILE)


if __name__ == "__main__":
    main()
