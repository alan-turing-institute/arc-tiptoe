import logging
import os

import numpy as np
from sklearn.decomposition import PCA

# New size for the embeddings
NEW_DIM = 192
NUM_CLUSTERS = 1280
PCA_COMPONENTS_FILE = f"dim_reduce/dim_reduced/pca_{NEW_DIM}.npy"


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

    # create the directory if it does not exist
    if not os.path.exists(os.path.dirname(PCA_COMPONENTS_FILE)):
        os.makedirs(os.path.dirname(PCA_COMPONENTS_FILE))

    # Load the training embeddings
    logger.info("Loading training embeddings")
    train_embeddings = np.load("embedding/embeddings/msmarco_combined_embeddings.npy")

    # Choose a subset of the embeddings for training
    subset_size = 500000

    rng = np.random.default_rng(42)  # For reproducibility
    if len(train_embeddings) > subset_size:
        logger.info("Using a subset of size %d", subset_size)
        indices = rng.choice(len(train_embeddings), size=subset_size, replace=False)
        train_embeddings = train_embeddings[indices]
    else:
        logger.info("Using all embeddings, size: %d", len(train_embeddings))
    logger.info("Loaded %d embeddings", len(train_embeddings))

    # Adjust the precision of the embeddings
    train_embeddings = [adjust_precision(embed) for embed in train_embeddings]
    logger.info("Loaded and adjusted precision")

    # Train PCA
    logger.info("Training PCA with %d components", NEW_DIM)
    pca = PCA(n_components=NEW_DIM, random_state=42)
    pca.fit(train_embeddings)
    logger.info("Ran PCA")

    # Save the PCA components
    logger.info("Saving PCA components to %s", PCA_COMPONENTS_FILE)
    np.save(PCA_COMPONENTS_FILE, np.transpose(pca.components_))
    logger.info("PCA components saved successfully")

    # Save the mean for later use
    logger.info("Saving PCA mean")
    np.save("dim_reduce/dim_reduced/pca_mean.npy", pca.mean_)


if __name__ == "__main__":
    main()
