"""
Dimentionality reduction methods. Includes:
- PCA

TODO:
- t-SNE?
- UMAP?
- GemmaEmbed dim reduction?
"""

import os

import numpy as np
from sklearn.decomposition import PCA


## PCA Methods
def _train_pca(train_vecs, new_dim):
    """Train PCA on the provided vectors."""
    pca = PCA(n_components=new_dim, svd_solver="full")
    pca.fit(train_vecs)
    return pca


def run_pca(pca_components, vecs):
    """Run PCA on the provided vectors using the given components. Quantize values."""
    # Apply PCA
    transformed = np.matmul(vecs, pca_components)

    # Adaptive scaling to fit within int8 range
    data_min = np.min(transformed)
    data_max = np.max(transformed)
    data_range = max(abs(data_min), abs(data_max))

    # TODO: generalise quantisation
    scale_factor = 127.0 / data_range
    quantized = np.clip(np.round(transformed * scale_factor), -127, 127)

    return quantized.astype(np.int8)


def adjust_precision(vec):
    """Adjust the precision of the vector to fit within the required dimensions."""
    return np.round(np.array(vec) * (1 << 5))


def train_pca(pca_compents_file, train_vecs, new_dim, logger=None):
    """Train PCA on the provided vectors."""
    if logger:
        logger.info("Training PCA with %d components", new_dim)

    # create directory if it does not exist
    if not os.path.exists(os.path.dirname(pca_compents_file)):
        os.makedirs(os.path.dirname(pca_compents_file))

    # Choose subset of embeddings for training
    subset_size = 500000
    rng = np.random.default_rng(42)  # For reproducibility

    if len(train_vecs) > subset_size:
        if logger:
            logger.info("Using a subset of size %d", subset_size)
        indices = rng.choice(len(train_vecs), size=subset_size, replace=False)
        train_vecs = train_vecs[indices]
    else:
        if logger:
            logger.info("Using all embeddings, size: %d", len(train_vecs))
    if logger:
        logger.info("Loaded %d embeddings", len(train_vecs))

    # Adjust the precision of the embeddings
    train_vecs = [adjust_precision(embed) for embed in train_vecs]
    if logger:
        logger.info("Loaded and adjusted precision")

    # Train PCA
    pca = _train_pca(train_vecs, new_dim)
    if logger:
        logger.info("Ran PCA")

    # Save the PCA components
    if logger:
        logger.info("Saving PCA components to %s", pca_compents_file)
    np.save(pca_compents_file, np.transpose(pca.components_))
    if logger:
        logger.info("PCA components saved successfully")


def transform_embeddings(
    pca_components: np.ndarray,
    embeddings: np.ndarray,
    out_file: str | None = None,
    logger=None,
):
    """Transform embeddings using PCA components"""
    if logger:
        logger.info("Transforming embeddings")
    out_embeddings = run_pca(pca_components, embeddings)
    print(
        "check not zero here 2",
        np.sum(pca_components),
        np.sum(embeddings),
        np.sum(out_embeddings),
    )
    if logger:
        logger.info("Transformed embeddings")
    if out_file is not None:
        np.save(out_file, out_embeddings)
        return 1
    return out_embeddings
