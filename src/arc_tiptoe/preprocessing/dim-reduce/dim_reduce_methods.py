"""
Dimentionality reduction methods. Includes:
- PCA

TODO:
- t-SNE
- UMAP
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
    """Run PCA on the provided vectors using the given components."""
    return np.clip(np.round(np.matmul(vecs, pca_components) / 10), -16, 15).astype(
        np.int8
    )


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


def transform_embeddings(pca_components, in_file, out_file, logger=None):
    """Transform embeddings using PCA components."""
    with open(in_file, "r", encoding="utf-8") as f:
        lines = [line for line in f.readlines() if line.strip()]
    if len(lines) == 0:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("")
        return
    docids, in_embeddings_text, urls = zip(
        *(line.split(" | ") for line in lines), strict=True
    )
    in_embeddings = [
        [float(i) for i in embed.split(",")] for embed in in_embeddings_text
    ]
    if logger:
        logger.info("Loaded %d embeddings from %s", len(in_embeddings), in_file)
    else:
        print(f"Loaded {len(in_embeddings)} embeddings from {in_file}")
    if logger:
        logger.info("in file = %s", in_file)
    else:
        print(f"in file = {in_file}")
    if logger:
        logger.info(
            "len of embeddings = %d, len of lines =%d, len of in_embeddings_text = %d",
            len(in_embeddings),
            len(lines),
            len(in_embeddings_text),
        )
    else:
        print(
            f"len of embeddings = {len(in_embeddings)}, len of lines = {len(lines)}"
            f", len of in_embeddings_text = {len(in_embeddings_text)}",
        )
    in_embeddings = [adjust_precision(embed) for embed in in_embeddings]
    out_embeddings = run_pca(pca_components, in_embeddings)

    # create file to write to
    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))
    with open(out_file, "w", encoding="utf-8") as f:
        for docid, embed, url in zip(docids, out_embeddings, urls):
            f.write(f"{docid} | {','.join(map(str, embed))} | {url}")
