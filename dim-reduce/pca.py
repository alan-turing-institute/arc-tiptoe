"""PCA methods for dimensionality reduction."""

import os

import numpy as np
from sklearn.decomposition import PCA

NEW_DIM = 192
NUM_CLUSTERS = 1280
PCA_COMPONENTS_FILE = f"dim_reduce/dim_reduced/pca_{NEW_DIM}.npy"


def train_pca(train_vecs):
    """Train PCA on the provided vectors."""
    pca = PCA(n_components=NEW_DIM, svd_solver="full")
    pca.fit(train_vecs)
    return pca


def run_pca(pca_components, vecs):
    """Run PCA on the provided vectors using the given components."""
    return np.clip(np.round(np.matmul(vecs, pca_components) / 10), -16, 15)


def adjust_precision(vec):
    """Adjust the precision of the vector to fit within the required dimensions."""
    return np.round(np.array(vec) * (1 << 5))


def transform_embeddings(pca_components, in_file, out_file, logger=None):
    """Transform embeddings using PCA components."""
    with open(in_file, "r", encoding="utf-8") as f:
        lines = [line for line in f.readlines() if line.strip()]
    if len(lines) == 0:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("")
        return
    docids, in_embeddings_text, urls = zip(*(line.split(" | ") for line in lines))
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
        lines = []
        # convert to integers
        for i in range(len(out_embeddings)):
            embed_str = ",".join([str(int(ch)) for ch in out_embeddings[i]])
            line = f"{docids[i]} | {embed_str} | {urls[i].strip()}\n"
            lines.append(line)
        if logger:
            logger.info(
                "Writing %d transformed embeddings to %s", len(out_embeddings), out_file
            )
        else:
            print(f"Writing {len(out_embeddings)} transformed embeddings to {out_file}")
        f.write("".join(lines))
