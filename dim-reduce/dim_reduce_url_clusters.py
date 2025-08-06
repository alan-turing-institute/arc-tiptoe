"""Script to perform PCA on URL clusters and save the transformed embeddings."""

import concurrent.futures as cf
import glob
import logging
import os
import re

import numpy as np
from pca import transform_embeddings

# New size for the embeddings
NEW_DIM = 192
PCA_COMPONENTS_FILE = f"dim_reduce_dim_reduced/pca_{NEW_DIM}.npy"
DIRECTORY = "clustering/assignments/"
OUT_DIRECTORY = f"clustering/dim_red_assignments/pca_{NEW_DIM}/"


def main():
    """Main function to execute the PCA transformation on URL clusters."""
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )
    logging.info("Starting PCA transformation...")

    if not os.path.exists(PCA_COMPONENTS_FILE):
        raise FileNotFoundError(f"PCA components file not found: {PCA_COMPONENTS_FILE}")

    logging.info("Loading PCA components from %s", PCA_COMPONENTS_FILE)
    pca_components = np.load(PCA_COMPONENTS_FILE)

    directories = glob.glob(f"{DIRECTORY}/*")

    files = []
    if not os.path.exists(OUT_DIRECTORY):
        os.mkdir(OUT_DIRECTORY)
    logging.info("Found directories: %s", directories)
    for directory in directories:
        logging.info("Processing directory: %s", directory)
        files += glob.glob(f"{directory}/clusters/*")
        if not os.path.exists(re.sub(f"{DIRECTORY}", f"{OUT_DIRECTORY}", directory)):
            os.mkdir(re.sub(f"{DIRECTORY}", f"{OUT_DIRECTORY}", directory))

        if not os.path.exists(
            re.sub(f"{DIRECTORY}", f"{OUT_DIRECTORY}", f"{directory}/clusters")
        ):
            os.mkdir(
                re.sub(f"{DIRECTORY}", f"{OUT_DIRECTORY}", f"{directory}/clusters")
            )
    logging.info("Found files: %s", files)
    logging.info("Total files to process: %d", len(files))
    if not files:
        logging.warning("No files found to process.")
        return
    logging.info("Starting to transform embeddings...")
    with cf.ThreadPoolExecutor(max_workers=32) as executor:
        for f in files:
            executor.submit(
                transform_embeddings,
                pca_components,
                f,
                re.sub(f"{DIRECTORY}", f"{OUT_DIRECTORY}", f),
            )

            executor.submit(
                transform_embeddings,
                pca_components,
                f,
                re.sub(f"{DIRECTORY}", f"{OUT_DIRECTORY}", f),
            )


if __name__ == "__main__":
    main()
