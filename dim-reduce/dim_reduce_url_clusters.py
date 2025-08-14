"""Script to perform PCA on URL clusters and save the transformed embeddings."""

import concurrent.futures as cf
import glob
import logging
import os
import re

import numpy as np
from pca import transform_embeddings
from tqdm import tqdm

# New size for the embeddings
NEW_DIM = 192
PCA_COMPONENTS_FILE = f"dim_reduce/dim_reduced/pca_{NEW_DIM}.npy"
DIRECTORY = "clustering/clusters/"
OUT_DIRECTORY = f"clustering/dim_red_assignments/pca_{NEW_DIM}/"


# def main():
#     """Main function to execute the PCA transformation on URL clusters."""
#     logging.basicConfig(
#         format="%(asctime)s - %(message)s",
#         level=logging.INFO,
#         handlers=[logging.StreamHandler()],
#     )
#     logging.info("Starting PCA transformation...")

#     if not os.path.exists(PCA_COMPONENTS_FILE):
#         raise FileNotFoundError(f"PCA components file not found: {PCA_COMPONENTS_FILE}")

#     logging.info("Loading PCA components from %s", PCA_COMPONENTS_FILE)
#     pca_components = np.load(PCA_COMPONENTS_FILE)

#     directories = glob.glob(f"{DIRECTORY}/*")

#     files = []
#     if not os.path.exists(OUT_DIRECTORY):
#         os.mkdir(OUT_DIRECTORY)
#     logging.info("Found directories: %s", directories)
#     for directory in directories:
#         logging.info("Processing directory: %s", directory)
#         files += glob.glob(f"{directory}/clusters/*")
#         if not os.path.exists(re.sub(f"{DIRECTORY}", f"{OUT_DIRECTORY}", directory)):
#             os.mkdir(re.sub(f"{DIRECTORY}", f"{OUT_DIRECTORY}", directory))

#         if not os.path.exists(
#             re.sub(f"{DIRECTORY}", f"{OUT_DIRECTORY}", f"{directory}/clusters")
#         ):
#             os.mkdir(
#                 re.sub(f"{DIRECTORY}", f"{OUT_DIRECTORY}", f"{directory}/clusters")
#             )
#     logging.info("Found files: %s", files)
#     logging.info("Total files to process: %d", len(files))
#     if not files:
#         logging.warning("No files found to process.")
#         return
#     logging.info("Starting to transform embeddings...")
#     with cf.ThreadPoolExecutor(max_workers=32) as executor:
#         for f in tqdm(files, desc="Transforming embeddings"):
#             executor.submit(
#                 transform_embeddings,
#                 pca_components,
#                 f,
#                 re.sub(f"{DIRECTORY}", f"{OUT_DIRECTORY}", f),
#             )

#             executor.submit(
#                 transform_embeddings,
#                 pca_components,
#                 f,
#                 re.sub(f"{DIRECTORY}", f"{OUT_DIRECTORY}", f),
#             )


def main():
    """Transform cluster assignment files using PCA."""
    # Create logger
    logger = logging.getLogger("dim_reduce_url_clusters")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if not os.path.exists(PCA_COMPONENTS_FILE):
        raise FileNotFoundError(f"PCA components file not found: {PCA_COMPONENTS_FILE}")

    logger.info("Loading PCA components...")
    pca_components = np.load(PCA_COMPONENTS_FILE)

    # Create output directory
    os.makedirs(OUT_DIRECTORY, exist_ok=True)

    # Transform each cluster file
    for i in tqdm(range(1280), desc="Transforming clusters"):
        input_file = f"{DIRECTORY}msmarco_cluster_{i}.txt"
        output_file = f"{OUT_DIRECTORY}cluster_{i}.txt"

        if os.path.exists(input_file):
            try:
                transform_embeddings(pca_components, input_file, output_file)
            except (IOError, ValueError, np.linalg.LinAlgError) as e:
                logger.error("Error transforming cluster %d: %s", i, e)
                # Create empty file on error
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write("")
        else:
            # Create empty file if input doesn't exist
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("")

    logger.info("Transformation complete! Files saved to %s", OUT_DIRECTORY)


if __name__ == "__main__":
    main()
