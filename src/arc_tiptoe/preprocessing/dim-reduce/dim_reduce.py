"""
Class for dimensionality reduction methods.
"""

import logging

import numpy as np
import src.arc_tiptoe.preprocessing.dim_reduce.dim_reduce_methods as drm


class DimReduce:
    """Base class for all dimensionality reduction methods."""

    def __init__(self, config):
        self.config = config
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    f"{self.config.data['dataset']}_"
                    f"{self.config.dim_red['dim_red_method']}.log"
                ),
            ],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Starting dim_red process, checking for precomputed")

        if self.config.dim_red_done:
            self.logger.info("Dimensionality reduction already completed.")
        else:
            self.logger.info("Dimensionality reduction not yet completed.")
            self.dim_red_dimension = self.config.dim_red["dim_red_dimension"]
            self.embeddings = np.load(f"{self.config.embeddings_path}/embeddings.npy")
            self.num_clusters = np.ceil(np.sqrt(len(self.embeddings))).astype(int)
            self.dim_red_path = self.config.dim_red_path

    def reduce_dimensions(self):
        """Reduce the dimensions of the embeddings."""
        self.logger.info("transforming embeddings")
        if self.config.dim_red["dim_red_method"] == "pca":
            # PCA dimensionality reduction
            self.logger.info("Train PCA compenents for dim reduction")
            pca_compoenents_path = (
                f"{self.dim_red_path}/pca_{self.dim_red_dimension}.npy"
            )
            drm.train_pca(
                pca_compoenents_path,
                self.embeddings,
                self.dim_red_dimension,
                self.logger,
            )
