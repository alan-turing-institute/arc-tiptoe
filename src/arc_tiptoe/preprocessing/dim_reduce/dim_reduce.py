"""
Classes for dimensionality reduction methods.

TODO:
- Generalise train method?
- Add UMAP method
- Add t-SNE method
"""

import logging
import os
from abc import ABC, abstractmethod

import numpy as np

import arc_tiptoe.preprocessing.dim_reduce.dim_reduce_methods as drm

# from tqdm import tqdm


# from glob import glob


class DimReducer(ABC):
    """Base class for all dimensionality reduction methods."""

    def __init__(self, config, within_pipeline: bool = False):
        self.config = config
        self.within_pipeline = within_pipeline
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    f"data/{self.config.uuid}/"
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

    @abstractmethod
    def transform_embedding(self, embeddings):
        """Transform a single embedding using the dimensionality reduction method."""
        raise NotImplementedError()

    @abstractmethod
    def _transform_embeddings(self):
        """Transform a single embedding"""
        raise NotImplementedError()

    @abstractmethod
    def _reduce_dimensions(self):
        """Reduce the dimensions of the embeddings."""
        self.logger.info("transforming embeddings")
        raise NotImplementedError()

    def reduce_dimensions(self):
        """Public method to reduce dimensions and update config."""
        self._reduce_dimensions()
        self.config.dim_red_done = True
        self.config.save_config()
        if self.within_pipeline:
            return self.config

        return 1


class DimReducePCA(DimReducer):
    """Dimensionality reduction using PCA."""

    def __init__(self, config, within_pipeline: bool = False):
        super().__init__(config, within_pipeline)
        self.logger.info("Initialized PCA dimensionality reduction")
        self.pca_components = None
        self.pca_components_path = (
            f"{self.dim_red_path}/"
            f"{self.config.dim_red['dim_red_method']}_"
            f"{self.dim_red_dimension}.npy"
        )

    def _train_pca(self):
        """Train pca and save the componenents path"""
        drm.train_pca(
            self.pca_components_path,
            self.embeddings,
            self.dim_red_dimension,
            self.logger,
        )
        self.pca_components = np.load(self.pca_components_path)

    def _transform_embeddings(self):
        """Transform the embeddings pre-clustering"""
        self.logger.info("Transform embeddings using PCA components")
        output_file = (
            f"{self.dim_red_path}/"
            f"{self.config.dim_red['dim_red_method']}_"
            f"{self.dim_red_dimension}/embeddings.py"
        )
        drm.transform_embeddings(self.pca_components, self.embeddings, output_file)

    def _transform_clustered_embeddings(self, idx):
        """Transform a single embedding using PCA components."""
        input_file = f"{self.config.clustering_path}/assignments/cluster_{idx}.txt"
        output_file = (
            f"{self.dim_red_path}/"
            f"{self.config.dim_red['dim_red_method']}_"
            f"{self.dim_red_dimension}/cluster_{idx}.txt"
        )

        if not os.path.exists(output_file):
            self.logger.info("Transforming embeddings for cluster %d", idx)
            drm.transform_embeddings(self.pca_components, input_file, output_file)
            self.logger.info("Finished processing cluster %d", idx)
        else:
            self.logger.info("Output file for cluster %d already exists, skipping", idx)

    def _reduce_dimensions(self):
        """Reduce the dimensions of the embeddings using PCA."""
        self.logger.info("Train PCA compenents for dim reduction")

        # Transform the embeddings using the PCA components
        self._transform_embeddings()

    def transform_embedding(self, embeddings):
        """Transform a single embedding using PCA components."""
        if self.pca_components is None:
            if os.path.exists(self.pca_components_path):
                self.pca_components = np.load(self.pca_components_path)
            else:
                self._train_pca()
        return drm.run_pca(self.pca_components, embeddings)
