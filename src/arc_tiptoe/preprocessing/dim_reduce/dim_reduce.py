"""
Classes for dimensionality reduction methods.
"""

import logging
import os
from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm

import arc_tiptoe.preprocessing.dim_reduce.dim_reduce_methods as drm


class DimReduce(ABC):
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

    @abstractmethod
    def _transform_embeddings(self):
        raise NotImplementedError()

    @abstractmethod
    def _transform_urls_pca(self):
        raise NotImplementedError()

    @abstractmethod
    def reduce_dimensions(self):
        """Reduce the dimensions of the embeddings."""
        self.logger.info("transforming embeddings")
        raise NotImplementedError()


class DimReducePCA(DimReduce):
    """Dimensionality reduction using PCA."""

    def __init__(self, config):
        super().__init__(config)
        self.logger.info("Initialized PCA dimensionality reduction")
        self.pca_components = None

    def _transform_embedding(self, idx):
        """Transform a single embedding using PCA components."""
        input_file = f"{self.config.clustering_path}/cluster_{idx}.txt"
        output_file = (
            f"{self.dim_red_path}/pca_{self.dim_red_dimension}/cluster_{idx}.txt"
        )

        if not os.path.exists(output_file):
            self.logger.info("Transforming embeddings for cluster %d", idx)
            drm.transform_embeddings(self.pca_components, input_file, output_file)
            self.logger.info("Finished processing cluster %d", idx)
        else:
            self.logger.info("Output file for cluster %d already exists, skipping", idx)

    def _transform_embeddings(self):
        self.logger.info("Transform embeddings using PCA components")
        for idx in tqdm(
            range(self.num_clusters), desc="Transforming embedding clusters"
        ):
            self._transform_embedding(idx)

    def _transform_urls(self):
        self.logger.info("Transform url clusters using PCA components")
        output_file = (
            f"{self.dim_red_path}/pca_{self.dim_red_dimension}/"
            f"pca_{self.dim_red_dimension}/"
        )
        if not os.path.exists(output_file):
            drm.transform_embeddings(
                self.pca_components,
                f"{self.config.clustering_path}/url_clusters.txt",
                output_file,
                self.logger,
            )
        else:
            self.logger.info("Output file for url clusters already exists, skipping")

    def reduce_dimensions(self):
        """Reduce the dimensions of the embeddings using PCA."""
        self.logger.info("Train PCA compenents for dim reduction")
        pca_components_path = f"{self.dim_red_path}/pca_{self.dim_red_dimension}.npy"
        drm.train_pca(
            pca_components_path,
            self.embeddings,
            self.dim_red_dimension,
            self.logger,
        )
        self.pca_components = np.load(pca_components_path)

        # Transform the embeddings using the PCA components
        self._transform_embeddings()

        # Transform the url clusters using the PCA components
        self._transform_urls()
