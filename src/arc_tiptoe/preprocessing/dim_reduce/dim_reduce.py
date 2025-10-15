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
from copy import copy
from pathlib import Path

import numpy as np
from tqdm import tqdm

import arc_tiptoe.preprocessing.dim_reduce.dim_reduce_methods as drm
import arc_tiptoe.preprocessing.utils.utils as utils

# from glob import glob


class DimReducer(ABC):
    """Base class for all dimensionality reduction methods."""

    def __init__(self, config, within_pipeline: bool = False):
        self.config = config
        self.within_pipeline = within_pipeline
        # replace datasetname slashes with underscores for file paths
        dataset_name_safe = copy(self.config.dataset_name).replace("/", "_")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    f"data/{self.config.uuid}/"
                    f"{dataset_name_safe}_"
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
            self._gen_directory_structure()

    def _gen_directory_structure(self):
        """Generate the directory structure"""
        self.save_path = Path(f"{self.dim_red_path}/").joinpath(
            f"{self.config.dim_red['dim_red_method']}_" f"{self.dim_red_dimension}"
        )
        os.makedirs(self.save_path, exist_ok=True)

    @abstractmethod
    def _transform_embeddings(self):
        """Transform a single embedding"""
        raise NotImplementedError()

    @abstractmethod
    def _transform_clustered_embedding(self, cluster_file):
        """Transform a single clustered embedding"""
        raise NotImplementedError()

    @abstractmethod
    def _transform_centroids(self):
        """Transform the centroids"""
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
        if not self.config.dim_red_done:
            self.logger.info("Initialized PCA dimensionality reduction")
            self.pca_components = None
            self.pca_components_path = (
                f"{self.dim_red_path}/"
                f"{self.config.dim_red['dim_red_method']}_"
                f"{self.dim_red_dimension}.npy"
            )
        else:
            self._return_config()

    def _return_config(self):
        """Return the config if within pipeline"""
        if self.within_pipeline:
            return self.config

        return 1

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
        output_file = f"{self.save_path}/embeddings.npy"
        drm.transform_embeddings(self.pca_components, self.embeddings, output_file)

    def _transform_clustered_embedding(self, cluster_file):
        """Transform a single embedding using PCA components."""
        cluster_contents = utils.parse_file(cluster_file)
        if len(cluster_contents) == 0:
            return
        cluster_embeddings = np.array(
            [np.fromstring(content[1], sep=",") for content in cluster_contents]
        )
        reduced_embeddings = drm.run_pca(self.pca_components, cluster_embeddings)
        print(
            "check not zero here 4",
            np.sum(cluster_embeddings),
            np.sum(reduced_embeddings),
        )
        # Save the reduced embeddings back to the cluster file
        with open(
            f"{self.save_path}/clusters/{cluster_file.name}",
            "w",
            encoding="utf-8",
        ) as f:
            for i, content in enumerate(cluster_contents):
                embedding_str = ",".join(map(str, reduced_embeddings[i]))
                f.write(f"{content[0]} | {embedding_str} | {content[2]}\n")

    def _transform_centroids(self):
        """Transform the centroids using PCA components."""
        centroids = np.load(
            f"{self.config.clustering_path}/processing/centroids/final_centroids.npy"
        )
        print("check not zero here 3", np.sum(centroids))
        reduced_centroids = drm.run_pca(self.pca_components, centroids)
        np.savetxt(f"{self.save_path}/centroids.txt", reduced_centroids)

    def _reduce_dimensions(self):
        """Reduce the dimensions of the embeddings using PCA."""
        self.logger.info("Train PCA components for dim reduction")
        # Transform the embeddings using the PCA components
        if self.pca_components is None:
            if os.path.exists(self.pca_components_path):
                self.pca_components = np.load(self.pca_components_path)
                self.logger.info(
                    "Loaded existing PCA components from %s", self.pca_components_path
                )
            else:
                self._train_pca()
        self._transform_embeddings()
        if self.config.cluster["apply_clustering"]:
            if self.config.dim_red["dim_red_before_clustering"]:
                self.logger.info("Dimensionality reduction done before clustering")
                return 1
            self.logger.info("Transform clustered embeddings using PCA components")
            self.logger.info("Processing clustered embeddings")
            os.makedirs(f"{self.save_path}/clusters", exist_ok=True)
            cluster_files = Path(
                f"{self.config.clustering_path}/processing/processed_clusters/"
            ).glob("cluster_*.txt")
            for cluster_file in tqdm(
                cluster_files,
                desc="Processing clustered embeddings",
                unit="cluster",
            ):
                self._transform_clustered_embedding(cluster_file)
            self._transform_centroids()
        self.logger.info("Dimensionality reduction completed")

        return 1
