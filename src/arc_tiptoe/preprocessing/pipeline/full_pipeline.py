"""
Class for the full preprocessing pipeline.
"""

import glob
import logging
from pathlib import Path

from arc_tiptoe.preprocessing.clustering import clusterers
from arc_tiptoe.preprocessing.dim_reduce import dim_reducers
from arc_tiptoe.preprocessing.embedding import embedders
from arc_tiptoe.preprocessing.utils.config import PreProcessConfig


class PreprocessingPipeline:
    """
    Class to manage the full preprocessing pipeline, including embedding,
    dimensionality reduction, and clustering.

    NB: Dimensionality reduction is run within clustering.
    """

    def __init__(self, config: PreProcessConfig):
        self.config = config
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    f"data/{self.config.uuid}/{self.config.uuid}_preprocessing.log"
                ),
            ],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing preprocessing pipeline")

        # load these when appropriate
        self.embedder = None
        self.dim_reducer = None
        self.clusterer = None

    def _embed(self):
        """Run embedding step."""
        if self.config.embedding_done:
            self.logger.info("Skipping embedding step, already done")
            return

        self.embedder = embedders.embedders[self.config.embed_lib](
            self.config, within_pipeline=True
        )
        self.logger.info("Starting embedding step %s", "==" * 20)
        self.embedder.load_dataset()
        self.config = self.embedder.embed()

    def _cluster(self):
        """Run clustering step."""
        if self.config.clustering_done:
            self.logger.info("Skipping clustering step, already done")
            return

        self.clusterer = clusterers.clusterers[
            self.config.cluster["clustering_method"]
        ](self.config, within_pipeline=True, dim_reducer=self.dim_reducer)
        self.logger.info("Starting clustering step %s", "==" * 20)
        self.config = self.clusterer.cluster_and_assign()

    def _dim_reduce(self):
        """Run dimensionality reduction step."""
        if self.config.dim_red_done:
            self.logger.info("Skipping dimensionality reduction step, already done")
            return

        self.dim_reducer = dim_reducers.dim_reducers[
            self.config.dim_red["dim_red_method"]
        ](self.config, within_pipeline=True)
        self.logger.info("Starting dimensionality reduction step %s", "==" * 20)
        self.config = self.dim_reducer.reduce_dimensions()

    def _organise_data(self):
        """
        Organise the preprocessed data into a standard structure required for search.
        """
        self.logger.info("Organising preprocessed data %s", "==" * 20)
        # Copy assigned clusters to clustering directory
        if self.config.cluster["apply_clustering"]:
            src_path = Path(f"{self.config.clustering_path}").joinpath("assignments")

            for each_file in src_path.glob("*.*"):
                trg_path = src_path.parent
                each_file.rename(trg_path.joinpath(each_file.name))

    def run(self):
        """
        Run full preprocessing pipeline.

        The pipeline will always extract the embeddings first, and has the following
        logic for the other steps:
            - If clustering is enabled in the config:
                - If dimensionality reduction is also enabled, it will be applied
                within clustering, with the option of dim reduction before or after
                clustering taken into account here.
                - If dimensionality reduction is not enabled, only clustering will be
                run.
            - If clustering is not enabled in the config, but dimensionality reduction
            is enabled, only dimensionality reduction will be run.
            - If neither clustering nor dimensionality reduction are enabled, the
            pipeline will log that there is nothing to do and exit.
        """
        self._embed()

        if self.config.cluster["apply_clustering"]:
            self.logger.info("Clustering is enabled in config")
            if self.config.dim_red["apply_dim_red"]:
                self.logger.info(
                    "Dimensionality reduction is enabled in config,"
                    "applying within clustering"
                )
                self.dim_reducer = dim_reducers.dim_reducers[
                    self.config.dim_red["dim_red_method"]
                ](self.config)
                self._cluster()
            else:
                self.logger.info("Dimensionality reduction is not enabled in config")
                self._cluster()
        elif self.config.dim_red["apply_dim_red"]:
            self.logger.info(
                "Clustering is not enabled in config, but"
                "dimensionality reduction is enabled"
            )
            self._dim_reduce()
        else:
            self.logger.info(
                "Neither clustering nor dimensionality reduction"
                "are enabled in config, nothing to do"
            )

        self._organise_data()

        self.logger.info("%s", "==" * 20)
        self.logger.info("Preprocessing pipeline complete")
