"""
Class for the full preprocessing pipeline.
"""

import logging

from arc_tiptoe.preprocessing.clustering import clusterers
from arc_tiptoe.preprocessing.dim_reduce import dim_reducers
from arc_tiptoe.preprocessing.embedding import embedders
from arc_tiptoe.preprocessing.utils.config import PreProcessConfig


class PreprocessingPipeline:
    """Class to manage the full preprocessing pipeline, including embedding,
    dimensionality reduction, and clustering."""

    def __init__(self, config: PreProcessConfig):
        self.config = config
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{self.config.uuid}_preprocessing.log"),
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
        ](self.config, within_pipeline=True)
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

    def run(self):
        """Run full preprocessing pipeline."""
        self._embed()

        if self.config.dim_red["dim_red_before_clustering"]:
            self.logger.info("Running dimensionality reduction before clustering")
            self._dim_reduce()
            self._cluster()
        else:
            self.logger.info("Running clustering before dimensionality reduction")
            self._cluster()
            self._dim_reduce()

        self.logger.info("%s", "==" * 20)
        self.logger.info("Preprocessing pipeline complete")
