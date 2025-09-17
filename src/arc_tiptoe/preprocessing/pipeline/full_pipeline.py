"""
Class for the full preprocessing pipeline.
"""

import json
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

    def _organise_data(self):
        """
        Organise the preprocessed data into a standard structure required for search.
        """
        self.logger.info("Organising preprocessed data %s", "==" * 20)
        # Copy assigned clusters to clustering directory
        if (
            self.config.cluster["apply_clustering"]
            and self.config.dim_red["apply_dim_red"]
        ):
            # Move the reduced clustered assignments and centroids to the
            # clustering dire"ctory
            src_path = Path(
                f"{self.config.dim_red_path}/"
                f"{self.config.dim_red['dim_red_method']}_"
                f"{self.config.dim_red['dim_red_dimension']}/clusters"
            )
            for each_file in src_path.glob("*.*"):
                trg_path = Path(f"{self.config.clustering_path}")
                each_file.rename(trg_path.joinpath(each_file.name))

            src_path = Path(
                f"{self.config.dim_red_path}/"
                f"{self.config.dim_red['dim_red_method']}_"
                f"{self.config.dim_red['dim_red_dimension']}/centroids.txt"
            )
            trg_path = Path(f"{self.config.clustering_path}/centroids.txt")
            src_path.rename(trg_path)

            # Move the embeddings, first renaming original embeddings
            # to avoid overwriting
            trg_path = Path(f"{self.config.embeddings_path}/embeddings.npy")
            trg_path.rename(
                Path(f"{self.config.embeddings_path}/embeddings_original.npy")
            )

            src_path = Path(
                f"{self.config.dim_red_path}/"
                f"{self.config.dim_red['dim_red_method']}_"
                f"{self.config.dim_red['dim_red_dimension']}/embeddings.npy"
            )
            trg_path = Path(f"{self.config.embeddings_path}/embeddings.npy")
            src_path.rename(trg_path)
        elif (
            self.config.cluster["apply_clustering"]
            and not self.config.dim_red["apply_dim_red"]
        ):
            # Move the original clustered assignments and centroids to the
            # clustering dire"ctory
            src_path = Path(
                f"{self.config.cluster_path}/clusters/processing/processed_clusters"
            )
            for each_file in src_path.glob("*.*"):
                trg_path = Path(f"{self.config.clustering_path}/clusters")
                each_file.rename(trg_path.joinpath(each_file.name))

            src_path = Path(
                f"{self.config.cluster_path}/processing/centroids/final_centroids.txt"
            )
            trg_path = Path(f"{self.config.clustering_path}/centroids.txt")
            src_path.rename(trg_path)
        elif (
            self.config.dim_red["apply_dim_red"]
            and not self.config.cluster["apply_clustering"]
        ):
            # TODO: implement for dim red without clustering
            pass

    def _generate_search_config(self):
        """Generate a config file for use in search."""
        self.logger.info("Generating search config file")
        search_config = {
            "uuid": self.config.uuid,
            "data_path": f"data/{self.config.uuid}",
            "embedding": {
                "model_name": self.config.embed_model,
                "embedding_dim": self.config.embed_pars.get("embedding_dimension", 768),
                "reduced_dimension": self.config.dim_red.get("dim_red_dimension", 192),
            },
            "clustering": {
                "total_clusters": self.config.cluster.get("num_clusters"),
                "search_top_k": 1,  # 1 by default
                "centroids_file": f"data/{self.config.uuid}/clusters/centroids.txt",
                "cluster_dir": f"data/{self.config.uuid}/clusters",
            },
            "dim_reduction": {
                "applied": self.config.dim_red.get("apply_dim_red"),
                "method": self.config.dim_red.get("dim_red_method"),
                "pca_components_file": (
                    f"{self.config.dim_red_path}/"
                    f"{self.config.dim_red.get('dim_red_method')}_"
                    f"{self.config.dim_red.get('dim_red_dimension')}.npy"
                ),
            },
            "server_config": self._calculate_server_config(),
            "artifacts": {
                "faiss_index": (
                    f"data/{self.config.uuid}/artifacts/"
                    f"dim{self.config.dim_red.get('dim_red_dimension')}/"
                    f"index.faiss"
                ),
                "artifact_directory": (
                    f"data/{self.config.uuid}/artifacts/"
                    f"dim{self.config.dim_red.get('dim_red_dimension')}"
                ),
            },
        }

        # save search config
        search_config_path = Path(self.config.orig_config_path).with_suffix("")
        search_config_path = f"{search_config_path}_search_config.json"
        with open(search_config_path, "w") as f:
            json.dump(search_config, f, indent=4)

    def _calculate_server_config(self):
        """Calculate optimal server configuration based on cluster count"""
        total_clusters = self.config.cluster.get("num_clusters", 1280)

        # Calculate embedding servers (1 per 16 clusters, max 80)
        embedding_servers = min(max(1, (total_clusters + 15) // 16), 80)
        clusters_per_embedding_server = (
            total_clusters + embedding_servers - 1
        ) // embedding_servers

        # Calculate URL servers (1 per 160 clusters, max 8)
        url_servers = min(max(1, (total_clusters + 159) // 160), 8)
        clusters_per_url_server = (total_clusters + url_servers - 1) // url_servers

        return {
            "embedding_servers": embedding_servers,
            "url_servers": url_servers,
            "clusters_per_embedding_server": clusters_per_embedding_server,
            "clusters_per_url_server": clusters_per_url_server,
            "embedding_hint_size": 500,
            "url_hint_size": 100,
        }

    def run(self):
        """
        Run full preprocessing pipeline.

        The pipeline will always extract the embeddings first, and has the following
        logic for the other steps:

            - If dimensionality reduction and clustering are both enabled, and
            dimensionality reduction is set to be applied before clustering, then
            the embeddings will be reduced before clustering.
            - Otherwise the clustering will be applied first, and if dimensionality
            reduction is enabled it will be applied after clustering.
                - This is done by letting the clustering write to file first and then
                this is picked up by the dimensionality reduction step.
            - If only clustering is enabled, then only clustering will be applied.
            - If only dimensionality reduction is enabled, then only dimensionality
            reduction will be applied.
            - If neither is enabled, then nothing further will be done.

        Finally the data will be organised into a standard structure.
        """
        self._embed()

        if (
            self.config.cluster["apply_clustering"]
            and self.config.dim_red["apply_dim_red"]
        ):
            if self.config.dim_red["dim_red_before_clustering"]:
                self.logger.info(
                    "Dimensionality reduction set to be applied before clustering"
                )
                self._dim_reduce()
                self._cluster()
            else:
                self.logger.info(
                    "Clustering set to be applied before dimensionality reduction"
                )
                self._cluster()
                self._dim_reduce()
        elif self.config.cluster["apply_clustering"]:
            self.logger.info("Only clustering to be applied")
            self._cluster()
        elif self.config.dim_red["apply_dim_red"]:
            self.logger.info("Only dimensionality reduction to be applied")
            self._dim_reduce()
        else:
            self.logger.info("No clustering or dimensionality reduction to apply")

        self._generate_search_config()
        self._organise_data()

        self.logger.info("%s", "==" * 20)
        self.logger.info("Preprocessing pipeline complete")
