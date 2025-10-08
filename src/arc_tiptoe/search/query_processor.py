"""
Query processing for search protocol - integrates with preprocessing pipeline
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from arc_tiptoe.preprocessing.utils.config import PreProcessConfig


class QueryProcessor:
    """
    Handles query embedding and cluster assignment for the search protocol.
    Reads configuration from preprocessing pipeline to ensure consistency.
    """

    def __init__(self, config_path: str, preamble: str | None = None):
        """
        Initialize query processor with preprocessing config.

        Args:
            config_path: Path to preprocessing config JSON or search config JSON
            preamble: Optional preamble path override
        """
        self.config_path = config_path
        self.preamble = preamble

        # Load configuration
        if self._is_search_config(config_path):
            self.config = self._load_search_config(config_path)
        else:
            self.config = self._load_preprocessing_config(config_path)

        # Set up paths
        self._setup_paths()

        # Load components
        self._load_model()
        self._load_pca_components()
        self._load_cluster_index()

        print("QueryProcessor initialized:", file=sys.stderr)
        print(f"Model: {self.model_name}", file=sys.stderr)
        print(f"Dimensions: {self.original_dim} -> {self.reduced_dim}", file=sys.stderr)
        print(f"Clusters: {self.num_clusters}", file=sys.stderr)

    def _is_search_config(self, config_path: str) -> bool:
        """Check if this is a search config or preprocessing config"""
        with open(config_path, "r") as f:
            config = json.load(f)
        return "embedding" in config and "clustering" in config

    def _load_search_config(self, config_path: str) -> dict:
        """Load search configuration"""
        with open(config_path, "r") as f:
            config = json.load(f)

        return {
            "uuid": config["uuid"],
            "data_path": config["data_path"],
            "model_name": config["embedding"]["model_name"],
            "original_dim": config["embedding"]["embedding_dim"],
            "reduced_dim": config["embedding"]["reduced_dimension"],
            "num_clusters": config["clustering"]["total_clusters"],
            "search_top_k": config["clustering"]["search_top_k"],
            "centroids_file": config["clustering"]["centroids_file"],
            "cluster_directory": config["clustering"]["cluster_dir"],
            "pca_applied": config["dim_reduction"]["applied"],
            "pca_components_file": config["dim_reduction"]["pca_components_file"],
            "faiss_index": config["artifacts"]["faiss_index"],
            "artifact_directory": config["artifacts"]["artifact_directory"],
        }

    def _load_preprocessing_config(self, config_path: str) -> dict:
        """Load preprocessing configuration and extract relevant info"""
        preprocess_config = PreProcessConfig(config_path)

        return {
            "uuid": preprocess_config.uuid,
            "data_path": f"data/{preprocess_config.uuid}",
            "model_name": preprocess_config.embed_model,
            "original_dim": 768,
            "reduced_dim": preprocess_config.dim_red.get("dim_red_dimension", 192),
            "num_clusters": preprocess_config.cluster.get("num_clusters", 1280),
            "search_top_k": 1,  # Default
            "centroids_file": f"data/{preprocess_config.uuid}/clusters/centroids.txt",
            "cluster_directory": f"data/{preprocess_config.uuid}/clusters",
            "pca_applied": preprocess_config.dim_red["apply_dim_red"],
            "pca_components_file": (
                f"data/{preprocess_config.uuid}/dim_red/pca_components_"
                f"{preprocess_config.dim_red.get('dim_red_dimension', 192)}.npy"
            ),
            "faiss_index": (
                f"data/{preprocess_config.uuid}/artifact/"
                f"dim{preprocess_config.dim_red.get('dim_red_dimension', 192)}/"
                f"index.faiss"
            ),
            "artifact_directory": (
                f"data/{preprocess_config.uuid}/artifact/"
                f"dim{preprocess_config.dim_red.get('dim_red_dimension', 192)}"
            ),
        }

    def _setup_paths(self):
        """Setup file paths based on configuration"""
        # Extract configuration values
        self.model_name = self.config["model_name"]
        self.original_dim = self.config["original_dim"]
        self.reduced_dim = self.config["reduced_dim"]
        self.num_clusters = self.config["num_clusters"]
        self.search_top_k = self.config["search_top_k"]

        if self.preamble:
            # Override with provided preamble
            base_path = Path(self.preamble) / self.config["data_path"]
        else:
            # Use config paths directly
            data_path = Path(self.config["data_path"])

            current_dir = Path.cwd()
            if current_dir.name == "search" and not Path(data_path).is_absolute():
                base_path = Path("..") / data_path
            else:
                base_path = Path(data_path)

        # If using preamble, adjust paths
        if self.preamble:
            self.centroids_file = str(
                base_path / "clusters" / "centroids" / "final_centroids.txt"
            )
            self.pca_components_file = str(
                base_path / "dim_red" / f"pca_components_{self.reduced_dim}.npy"
            )
            self.faiss_index_file = str(
                base_path / "artifact" / f"dim{self.reduced_dim}" / "index.faiss"
            )
        else:
            current_dir = Path.cwd()
            if current_dir.name == "search":
                self.centroids_file = str(Path("..") / self.config["centroids_file"])
                self.pca_components_file = str(
                    Path("..") / self.config["pca_components_file"]
                )
                self.faiss_index_file = str(Path("..") / self.config["faiss_index"])
            else:
                # Use paths as-is
                self.centroids_file = self.config["centroids_file"]
                self.pca_components_file = self.config["pca_components_file"]
                self.faiss_index_file = self.config["faiss_index"]

    def _load_model(self):
        """Load the embedding model"""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = "1"
        # TODO refactor for GPU
        self.model = SentenceTransformer(self.model_name, device="cpu")

    def _load_pca_components(self):
        """Load PCA components for dimensionality reduction"""
        if not self.config["pca_applied"]:
            self.pca_components = None
            print("No PCA applied", file=sys.stderr)
            return 1

        self.pca_components = np.load(self.pca_components_file)
        return 1

    def _load_cluster_index(self):
        """Load FAISS index for cluster assignment"""
        if os.path.exists(self.faiss_index_file):
            self.cluster_index = faiss.read_index(self.faiss_index_file)
            print(f"Loaded FAISS index: {self.faiss_index_file}", file=sys.stderr)
        else:
            # Fallback to creating index from centroids file
            print("FAISS index not found, creating from centroids", file=sys.stderr)
            centroids = np.loadtxt(self.centroids_file)
            self.cluster_index = faiss.IndexFlatIP(centroids.shape[1])
            self.cluster_index.add(centroids.astype("float32"))
            print(
                f"Created FAISS index from centroids: {centroids.shape}",
                file=sys.stderr,
            )

    def process_query(self, query: str, top_k_clusters: int | None = None) -> dict:
        """
        Process a single query and return cluster assignment and quantized embedding.

        Args:
            query: The query string to process
            top_k_clusters: Number of top clusters to return (default from config)

        Returns:
            Dictionary with cluster_index and quantized embedding
        """
        if top_k_clusters is None:
            top_k_clusters = self.search_top_k

        # Generate embedding using same model as preprocessing
        embedding = self.model.encode_query([query], convert_to_numpy=True)[0]

        # Apply dim reduction if configured
        if self.pca_components is not None:
            embedding_reduced = np.matmul(embedding, self.pca_components)
        else:
            embedding_reduced = embedding

        # Quantize embedding
        data_min = np.min(embedding_reduced)
        data_max = np.max(embedding_reduced)
        data_range = max(abs(data_min), abs(data_max))
        scale = 127.0 / data_range
        embedding_quantized = np.clip(
            np.round(embedding_reduced * scale), -127, 127
        ).astype(np.int8)

        # find nearest clusters
        cluster_indices = self._find_nearest_clusters(embedding_reduced, top_k_clusters)

        result = {
            "ClusterIndex": int(cluster_indices[0]) if cluster_indices else 0,
            "Emb": embedding_quantized.tolist(),
            "TopKClusterIndices": cluster_indices,
        }
        # print(result)
        return result

    def _find_nearest_clusters(self, embedding: np.ndarray, top_k: int) -> List[int]:
        """Find the top-k nearest clusters fro a given embedding"""
        if self.cluster_index is None:
            return [0]

        query_embedding = embedding.reshape(1, -1).astype("float32")

        # search for clusters
        distances, indices = self.cluster_index.search(
            query_embedding, min(top_k, self.num_clusters)
        )

        # return valid_clusters
        valid_clusters = []
        for i in range(len(indices[0])):
            cluster_id = indices[0][i]
            if 0 <= cluster_id < self.num_clusters:
                valid_clusters.append(int(cluster_id))

        return valid_clusters if valid_clusters else [0]


def main():
    """Main function for command-line usage from Go"""
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} config_path num_clusters [top_k_clusters]")
        sys.exit(1)

    config_path = sys.argv[1]
    num_clusters = int(sys.argv[2])
    top_k_clusters = int(sys.argv[3]) if len(sys.argv) > 3 else None

    # Extract preamble from path if provided
    preamble = None
    if len(sys.argv) >= 5:
        preamble = sys.argv[4]

    try:
        processor = QueryProcessor(config_path, preamble)

        # print("Processor loaded", file=sys.stderr)

        for line in sys.stdin:
            query = line.strip()
            if not query:
                continue

            result = processor.process_query(query, top_k_clusters)
            print(json.dumps(result))
            sys.stdout.flush()

    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
