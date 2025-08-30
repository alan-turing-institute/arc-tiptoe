"""
Clustering class for document embeddings.
"""

import logging
import os

import faiss
import numpy as np
from tqdm import tqdm

from arc_tiptoe.preprocessing.clustering.cluster_methods import kmeans_clustering
from arc_tiptoe.preprocessing.utils.config import PreProcessConfig

# How many clusters to assign embeddings to if they're close to multiple centroids
MULTI_ASSIGN = 2


class Cluter:
    """Clutering class, clusters and assigns."""

    def __init__(self, config: PreProcessConfig):
        self.config = config
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    f"{self.config.data['dataset']}_"
                    f"{self.config.cluster['clustering_method']}.log"
                ),
            ],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Starting clustering process, checking for precomputed")

        if config.clustering_done:
            self.logger.info("Clustering already complete")
        else:
            self.logger.info("Clustering not yet complete, starting clustering")
            self._gen_directory_structure()
            self.embeddings = np.load(f"{self.config.embeddings_path}/embeddings.npy")
            self.urls = np.load(f"{self.config.embeddings_path}/docids.npy")
            self.num_clusters = np.ceil(np.sqrt(len(self.embeddings))).astype(int)

    def _gen_directory_structure(self):
        """
        Generate the directory structure for the clustering."""
        os.makedirs(
            os.path.join("data", self.config.uuid, "clustering", "centroids"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join("data", self.config.uuid, "clustering", "assignments"),
            exist_ok=True,
        )
        self.config.clustering_path = os.path.join(
            "data", self.config.uuid, "clustering"
        )

    def _compute_centroids(self):
        """Compute centroids for the embeddings."""
        if self.config.cluster["clustering_method"] == "kmeans":
            centroids = kmeans_clustering(self.embeddings, self.num_clusters)
        else:
            err = "Clustering method not recognized."
            raise ValueError(err)

        np.savetxt(f"{self.config.clustering_path}/centroids/centroids.npy", centroids)

    def _assign_embeddings(self):
        """Assign embeddings to clusters."""
        centroids_path = f"{self.config.clustering_path}/centroids/centroids.npy"

        centroids = np.load(centroids_path)
        self.logger.info("Loaded centroids from %s", centroids_path)
        self.logger.info("Centroids shape: %s", centroids.shape)

        cluster_files = [
            (f"{self.config.clustering_path}/assignments/cluster_{i}.txt")
            for i in range(self.num_clusters)
        ]
        assignment_dict = {}
        index = faiss.IndexFlatL2(self.embeddings.shape[1])
        if centroids.ndim == 1:
            centroids = centroids.reshape(1, -1)
        index.add(centroids.astype(np.float32))
        distances, assignments = index.search(
            self.embeddings.astype(np.float32), MULTI_ASSIGN
        )

        percentilers = []
        for i in range(1, MULTI_ASSIGN):
            percentilers.append(
                np.percentile([(dist[i] - dist[0]) for dist in distances], 20)
            )

        for idx, _ in enumerate(assignments):
            for k in range(MULTI_ASSIGN):
                if (k == 0) or (
                    k > 0
                    and (distances[idx][k] - distances[idx][0]) < percentilers[k - 1]
                ):
                    cluster = assignments[idx][k]
                    if cluster not in assignment_dict:
                        assignment_dict[cluster] = [idx]
                    else:
                        assignment_dict[cluster].append(idx)

        for i in tqdm(
            range(self.num_clusters), desc="Writing clusters", unit="cluster"
        ):
            with open(cluster_files[i], "w", encoding="utf-8") as f:
                if i in assignment_dict:
                    for idx in assignment_dict[i]:
                        embed = self.embeddings[idx]
                        url = self.urls[idx]
                        embstr = ",".join([f"{ch}" for ch in embed])
                        doc_id = idx
                        data_str = f"{doc_id} | {embstr} | {url}\n"
                        f.write(data_str)
                else:
                    self.logger.info("No assignments for cluster %d", i)

        self.logger.info("Finished assignment of embeddings to clusters")

    def _parse_file(self, cluster_file):
        """Parse a cluster file and return its contents."""
        contents = []
        with open(cluster_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in tqdm(lines, desc=f"Parsing {cluster_file}"):
                if len(line) <= 1:
                    continue
                contents.append(line.split(" | "))
        return contents

    def _create_bundles(self, contents):
        """Create bundles of URLs based on clustering."""
        self.logger.info("Creating bundles")
        num_bundles = int(
            np.ceil(
                float(len(contents)) / float(self.config.cluster["avg_bundle_size"])
            )
        )
        self.logger.info("Num_bundles = %d", num_bundles)

        embed_contents = [elem[1] for elem in contents]
        data = np.loadtxt(embed_contents, delimiter=",")
        kmeans = faiss.Kmeans(data.shape[1], num_bundles, nredo=3)
        if len(data) >= 1 and len(np.shape(data)) == 2:
            kmeans.train(data.astype(np.float32))
            centroids = kmeans.centroids
            _, assignments = kmeans.index.search(data.astype(np.float32), 1)
        else:
            centroids = np.zeros((1, data.shape[1]))
            assignments = [[0]]

        self.logger.info("Creating assignment dictionary")
        assignment_dict = {}
        for i, cluster_pair in tqdm(
            enumerate(assignments), desc="Creating assignment dictionary"
        ):
            cluster = cluster_pair[0]
            if cluster not in assignment_dict:
                assignment_dict[cluster] = [i]
            else:
                assignment_dict[cluster].append(i)

        return centroids, assignment_dict

    def _process_cluster(self, cluster_file, cluster_idx):
        """Process a single cluster file to create bundles and centroids."""
        self.logger.info("**** PROCESS CLUSTER *****")
        contents = self._parse_file(cluster_file)
        self.logger.info("LEN = %d", len(contents))
        if len(contents) == 0:
            return

        centroids, assignment_dict = self._create_bundles(contents)

        self.logger.info(f"Writing to file -- {cluster_file}/{cluster_idx}")

    def _process_urls(self):
        """Process URLs associated with embeddings."""
        cluster_files = [
            (f"{self.config.clustering_path}/assignments/cluster_{i}.txt")
            for i in range(self.num_clusters)
        ]

        for idx, cluster in enumerate(cluster_files):
            self._process_cluster(cluster, idx)

    def cluster_and_assign(self):
        """Cluster and assign embeddings and urls."""
        self.logger.info(
            "Clustering method: %s", self.config.cluster["clustering_method"]
        )
        self.logger.info("Number of clusters: %d", self.num_clusters)
        self._compute_centroids()

        self.logger.info("Initalised clustering, starting assignment")
        self._assign_embeddings()

        self.logger.info("Clustering URLs")
        slef._process_urls()
