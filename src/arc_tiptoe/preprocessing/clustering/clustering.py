"""
Clustering class for document embeddings.
"""

import logging
import os
import zlib
from abc import ABC, abstractmethod

import faiss
import numpy as np
from tqdm import tqdm

import arc_tiptoe.preprocessing.clustering.cluster_methods as cm
from arc_tiptoe.preprocessing.utils.config import PreProcessConfig

# How many clusters to assign embeddings to if they're close to multiple centroids
MULTI_ASSIGN = 2


class Clusterer(ABC):
    """Clutering class, clusters and assigns."""

    def __init__(self, config: PreProcessConfig, within_pipeline: bool = False):
        self.config = config
        self.within_pipeline = within_pipeline
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
            self.avg_bundle_size = self.config.cluster["avg_bundle_size"]
            self.urls_per_bundle = self.config.cluster["urls_per_bundle"]
            self.max_size = self.config.cluster["max_size"]

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

    @abstractmethod
    def _compute_centroids(self):
        """Compute teh centroids for the embeddings. Rewrite in child class for given
        clustering method."""
        raise NotImplementedError()

    def _assign_embedding(
        self, idx, distances, assignments, percentilers, assignment_dict
    ):
        """Assign a single embedding to a cluster"""
        for k in range(MULTI_ASSIGN):
            if (k == 0) or (
                k > 0 and (distances[idx][k] - distances[idx][0]) < percentilers[k - 1]
            ):
                cluster = assignments[idx][k]
                if cluster not in assignment_dict:
                    assignment_dict[cluster] = [idx]
                else:
                    assignment_dict[cluster].append(idx)

    def _write_assignment_to_cluster(self, cluster_files, i, assignment_dict):
        """Write single assignment to cluster file"""
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
            self._assign_embedding(
                idx, distances, assignments, percentilers, assignment_dict
            )

        for i in tqdm(
            range(self.num_clusters), desc="Writing clusters", unit="cluster"
        ):
            self._write_assignment_to_cluster(cluster_files, i, assignment_dict)

        self.logger.info("Finished assignment of embeddings to clusters")

    def _parse_file(self, cluster_file):
        """Parse a cluster file and return its contents."""
        contents = []
        with open(cluster_file, encoding="utf-8") as f:
            lines = f.readlines()
            for line in tqdm(lines, desc=f"Parsing {cluster_file}"):
                if len(line) <= 1:
                    continue
                contents.append(line.split(" | "))
        return contents

    def _get_size(self, contents):
        """Calculate the size of the contents."""
        out = zlib.compress(
            bytes(" ".join([content[2] for content in contents]), "utf-8"), level=9
        )
        return len(out)

    @abstractmethod
    def _embed_contents(self, contents, num_bundles):
        """Embed the contents using FAISS. Rewrite in child class for given clustering
        method."""
        raise NotImplementedError()

    def _create_assignment_dict(self, cluster_pair, contents, assignment_dict, idx):
        """Create an assignment dictionary for a given cluster pair."""
        cluster = cluster_pair[0]
        if cluster not in assignment_dict:
            assignment_dict[cluster] = [contents[idx]]
        else:
            assignment_dict[cluster].append(contents[idx])
        return assignment_dict

    def _divide_contents_arbitrarily(self, contents, centroids, assignment_dict):
        """Divide contents arbitrarily when all documents are the same."""
        self.logger.info("All documents assigned to one cluster, dividing arbitrarily")
        for i in tqdm(
            range(int(np.ceil(len(contents) / float(self.urls_per_bundle)))),
            desc="Dividing contents arbitrarily",
        ):
            centroids[i] = centroids[next(iter(assignment_dict.keys()))]
            upper_bound = min(i + 1) * self.urls_per_bundle, len(contents)
            assignment_dict[i] = [
                contents[j] for j in range(i * self.urls_per_bundle, upper_bound)
            ]
        return (centroids, assignment_dict)

    def _process_large_url_clusters(self, cluster, assignment_dict, centroids):
        """Process large URL clusters by creating sub-bundles."""
        self.logger.info("Cluster %d exceed max size, creating bundles", cluster)
        (sub_centroids, sub_assignment_dict) = self._create_bundles(
            assignment_dict[cluster]
        )
        replace_idx = sorted(sub_assignment_dict.keys())[0]
        centroids[cluster] = sub_centroids[replace_idx]
        assignment_dict[cluster] = sub_assignment_dict[replace_idx]
        offset = len(centroids)
        centroids = np.vstack((centroids, sub_centroids[replace_idx + 1 :]))
        for sub_cluster in sub_assignment_dict:
            if sub_cluster != replace_idx:
                assignment_dict[sub_cluster + offset] = sub_assignment_dict[sub_cluster]

    def _create_bundles(self, contents):
        """Create bundles of URLs based on clustering."""
        self.logger.info("Creating bundles")
        num_bundles = int(
            np.ceil(float(self._get_size(contents)) / float(self.avg_bundle_size))
        )
        self.logger.info("Num_bundles = %d", num_bundles)

        # Embed contents and perform clustering
        centroids, assignments = self._embed_contents(contents, num_bundles)

        # Create a dictionary to hold the assignments
        self.logger.info("Creating assignment dictionary")
        assignment_dict = {}
        for idx, cluster_pair in tqdm(
            enumerate(assignments), desc="Creating assignment dictionary"
        ):
            assignment_dict = self._create_assignment_dict(
                cluster_pair, contents, assignment_dict, idx
            )

        # Divide arbitrarily when all documents are the same
        if len(assignment_dict) == 1 and num_bundles > 1:
            centroids, assignments = self._divide_contents_arbitrarily(
                contents, centroids, assignment_dict
            )
            return (centroids, assignments)

        # If documents are not same, proceed with further processing
        self.logger.info("Documents assigned to multiple clusters, proceeding")
        init_clusters = list(assignment_dict.keys()).copy()
        for cluster in tqdm(init_clusters, desc="Processing initial clusters"):
            if self._get_size(assignment_dict[cluster]) > self.max_size:
                self._process_large_url_clusters(cluster, assignment_dict, centroids)

        return (centroids, assignment_dict)

    def _write_bundle(self, cluster_idx, bundle, assignment_dict):
        """Write a single bundle to file."""
        with open(
            f"{self.config.clustering_path}/{cluster_idx}/clusters/bundle_{bundle}.txt",
            "w",
            encoding="utf-8",
        ) as f:
            self.logger.info("Writing content to bundle %d", bundle)
            for elem in assignment_dict[bundle]:
                if len(elem) == 3:
                    f.write(f"{elem[0]} | {elem[1]} | {elem[2]}\n")
                else:
                    self.logger.warning(
                        "Skipping malformed element in bundle %d: %s", bundle, elem
                    )
                    self.logger.warning("Element length: %d", len(elem))

    def _process_cluster(self, cluster_file, cluster_idx):
        """Process a single cluster file to create bundles and centroids."""
        self.logger.info("**** PROCESS CLUSTER *****")
        contents = self._parse_file(cluster_file)
        self.logger.info("LEN = %d", len(contents))
        if len(contents) == 0:
            return

        centroids, assignment_dict = self._create_bundles(contents)

        self.logger.info("Writing to file -- %s/%s", cluster_file, cluster_idx)
        if not os.path.exists(f"{self.config.clustering_path}/{cluster_idx}/"):
            os.makedirs(f"{self.config.clustering_path}/{cluster_idx}/")
        if not os.path.exists(f"{self.config.clustering_path}/{cluster_idx}/clusters/"):
            os.makedirs(f"{self.config.clustering_path}/{cluster_idx}/clusters/")
        np.savetxt(
            f"{self.config.clustering_path}/{cluster_idx}/centroids.npy",
            centroids,
        )

        for bundle in tqdm(assignment_dict, desc="Writing bundles"):
            self._write_bundle(cluster_idx, bundle, assignment_dict)

        self.logger.info("Finished writeing bundles for cluster %d", cluster_idx)

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
        self._process_urls()

        self.config.clustering_done = True
        self.logger.info("Clustering process complete")
        self.config.save_config()
        if self.within_pipeline:
            return self.config

        return 1


class KMeansCluster(Clusterer):
    """KMeans clustering class."""

    def __init__(self, config: PreProcessConfig):
        super().__init__(config)
        self.logger.info("Initialized KMeans clustering")

    def _compute_centroids(self):
        """Compute centroids for the embeddings."""
        centroids = cm.kmeans_centroids(self.embeddings, self.num_clusters)
        np.savetxt(f"{self.config.clustering_path}/centroids/centroids.npy", centroids)

    def _embed_contents(self, contents, num_bundles):
        """Embed the contents using FAISS."""
        return cm.kmeans_embed_contents(contents, num_bundles, self.logger)
