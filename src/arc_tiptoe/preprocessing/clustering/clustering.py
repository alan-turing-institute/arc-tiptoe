"""
Clustering class for document embeddings.
"""

import logging
import os
from abc import ABC, abstractmethod

import faiss
import numpy as np
from tqdm import tqdm

import arc_tiptoe.preprocessing.clustering.cluster_methods as cm
from arc_tiptoe.preprocessing.utils import utils
from arc_tiptoe.preprocessing.utils.config import PreProcessConfig


class Clusterer(ABC):
    """
    Clutering class, clusters and assigns.

    If dimensionality reduction is to be applied, it is applied here, optionally before
    or after the clustering, with an option to input a pre-instanstiated dimensionality
    reduction object.
    """

    def __init__(self, config: PreProcessConfig, within_pipeline: bool = False):
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
                    f"{self.config.cluster['clustering_method']}_clustering.log"
                ),
            ],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Starting clustering process, checking for precomputed")

        if config.clustering_done:
            self.logger.info("Clustering already complete")
        else:
            self.logger.info("Clustering not yet complete, starting clustering")
            self.processing_path = os.path.join(
                self.config.clustering_path, "processing"
            )
            self._gen_directory_structure()
            self.embeddings = np.load(f"{self.config.embeddings_path}/embeddings.npy")
            self.urls = np.load(f"{self.config.embeddings_path}/doc_ids.npy")
            self.num_clusters = int(np.ceil(np.sqrt(len(self.embeddings))))
            self.config.cluster["num_clusters"] = self.num_clusters
            self.avg_sub_cluster_size = self.config.cluster["avg_sub_cluster_size"]
            self.max_size = self.config.cluster["max_size"]
            self.MULTI_ASSIGN = self.config.cluster.get("MULTI_ASSIGN", 2)

        self.config.save_config()

    def _gen_directory_structure(self):
        """
        Generate the directory structure for the clustering.

        This makes a processing clustering directory, from which the final clusters are
        extracted as the final step in a pipeline
        """

        os.makedirs(self.processing_path, exist_ok=True)
        os.makedirs(
            os.path.join(self.processing_path, "centroids"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(self.processing_path, "assignments"),
            exist_ok=True,
        )
        os.makedirs(os.path.join(self.processing_path, "bundles"), exist_ok=True)
        os.makedirs(
            os.path.join(self.processing_path, "processed_clusters"), exist_ok=True
        )

    def cluster_and_assign(self):
        """
        Cluster and assign embeddings and urls.

        Runs the following:
            - Uses given clustering technique to compute the centroids.
            - Assigns documents to given clusters
            - Processes the urls into associated bundles.
        """
        self.logger.info(
            "Clustering method: %s", self.config.cluster["clustering_method"]
        )
        self.logger.info("Number of clusters: %d", self.num_clusters)
        self._cluster_and_assign()
        self.config.clustering_done = True
        self.config.cluster["num_clusters"] = self.num_clusters
        self.config.save_config()
        if self.within_pipeline:
            return self.config

        return 1

    def _cluster_and_assign(self):
        """Sub method"""
        if self._check_centroids_done():
            self.logger.info("Centroids already computed, skipping computation")
        else:
            self.logger.info("Computing centroids")
            self._compute_centroids(save_centroids=True)

        self.logger.info("Initalised clustering, starting initial assignment")
        if self._check_assignments_done():
            self.logger.info("Assignments already computed, skipping assignment")
        else:
            self._assign_embeddings()

        self.logger.info("Processing clusters")
        self._process_clusters()

    @abstractmethod
    def _compute_centroids(self, num_clusters: int | None = None, save_centroids=False):
        """Compute the centroids for the embeddings. Rewrite in child class for given
        clustering method."""
        raise NotImplementedError()

    def _check_centroids_done(self):
        """Check if the centroids have been computed."""
        centroids_path = f"{self.processing_path}/centroids/centroids.npy"
        return os.path.isfile(centroids_path)

    def _check_assignments_done(self):
        """Check if the assignments have been computed."""
        for idx in range(self.num_clusters):
            assignment_path = f"{self.processing_path}/assignments/cluster_{idx}.txt"
            if not os.path.isfile(assignment_path):
                return False
        return True

    def _assign_embeddings(self):
        """Assign embeddings to the computed centroids."""
        cluster_files = [
            f"{self.processing_path}/assignments/cluster_{i}.txt"
            for i in range(self.num_clusters)
        ]
        assignment_dict = {}
        dim = self.embeddings.shape[1]
        centroids = np.load(f"{self.processing_path}/centroids/centroids.npy")
        if centroids.ndim == 1:
            centroids = centroids.reshape(1, -1)
        index = faiss.IndexFlatL2(dim)
        index.add(centroids)
        self.logger.info("Searching index for nearest centroids")
        distances, assignments = index.search(self.embeddings, self.MULTI_ASSIGN)

        # Find threshold for assigning to multiple clusters
        percentiles = []
        for i in range(1, self.MULTI_ASSIGN):
            percentiles.append(
                np.percentile([dist[i] - dist[0] for dist in distances], 20)
            )

        over_assign_count = 0
        for idx, assignment in tqdm(
            enumerate(assignments), desc="Assigning embeddings"
        ):
            over_assign_count, assignment_dict = self._assign_embedding(
                idx,
                assignment,
                distances,
                percentiles,
                assignment_dict,
                over_assign_count,
            )

        # Write assignments to files
        for idx in tqdm(
            range(self.num_clusters), desc="Writing assignments", unit="cluster"
        ):
            self._write_assignment(cluster_files[idx], idx, assignment_dict)

    def _assign_embedding(
        self,
        idx,
        assignment,
        distances,
        percentiles,
        assignment_dict,
        over_assign_count,
    ):
        """Assign a single embedding to the computed centroids."""
        for k in range(self.MULTI_ASSIGN):
            if k == 0 or (distances[idx][k] - distances[idx][0]) <= percentiles[k - 1]:
                cluster_id = assignment[k]
                if cluster_id not in assignment_dict:
                    assignment_dict[cluster_id] = [idx]
                else:
                    assignment_dict[cluster_id].append(idx)
                if k > 0:
                    over_assign_count += 1
        return over_assign_count, assignment_dict

    def _write_assignment(self, cluster_file, idx, assignments_dict):
        """Write the assignments to a file."""
        with open(cluster_file, "w", encoding="utf-8") as f:
            if idx in assignments_dict:
                for doc_idx in assignments_dict[idx]:
                    embed = self.embeddings[doc_idx]
                    url = self.urls[doc_idx]
                    embed_str = ",".join(map(str, embed))
                    f.write(f"{doc_idx} | {embed_str} | {url}\n")

    def _process_clusters(self):
        """Process the clusters to balance the sizes."""
        cluster_files = [
            f"{self.processing_path}/assignments/cluster_{i}.txt"
            for i in range(self.num_clusters)
        ]
        new_centroids = []
        total_elems = 0
        for idx, cluster_file in tqdm(
            enumerate(cluster_files), desc="Processing clusters", unit="cluster"
        ):
            processing = self._process_cluster(cluster_file, idx, total_elems)
            if processing is None:
                continue
            processed_centroids, sub_elems_count = processing
            new_cluster_centroids = processed_centroids
            total_elems += sub_elems_count
            if len(new_centroids) > 0:
                new_centroids = np.vstack((new_centroids, new_cluster_centroids))
            else:
                new_centroids = new_cluster_centroids

        # Update the final number of clusters
        self.num_clusters = len(new_centroids)
        self.logger.info("Final number of clusters: %d", self.num_clusters)
        self.logger.info("Writing final centroids to file")
        np.save(f"{self.processing_path}/centroids/final_centroids.npy", new_centroids)

    def _process_cluster(self, cluster_file, cluster_idx, total_elem_count):
        """Process a single cluser"""
        self.logger.info("**** PROCESS CLUSTER *****")
        self.logger.info("Processing cluster %d", cluster_idx)
        cluster_contents = utils.parse_file(cluster_file)
        self.logger.info("LEN = %d", len(cluster_contents))
        if len(cluster_contents) == 0:
            return None
        if len(cluster_contents) <= self.max_size:
            self.logger.info("Cluster size within limit, copying to processed")
            with open(
                f"{self.processing_path}/processed_clusters/cluster_{total_elem_count + 1}.txt",
                "w",
                encoding="utf-8",
            ) as f:
                for content in cluster_contents:
                    if len(content) == 3:
                        f.write(f"{content[0]} | {content[1]} | {content[2]}\n")
                    else:
                        self.logger.info("ERORR: content != 3")
                        self.logger.info("%s", len(content))
            cluster_embeddings = np.loadtxt(
                [item[1] for item in cluster_contents], delimiter=","
            )
            if cluster_embeddings.ndim == 1:
                cluster_embeddings = cluster_embeddings.reshape(1, -1)
            return np.array([np.mean(cluster_embeddings, axis=0)]), 1
        
        self.logger.info("Cluster size exceeds limit, sub-clustering")
        new_centroids, assignment_dict = self._sub_clustering(cluster_contents)

        self.logger.info(
            "Writing to file -- %s/cluster_%d.txt",
            self.processing_path + "/processed_clusters",
            cluster_idx,
        )

        sub_elems_count = 0
        for _, sub_cluster in tqdm(
            enumerate(assignment_dict), desc="Writing sub clusters"
        ):
            # Skip empty clusters
            if assignment_dict[sub_cluster] is None:
                continue
            if len(assignment_dict[sub_cluster]) == 0:
                continue

            sub_elems_count += 1
            sub_cluster_idx = total_elem_count + sub_elems_count + 1
            with open(
                f"{self.processing_path}/processed_clusters/"
                f"cluster_{sub_cluster_idx}.txt",
                "a",
                encoding="utf-8",
            ) as f:
                for elem in assignment_dict[sub_cluster]:
                    if len(elem) == 3:
                        f.write(f"{elem[0]} | {elem[1]} | {elem[2]}\n")
                    else:
                        self.logger.info("ERORR: elem != 3")
                        self.logger.info("%s", len(elem))
        self.logger.info("Finished write")

        return new_centroids, sub_elems_count

    def _sub_clustering(self, embedded_cluster_contents):
        """Sub-cluster the contents of a cluster to create bundles."""
        num_sub_clusters = int(
            2.0
            * np.ceil(
                float(len(embedded_cluster_contents)) / float(self.avg_sub_cluster_size)
            )
        )
        self.logger.info("Sub-clustering into %d sub clusters", num_sub_clusters)

        # extract the embeddings from the cluster and sub-cluster
        cluster_embeddings = np.loadtxt(
            [item[1] for item in embedded_cluster_contents], delimiter=","
        )
        if len(cluster_embeddings) <= 1 or len(cluster_embeddings.shape) != 2:
            new_centroids = cluster_embeddings
            assignments = [[0]]
        else:
            new_centroids, assignments = self._sub_cluster(
                cluster_embeddings, num_sub_clusters
            )

        # create assignment dict
        assignment_dict = {}
        for idx, assignment in tqdm(
            enumerate(assignments), desc="Assigning to subclusters"
        ):
            cluster_id = assignment[0]
            if cluster_id not in assignment_dict:
                assignment_dict[cluster_id] = [embedded_cluster_contents[idx]]
            else:
                assignment_dict[cluster_id].append(embedded_cluster_contents[idx])

        # print(f"length of assignments dict {len(assignment_dict)}")
        # print(f"length of new centroids {len(new_centroids)}")
        # print(f"assignment dict keys {list(assignment_dict.keys())}")
        # raise ValueError("Debugging")

        # If all documents are in the same cluster, divide them arbitrarily into
        # subclusters
        if len(assignment_dict) == 1 and num_sub_clusters > 1:
            self.logger.info(
                "All documents are in the same cluster, dividing arbitrarily"
            )
            for i in tqdm(
                range(num_sub_clusters),
                desc="Dividing arbitrarily",
            ):
                new_centroids[i] = new_centroids[list(assignment_dict.keys())[0]]
                upper_bound = min(
                    (i + 1) * self.avg_sub_cluster_size, len(embedded_cluster_contents)
                )
                assignment_dict[i] = [
                    embedded_cluster_contents[j]
                    for j in range(i * self.avg_sub_cluster_size, upper_bound)
                ]

        # If documents are not the same, proceed with further processing
        self.logger.info(
            "Documents are not the same, proceeding with further processing"
        )
        init_clusters = list(assignment_dict.keys()).copy()
        for cluster in tqdm(init_clusters, desc="Processing clusters"):
            if len(assignment_dict[cluster]) > self.max_size:
                self.logger.info(
                    "Cluster %d exceeds max size, creating sub clusters", cluster
                )
                sub_centroids, sub_assigmnet_dict = self._sub_clustering(
                    assignment_dict[cluster]
                )
                replace_idx = sorted(list(sub_assigmnet_dict.keys()))[0]
                new_centroids[cluster] = sub_centroids[replace_idx]
                assignment_dict[cluster] = sub_assigmnet_dict[replace_idx]
                offset = len(new_centroids)
                new_centroids = np.vstack(
                    (new_centroids, sub_centroids[replace_idx + 1 :])
                )
                for sub_cluster in sub_assigmnet_dict:
                    if sub_cluster != replace_idx:
                        assignment_dict[sub_cluster + offset] = sub_assigmnet_dict[
                            sub_cluster
                        ]

        return new_centroids, assignment_dict

    @abstractmethod
    def _sub_cluster(self, embedded_cluster_contents, num_sub_clusters):
        """Sub-cluster the content, rewrite for given clustering method."""
        return NotImplementedError()


class KMeansClusterer(Clusterer):
    """KMeans clustering class."""

    def __init__(self, config: PreProcessConfig, within_pipeline: bool = False):
        super().__init__(config, within_pipeline)
        self.logger.info("Initialized KMeans clustering")

    def _compute_centroids(
        self, num_clusters: int | None = None, save_centroids: bool = False
    ):
        """
        Compute centroids for the embeddings.

        If used for initial assignment this will save the centroids, returning otherwise
        """
        num_clusters = num_clusters if num_clusters is not None else self.num_clusters
        self.logger.info("Computing %d centroids", num_clusters)
        centroids = cm.kmeans_centroids(self.embeddings, self.num_clusters)
        if save_centroids:
            np.save(f"{self.processing_path}/centroids/centroids.npy", centroids)
            return 1
        return centroids

    def _sub_cluster(self, embedded_cluster_contents, num_sub_clusters):
        """Sub-cluster the contents of a cluster using kmeas."""
        return cm.kmeans_sub_cluster(embedded_cluster_contents, num_sub_clusters)
