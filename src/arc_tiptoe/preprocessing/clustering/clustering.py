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
from arc_tiptoe.preprocessing.utils.dim_reduce.dim_reduce import DimReducer

# How many clusters to assign embeddings to if they're close to multiple centroids
MULTI_ASSIGN = 2


class Clusterer(ABC):
    """
    Clutering class, clusters and assigns.

    If dimensionality reduction is to be applied, it is applied here, optionally before
    or after the clustering, with an option to input a pre-instanstiated dimensionality
    reduction object.
    """

    def __init__(
        self,
        config: PreProcessConfig,
        within_pipeline: bool = False,
        dim_reducer: DimReducer = None,
    ):
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
            self.urls = np.load(f"{self.config.embeddings_path}/doc_ids.npy")
            self.num_clusters = np.ceil(np.sqrt(len(self.embeddings))).astype(int)
            self.avg_bundle_size = self.config.cluster["avg_bundle_size"]
            self.urls_per_bundle = self.config.cluster["urls_per_bundle"]
            self.max_size = self.config.cluster["max_size"]
            if self.config.dim_red["apply_dim_red"]:
                if dim_reducer is not None:
                    self.dim_reducer = dim_reducer
                else:
                    self.dim_reducer = DimReducer(self.config, within_pipeline=True)

    def _gen_directory_structure(self):
        """
        Generate the directory structure for the clustering."""
        os.makedirs(
            os.path.join(self.config.clustering_path, "centroids"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(self.config.clustering_path, "assignments"),
            exist_ok=True,
        )
        os.makedirs(os.path.join(self.config.clustering_path, "bundles"), exist_ok=True)

    @abstractmethod
    def _compute_centroids(self):
        """Compute the centroids for the embeddings. Rewrite in child class for given
        clustering method."""
        raise NotImplementedError()

    def _check_centroids_done(self):
        """Check if the centroids have been computed."""
        centroids_path = f"{self.config.clustering_path}/centroids/centroids.npy"
        return os.path.is_file(centroids_path)

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
                    f.write(data_str + "\n")
            else:
                self.logger.info("No assignments for cluster %d", i)

    def _check_assignments_done(self):
        """Check if the assignments have been computed"""
        cluster_files = [
            (f"{self.config.clustering_path}/assignments/cluster_{i}.txt")
            for i in range(self.num_clusters)
        ]
        all_done = True
        for _, cluster_file in enumerate(cluster_files):
            if not os.path.isfile(cluster_file):
                self.logger.info("Cluster file %s not found", cluster_file)
                all_done = False
            else:
                if os.path.getsize(cluster_file) == 0:
                    self.logger.info("Cluster file %s is empty", cluster_file)
                    all_done = False
        return all_done

    def _assign_embeddings(self):
        """Initial assigning of embeddings to clusters."""
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

    def _get_cluster_size(self, cluster_contents):
        """Calculate the size of the contents."""
        out = zlib.compress(
            bytes(
                " ".join([cluster_content[2] for cluster_content in cluster_contents]),
                "utf-8",
            ),
            level=9,
        )
        return len(out)

    def _cluster_by_url(self, cluster_contents):
        """Cluster the contents by URL."""
        cluster_size = self._get_cluster_size(cluster_contents)
        num_bundles = np.ceil(float(cluster_size) / float(self.avg_bundle_size)).astype(
            int
        )
        embed_contents = np.loadtxt(
            [elem[1] for elem in cluster_contents], delimiter=","
        )
        # TODO: PICK BACK UP HERE TOMORROW

    def _singleton_cluster(self, cluster_contents):
        """Handle a cluster that is within size limits by assigning it as is."""
        embed_contents = np.loadtxt(
            [elem[1] for elem in cluster_contents], delimiter=","
        )
        if len(embed_contents) <= 1:
            centroid = embed_contents
        else:
            centroid = np.mean(embed_contents, axis=0)
        clustered_dict = {0: self._cluster_by_url(cluster_contents)}
        return ([centroid], clustered_dict)

    def _process_cluster(self, cluster, idx):
        """Process a single clusters"""
        cluster_contents = self._parse_file(cluster)
        self.logger.info("LEN = %d", len(cluster_contents))

        # If no contents, return
        if len(cluster_contents) == 0:
            self.logger.info("NO CONTENTS")
            return None

        # If the cluster is larger than the stated max size, split the cluster
        if len(cluster_contents) > self.max_size:
            self.logger.info("Cluster larger than max size, splitting")
            centroids, clustered_dict = self._split_cluster(cluster_contents)
        else:
            self.logger.info("Cluster within size limits, assigning as is")
            centroids, clustered_dict = self._singleton_cluster(cluster_contents)

    def _process_clusters(self):
        """Process all clusters to assign overlaps and break up big clusters."""
        cluster_files = [
            (f"{self.config.clustering_path}/assignments/cluster_{i}.txt")
            for i in range(self.num_clusters)
        ]

        # Create directory for final clusters:
        if not os.path.exists(f"{self.config.clustering_path}/clusters"):
            os.makedirs(f"{self.config.clustering_path}/clusters")

        new_centroids = []

        for idx, cluster in tqdm(enumerate(cluster_files), desc="Processing Clusters"):
            self._process_cluster(cluster, idx)

    def clusters_and_assign(self):
        """
        Cluster and assign embeddings and urls.

        Runs the following:
            - Uses given clustering technique to compute the centroids.
            - Assigns documents to given clusters
            - Processes the urls into associated bundles.

        If dim reduction is to be applied, it is applied here, optionally before or
        after the clustering.
        """
        self.logger.info(
            "Clustering method: %s", self.config.cluster["clustering_method"]
        )
        self.logger.info("Number of clusters: %d", self.num_clusters)

        if self.config.dim_red["apply_dim_red"]:
            if self.config.dim_red["dim_red_before_clustering"]:
                self.logger.info("Applying dimensionality reduction before clustering")
                self.dim_reducer.reduce_dimensions()
                self.embeddings = np.load(
                    f"{self.dim_reducer.dim_red_path}/"
                    f"pca_{self.dim_reducer.dim_red_dimension}/"
                    f"embeddings.npy"
                )

                if self._check_centroids_done():
                    self.logger.info("Centroids already computed, skipping computation")
                else:
                    self.logger.info("Computing centroids")
                    self._compute_centroids(initial_assignment=True)

                self.logger.info("Initalised clustering, starting initial assignment")
                if self._check_assignments_done():
                    self.logger.info(
                        "Assignments already computed, skipping assignment"
                    )
                else:
                    self._assign_embeddings()

                self.logger.info("Processing clusters")
                self._process_clusters()

            else:
                self.logger.info("Applying dimensionality reduction after clustering")

        if os.path.isfile(f"{self.config.clustering_path}/centroids/centroids.npy"):
            self.logger.info("Centroids already computed, skipping computation")
        else:
            self.logger.info("Computing centroids")
            self._compute_centroids()

        # if self._check_assignments_done():
        #     self.logger.info("Assignments already computed, skipping assignment")
        # else:
        #     self.logger.info("Initalised clustering, starting assignment")
        #     self._assign_embeddings()

        # self.logger.info("Clustering URLs")
        # self._process_urls()

        # self.config.clustering_done = True
        # self.logger.info("Clustering process complete")
        # self.config.save_config()
        # if self.within_pipeline:
        #     return self.config

        return 1


class KMeansClusterer(Clusterer):
    """KMeans clustering class."""

    def __init__(self, config: PreProcessConfig, within_pipeline: bool = False):
        super().__init__(config, within_pipeline)
        self.logger.info("Initialized KMeans clustering")

    def _compute_centroids(self, intial_assignment: bool = False):
        """
        Compute centroids for the embeddings.

        If used for initial assignment this will save the centroids, returning otherwise
        """
        centroids = cm.kmeans_centroids(self.embeddings, self.num_clusters)
        if initial_assignment:
            np.save(f"{self.config.clustering_path}/centroids/centroids.npy", centroids)
            return 1
        return centroids

    # def _embed_contents(self, contents, num_bundles):
    #     """Embed the contents using FAISS."""
    #     return cm.kmeans_embed_contents(contents, num_bundles, self.logger)


### WIP
# def _get_size(self, contents):
#     """Calculate the size of the contents."""
#     out = zlib.compress(
#         bytes(" ".join([content[2] for content in contents]), "utf-8"), level=9
#     )
#     return len(out)

# @abstractmethod
# def _embed_contents(self, contents, num_bundles):
#     """Embed the contents using FAISS. Rewrite in child class for given clustering
#     method."""
#     raise NotImplementedError()

# def _create_assignment_dict(self, cluster_pair, contents, assignment_dict, idx):
#     """Create an assignment dictionary for a given cluster pair."""
#     cluster = cluster_pair[0]
#     if cluster not in assignment_dict:
#         assignment_dict[cluster] = [contents[idx]]
#     else:
#         assignment_dict[cluster].append(contents[idx])
#     return assignment_dict

# def _divide_contents_arbitrarily(self, contents, centroids, assignment_dict):
#     """Divide contents arbitrarily when all documents are the same."""
#     self.logger.info("All documents assigned to one cluster, dividing arbitrarily")
#     for i in tqdm(
#         range(int(np.ceil(len(contents) / float(self.urls_per_bundle)))),
#         desc="Dividing contents arbitrarily",
#     ):
#         centroids[i] = centroids[list(assignment_dict.keys())[0]]
#         upper_bound = min((i + 1) * self.urls_per_bundle, len(contents))
#         assignment_dict[i] = [
#             contents[j] for j in range(i * self.urls_per_bundle, upper_bound)
#         ]
#     return (centroids, assignment_dict)

# def _process_large_url_clusters(self, cluster, assignment_dict, centroids):
#     """Process large URL clusters by creating sub-bundles."""
#     self.logger.info("Cluster %d exceed max size, creating bundles", cluster)
#     (sub_centroids, sub_assignment_dict) = self._create_bundles(
#         assignment_dict[cluster]
#     )
#     replace_idx = sorted(sub_assignment_dict.keys())[0]
#     centroids[cluster] = sub_centroids[replace_idx]
#     assignment_dict[cluster] = sub_assignment_dict[replace_idx]
#     offset = len(centroids)
#     centroids = np.vstack((centroids, sub_centroids[replace_idx + 1 :]))
#     for sub_cluster in sub_assignment_dict:
#         if sub_cluster != replace_idx:
#             assignment_dict[sub_cluster + offset] = sub_assignment_dict[sub_cluster]

# def _create_bundles(self, contents):
#     """Create bundles of URLs based on clustering."""
#     self.logger.info("Creating bundles")
#     num_bundles = int(
#         np.ceil(float(self._get_size(contents)) / float(self.avg_bundle_size))
#     )
#     self.logger.info("Num_bundles = %d", num_bundles)

#     # Embed contents and perform clustering
#     centroids, assignments = self._embed_contents(contents, num_bundles)

#     # Create a dictionary to hold the assignments
#     self.logger.info("Creating assignment dictionary")
#     assignment_dict = {}
#     for idx, cluster_pair in tqdm(
#         enumerate(assignments), desc="Creating assignment dictionary"
#     ):
#         assignment_dict = self._create_assignment_dict(
#             cluster_pair, contents, assignment_dict, idx
#         )

#     # Divide arbitrarily when all documents are the same
#     if len(assignment_dict) == 1 and num_bundles > 1:
#         centroids, assignments = self._divide_contents_arbitrarily(
#             contents, centroids, assignment_dict
#         )
#         return (centroids, assignments)

#     # If documents are not same, proceed with further processing
#     self.logger.info("Documents assigned to multiple clusters, proceeding")
#     init_clusters = list(assignment_dict.keys()).copy()
#     for cluster in tqdm(init_clusters, desc="Processing initial clusters"):
#         if self._get_size(assignment_dict[cluster]) > self.max_size:
#             self._process_large_url_clusters(cluster, assignment_dict, centroids)

#     return (centroids, assignment_dict)

# def _write_bundle(self, cluster_idx, bundle, assignment_dict):
#     """Write a single bundle to file."""
#     with open(
#         f"{self.config.clustering_path}/bundles/"
#         f"{cluster_idx}/clusters/bundle_{bundle}.txt",
#         "w",
#         encoding="utf-8",
#     ) as f:
#         self.logger.info("Writing content to bundle %d", bundle)
#         for elem in assignment_dict[bundle]:
#             if len(elem) == 3:
#                 f.write(f"{elem[0]} | {elem[1]} | {elem[2]}\n")
#             else:
#                 self.logger.warning(
#                     "Skipping malformed element in bundle %d: %s", bundle, elem
#                 )
#                 self.logger.warning("Element length: %d", len(elem))

# def _process_cluster(self, cluster_file, cluster_idx):
#     """Process a single cluster file to create bundles and centroids."""
#     self.logger.info("**** PROCESS CLUSTER *****")
#     contents = self._parse_file(cluster_file)
#     self.logger.info("LEN = %d", len(contents))
#     if len(contents) == 0:
#         return

#     centroids, assignment_dict = self._create_bundles(contents)

#     self.logger.info("Writing to file -- %s/%s", cluster_file, cluster_idx)
#     if not os.path.exists(f"{self.config.clustering_path}/bundles/{cluster_idx}/"):
#         os.makedirs(f"{self.config.clustering_path}/bundles/{cluster_idx}/")
#     if not os.path.exists(
#         f"{self.config.clustering_path}/bundles/{cluster_idx}/clusters/"
#     ):
#         os.makedirs(
#             f"{self.config.clustering_path}/bundles/{cluster_idx}/clusters/"
#         )
#     np.savetxt(
#         f"{self.config.clustering_path}/bundles/{cluster_idx}/centroids.npy",
#         centroids,
#     )

#     for bundle in tqdm(assignment_dict, desc="Writing bundles"):
#         self._write_bundle(cluster_idx, bundle, assignment_dict)

#     self.logger.info("Finished writing bundles for cluster %d", cluster_idx)

# def _process_urls(self):
#     """Process URLs associated with embeddings."""
#     cluster_files = [
#         (f"{self.config.clustering_path}/assignments/cluster_{i}.txt")
#         for i in range(self.num_clusters)
#     ]

#     for idx, cluster in enumerate(cluster_files):
#         self._process_cluster(cluster, idx)
