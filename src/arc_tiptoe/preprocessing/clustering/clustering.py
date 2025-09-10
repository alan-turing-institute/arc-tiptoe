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
from arc_tiptoe.preprocessing.dim_reduce.dim_reduce import DimReducer
from arc_tiptoe.preprocessing.dim_reduce.dim_reducers import dim_reducers
from arc_tiptoe.preprocessing.utils.config import PreProcessConfig

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
                    f"data/{self.config.uuid}"
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
                    self.dim_reducer = dim_reducers[
                        self.config.dim_red["dim_red_method"]
                    ](self.config, within_pipeline=False)

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
    def _compute_centroids(
        self, num_clusters: int | None = None, initial_assignment=False
    ):
        """Compute the centroids for the embeddings. Rewrite in child class for given
        clustering method."""
        raise NotImplementedError()

    def _check_centroids_done(self):
        """Check if the centroids have been computed."""
        centroids_path = f"{self.config.clustering_path}/centroids/centroids.npy"
        return os.path.isfile(centroids_path)

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

    def _dim_red_contents(self, cluster_contents, embed_contents):
        """Dimensionality reduce the contents if needed."""
        dim_red_embed_contents = self.dim_reducer.transform_embedding(embed_contents)
        return [
            (
                cluster_contents[i][0],
                ",".join([str(ch) for ch in dim_red_embed_contents[i]]),
                cluster_contents[i][2],
            )
            for i in range(len(cluster_contents))
        ]

    @abstractmethod
    def _sub_clustering(self, embedded_cluster_contents, num_bundles):
        """
        Sub-cluster the contents of a cluster. Overwrite for given clustering method
        """
        raise NotImplementedError()

    def _divide_cluster_arbitrarily(
        self,
        cluster_contents,
        embed_contents,
        num_bundles,
        apply_dim_red,
        dim_red_before,
    ):
        """Divide a cluster arbitrarily when all documents are the same."""
        self.logger.info("All documents assigned to one cluster, dividing arbitrarily")
        output_list = []
        for idx in tqdm(range(num_bundles), desc="Dividing arbitrarily"):
            upper_bound = min((idx + 1) * self.urls_per_bundle, len(cluster_contents))
            if apply_dim_red and not dim_red_before:
                output_list.append(
                    self._dim_red_contents(
                        cluster_contents[idx * self.urls_per_bundle : upper_bound],
                        embed_contents[idx * self.urls_per_bundle : upper_bound],
                    )
                )
            else:
                output_list.append(
                    cluster_contents[idx * self.urls_per_bundle : upper_bound]
                )
        return output_list

    def _divide_cluster_recursively(
        self, assignments_dict, apply_dim_red, dim_red_before
    ):
        """Divide the cluster recursively"""
        output_list = []
        init_clusters = list(assignments_dict.keys()).copy()
        for cluster in init_clusters:
            if self._get_cluster_size(assignments_dict[cluster]) > self.max_size:
                sub_output_list = self._cluster_by_url(assignments_dict[cluster])
                output_list = output_list + sub_output_list
            else:
                embed_contents = np.loadtxt(
                    [elem[1] for elem in assignments_dict[cluster]], delimiter=","
                )
                if len(np.shape(embed_contents)) < 2:
                    if apply_dim_red and not dim_red_before:
                        self.logger.info("Applying dim reduction after clustering")
                        output_contents = self._dim_red_contents(
                            assignments_dict[cluster], embed_contents
                        )
                    else:
                        output_list.append(output_contents)
                else:
                    if apply_dim_red and not dim_red_before:
                        self.logger.info("Applying dim reduction after clustering")
                        output_list.append(
                            self._dim_red_contents(
                                assignments_dict[cluster], embed_contents
                            )
                        )
                    else:
                        output_list.append(assignments_dict[cluster])
        return output_list

    def _cluster_by_url(self, cluster_contents):
        """Cluster the contents by URL."""
        cluster_size = self._get_cluster_size(cluster_contents)
        num_bundles = np.ceil(float(cluster_size) / float(self.avg_bundle_size)).astype(
            int
        )
        embed_contents = np.loadtxt(
            [elem[1] for elem in cluster_contents], delimiter=","
        )
        apply_dim_red = self.config.dim_red["apply_dim_red"]
        dim_red_before = self.config.dim_red["dim_red_before_clustering"]
        if len(embed_contents.shape) < 2:
            if apply_dim_red and not dim_red_before:
                self.logger.info("Applying dim reduction after clustering")
                return [self._dim_red_contents(cluster_contents, embed_contents)]
            return [cluster_contents]
        if num_bundles == 1:
            if apply_dim_red and not dim_red_before:
                self.logger.info("Applying dim reduction after clustering")
                return [self._dim_red_contents(cluster_contents, embed_contents)]
            return [cluster_contents]

        if len(embed_contents) > 1 and len(np.shape(embed_contents)) == 2:
            assignments = self._sub_clustering(embed_contents, num_bundles)
        else:
            assignments = [[0]]

        assignments_dict = {}
        for idx, cluster_pair in enumerate(assignments):
            cluster = cluster_pair[0]
            if cluster not in assignments_dict:
                assignments_dict[cluster] = [cluster_contents[idx]]
            else:
                assignments_dict[cluster].append(cluster_contents[idx])

        # divide arbitrarily when all docs the same
        if len(assignments_dict) == 1 and num_bundles > 1:
            return self._divide_cluster_arbitrarily(
                cluster_contents,
                embed_contents,
                num_bundles,
                apply_dim_red,
                dim_red_before,
            )

        # recursively compute otherwise
        return self._divide_cluster_recursively(
            assignments_dict, apply_dim_red, dim_red_before
        )

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

    def _split_cluster(self, cluster_contents):
        """Split a cluster that is too large into smaller clusters."""
        num_bundles = int(
            2
            * np.ceil(
                float(
                    self._get_cluster_size(cluster_contents)
                    / float(self.avg_bundle_size)
                )
            ).astype(int)
        )
        self.logger.info("Splitting cluster into %d bundles", num_bundles)
        embed_contents = np.loadtxt(
            [elem[1] for elem in cluster_contents], delimiter=","
        )
        if len(embed_contents) > 1 and len(np.shape(embed_contents)) == 2:
            assignments = self._sub_clustering(embed_contents, num_bundles)
            centroids = self._compute_centroids(num_clusters=num_bundles)
        else:
            return []

        assignments_dict = {}
        membership_dict = {}
        for idx in range(num_bundles):
            assignments_dict[idx] = []
            membership_dict[idx] = set()
        for idx, cluster_pair in enumerate(assignments):
            cluster = cluster_pair[0]
            assignments_dict[cluster].append(cluster_contents[idx])
            membership_dict[cluster].add(cluster_contents[idx][2])

        num_zeros = 0
        for i in range(num_bundles):
            if len(assignments_dict[i]) == 0:
                num_zeros += 1
        self.logger.info("Number of empty sub-clusters: %d", num_zeros)
        if num_zeros == num_bundles - 1 and num_bundles > 0:
            for idx in range(num_bundles):
                centroids[i] = centroids[next(iter(assignments_dict.keys()))]
                upper_bound = min(
                    (idx + 1) * self.avg_bundle_size, len(cluster_contents)
                )
                assignments_dict[idx] = self._cluster_by_url(
                    [
                        cluster_contents[j]
                        for j in range(idx * self.avg_bundle_size, upper_bound)
                    ]
                )
            return centroids, assignments_dict

        # Clear out empty clusters and recompute centroids
        new_assignments_dict = {}
        num_zeros = 0
        for idx in range(len(centroids)):
            if idx in assignments_dict:
                new_assignments_dict[idx - num_zeros] = assignments_dict[idx]
                if len(assignments_dict[idx - num_zeros]) > 0:
                    centroids[idx - num_zeros] = np.loadtxt(
                        [elem[1] for elem in assignments_dict[idx]], delimiter=","
                    )
                else:
                    num_zeros += 1
        centroids = centroids[: len(centroids) - num_zeros]
        assignments_dict = new_assignments_dict

        init_clusters = list(assignments_dict.keys()).copy()
        clustered_dict = {}
        for cluster in init_clusters:
            if len(assignments_dict[cluster]) > self.max_size:
                sub_centroids, sub_clustered_dict = self._split_cluster(
                    assignments_dict[cluster]
                )
                offset = len(centroids)
                centroids[cluster] = sub_centroids[0]
                clustered_dict[cluster] = sub_clustered_dict[0]
                if len(sub_centroids) > 1:
                    centroids = np.row_stack((centroids, sub_centroids[1:]))
                for sub_cluster in sub_clustered_dict:
                    if sub_cluster != 0:
                        clustered_dict[sub_cluster + offset - 1] = sub_clustered_dict[
                            sub_cluster
                        ]
            else:
                clustered_dict[cluster] = self._cluster_by_url(
                    assignments_dict[cluster]
                )

        return centroids, clustered_dict

    def _pack_url_bundles(self, bundles):
        """Pack url bundles into a single string."""
        packed_bundles = []
        bundles.sort(key=self._get_cluster_size, reverse=True)
        for new_bundles in bundles:
            placed = False
            for idx, packed_bundle in enumerate(packed_bundles):
                if (
                    not placed
                    and self._get_cluster_size(packed_bundle + new_bundles)
                    < self.urls_per_bundle
                ):
                    packed_bundles[idx] = packed_bundle + new_bundles
                    placed = True
            if not placed:
                packed_bundles.append(new_bundles)
        return packed_bundles

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

        # Write clusters
        self.logger.info("Writing processed cluster")
        if not os.path.exists(f"{self.config.clustering_path}/clusters/{idx}"):
            os.makedirs(f"{self.config.clustering_path}/clusters/{idx}")

        total_elems = 0
        for idx, cluster in tqdm(enumerate(clustered_dict), desc="Writing clusters"):
            file_name = (
                f"{self.config.clustering_path}/clusters/{idx}/cluster_{idx}.txt"
            )
            with open(file_name, "w", encoding="utf-8") as f:
                packed_url_bundles = self._pack_url_bundles(clustered_dict[cluster])
                for j, bundle in enumerate(packed_url_bundles):
                    for elem in bundle:
                        if len(elem) == 3:
                            f.write(f"{elem[0]} | {elem[1]} | {elem[2]}")
                            total_elems += 1
                    f.write("-------------------------\n")
        return total_elems, centroids

    def _process_clusters(self):
        """Process all clusters to assign overlaps and break up big clusters."""
        cluster_files = [
            (f"{self.config.clustering_path}/assignments/cluster_{i}.txt")
            for i in range(self.num_clusters)
        ]

        # Create directory for final clusters:
        if not os.path.exists(f"{self.config.clustering_path}/clusters"):
            os.makedirs(f"{self.config.clustering_path}/clusters")

        centroids = []
        total = 0
        for idx, cluster in tqdm(enumerate(cluster_files), desc="Processing Clusters"):
            return_val = self._process_cluster(cluster, idx)
            if return_val is None:
                continue

            total_elems, cluster_centroids = return_val
            total += total_elems
            self.logger.info("Total elements in processed cluster: %d", total_elems)
            if len(centroids) > 0:
                centroids = np.row_stack((centroids, cluster_centroids))
            else:
                centroids = cluster_centroids
        np.savetxt(
            f"{self.config.clustering_path}/centroids/final_centroids.txt", centroids
        )
        with open(
            f"{self.config.clustering_path}/cluster_count.txt", "w", encoding="utf-8"
        ) as f:
            f.write(f"Total clusters: {total}")

    def _cluster_and_assign(self):
        """Sub method"""
        if self._check_centroids_done():
            self.logger.info("Centroids already computed, skipping computation")
        else:
            self.logger.info("Computing centroids")
            self._compute_centroids(initial_assignment=True)

        self.logger.info("Initalised clustering, starting initial assignment")
        if self._check_assignments_done():
            self.logger.info("Assignments already computed, skipping assignment")
        else:
            self._assign_embeddings()

        self.logger.info("Processing clusters")
        self._process_clusters()

    def cluster_and_assign(self):
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

                self._cluster_and_assign()
                self.config.dim_red_done = True
                self.config.clustering_done = True
                self.config.save_config()
                if self.within_pipeline:
                    return self.config
                return 1

            self.logger.info("Applying dimensionality reduction after clustering")
            self._cluster_and_assign()
            self.config.dim_red_done = True
            self.config.clustering_done = True
            self.config.save_config()
            if self.within_pipeline:
                return self.config
            return 1

        self.logger.info("No application of dimensionality reduction")
        self._cluster_and_assign()
        self.config.clustering_done = True
        self.config.save_config()
        if self.within_pipeline:
            return self.config
        return 1


class KMeansClusterer(Clusterer):
    """KMeans clustering class."""

    def __init__(
        self,
        config: PreProcessConfig,
        within_pipeline: bool = False,
        dim_reducer: DimReducer = None,
    ):
        super().__init__(config, within_pipeline, dim_reducer)
        self.logger.info("Initialized KMeans clustering")

    def _compute_centroids(
        self, num_clusters: int | None = None, initial_assignment: bool = False
    ):
        """
        Compute centroids for the embeddings.

        If used for initial assignment this will save the centroids, returning otherwise
        """
        num_clusters = num_clusters if num_clusters is not None else self.num_clusters
        self.logger.info("Computing %d centroids", num_clusters)
        centroids = cm.kmeans_centroids(self.embeddings, self.num_clusters)
        if initial_assignment:
            np.save(f"{self.config.clustering_path}/centroids/centroids.npy", centroids)
            return 1
        return centroids

    def _sub_clustering(self, embedded_cluster_contents, num_bundles):
        """Sub-cluster the contents of a cluster using kmeas."""
        return cm.kmeans_sub_cluster(embedded_cluster_contents, num_bundles)
