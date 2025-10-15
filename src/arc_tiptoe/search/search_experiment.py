"""
Full search experiment class.
"""

import asyncio
import json
import os
import subprocess
from pathlib import Path
from uuid import uuid4

import faiss
import ir_datasets
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm_sync
from tqdm.asyncio import tqdm


class SearchExperimentAsync:
    """Class to run a full search experiment"""

    def __init__(
        self,
        config_path: str,
        search_dir: str = "search",
        dataset_name: str = "msmarco-passage/dev/small",
    ):
        self.config_path = config_path
        with open(config_path, encoding="utf-8") as f:
            self.config = json.load(f)

        self.search_dir = Path(search_dir).resolve()
        self.dataset_name = dataset_name
        self.queries = []
        self.cluster_search_num = self.config["clustering"].get("search_top_k", 4)
        self.results_df = pd.DataFrame(
            columns=["query_id"]
            + [f"cluster_{i+1}_res" for i in range(self.cluster_search_num)]
            + [f"cluster_{i+1}_total_comm" for i in range(self.cluster_search_num)]
        )
        self.results = []
        self.num_clusters = self.config["clustering"].get("total_clusters")

        self._load_queries()

    def _load_queries(self):
        query_iter = ir_datasets.load(self.dataset_name).queries_iter()
        self.queries = list(query_iter)

    async def _single_query_search(self, query):
        """Run search for a single query."""
        cmd = [
            "go",
            "run",
            "main.go",
            "--search_config",
            f"../{self.config_path}",
            "multi-cluster-experiment",
            query[1],
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.search_dir,
        )

        stdout, stderr = await process.communicate()

        print(stderr.decode())

        output = stdout.decode().splitlines()
        print(output)
        result = self._parse_go_output(output[-1], query[0])
        self.results.append(result)
        return result

    def _parse_go_output(self, output, query_id):
        """Parse the output from the Go program and return a structured dictionary."""

        results = json.loads(output.splitlines()[-1])
        structured_results = {"query_id": query_id}
        for cluster_res in results["all_results"]:
            structured_results[f"cluster_{cluster_res['cluster_rank']}_res"] = (
                cluster_res["results"]
            )
            structured_results[f"cluster_{cluster_res['cluster_rank']}_total_comm"] = (
                cluster_res["perf_up"] + cluster_res["perf_down"]
            )
        return structured_results

    async def run_experiment(self, output_path: str | None = None):
        """Run the full search experiment."""
        tasks = [self._single_query_search(query) for query in self.queries]
        await tqdm.gather(*tasks)
        for result in self.results:
            self.results_df = pd.concat(
                [self.results_df, pd.DataFrame([result])], ignore_index=True
            )
        if output_path is not None:
            self._save_results(output_path)
        return self.results_df

    def _save_results(self, output_path: str):
        """Save the results DataFrame to a CSV file."""
        self.results_df.to_csv(output_path, index=False)


class SearchExperimentSingleThread:
    """Class to run a full search experiment"""

    def __init__(
        self,
        config_path: str,
        queries_path: str,
        search_dir: str = "search",
    ):
        self.config_path = config_path
        with open(config_path, encoding="utf-8") as f:
            self.config = json.load(f)

        self.queries_path = queries_path
        self.search_dir = Path(search_dir).resolve()
        self.queries = None
        self.cluster_search_num = self.config["clustering"].get("search_top_k", 4)
        self.results_df = pd.DataFrame(
            columns=["query_id"]
            + [f"cluster_{i+1}_res" for i in range(self.cluster_search_num)]
            + [f"cluster_{i+1}_total_comm" for i in range(self.cluster_search_num)]
        )
        self.results = []
        self.pca_components = self._load_pca_components()
        self.faiss_index_path = (
            f"data/{self.config['uuid']}/artifact/"
            f"dim{self.config['embedding'].get('reduced_dimension', 192)}/index.faiss"
        )
        self.centroids_path = f"data/{self.config['uuid']}/clusters/centroids.txt"
        self.cluster_index = self._load_cluster_index()
        self.num_clusters = self.config["clustering"].get("total_clusters")

        self._load_queries()

    def _load_queries(self):
        """Load preprocessed queries from a CSV file.

        Datframe should have columns: 'query_id', 'query_embed', 'query_text',
        """
        self.queries = pd.read_csv(self.queries_path)

    def _load_pca_components(self):
        """Load PCA components from a file if needed."""
        return np.load(self.config["dim_reduction"]["pca_components_file"])

    def _load_cluster_index(self):
        """Load the FAISS index for cluster search"""
        if os.path.exists(self.faiss_index_path):
            return faiss.read_index(self.faiss_index_path)

        centroids = np.loadtxt(self.centroids_path)
        cluster_index = faiss.IndexFlatL2(centroids.shape[1])
        cluster_index.add(centroids.astype("float32"))
        return cluster_index

    def _find_nearest_clusters(
        self, embedding: np.ndarray, top_k: int | None
    ) -> list[int]:
        """Find the top-k nearest clusters fro a given embedding"""
        if self.cluster_index is None:
            return [0]

        _, indices = self.cluster_index.search(embedding.reshape(1, -1), top_k)

        # only return valid clusters
        valid_clusters = []
        for i in range(len(indices[0])):
            if indices[0][i] < self.num_clusters:
                valid_clusters.append(int(indices[0][i]))

        return valid_clusters

    def _process_query(self, query_embed):
        """Process a single query embedding prior to search.

        This runs the search over the centroids to extract the top-k clusters, and
        runs the dimensionality reduction (if needed) and quantisation
        """
        if self.config["dim_reduction"]["applied"]:
            query_embed = np.matmul(query_embed, self.pca_components)

        cluster_indices = self._find_nearest_clusters(
            query_embed, self.cluster_search_num
        )

        # Quantise embedding
        data_min = np.min(query_embed)
        data_max = np.max(query_embed)
        data_range = max(abs(data_min), abs(data_max))
        scale = 127 / data_range if data_range != 0 else 1.0
        query_embed_quant = np.clip(np.round(query_embed * scale), -127, 127).astype(
            np.int8
        )

        return query_embed_quant, cluster_indices

    def _single_query_search(self, query):
        """Run search for a single query."""
        query_id = query["query_id"]
        query_embed = np.array(json.loads(query["embedding"]))
        query_text = query["text"]

        processed_query_embed, cluster_indices = self._process_query(query_embed)
        query_dict = {
            "queryEmbed": processed_query_embed.tolist(),
            "clusterIndices": cluster_indices,
            "queryText": query_text,
        }
        tmp_filename = f"{self.search_dir}/tmp_{uuid4()}.json"
        with open(tmp_filename, "w", encoding="utf-8") as f:
            json.dump(query_dict, f)

        cmd = [
            "go",
            "run",
            "main.go",
            "--search_config",
            f"{self.config_path}",
            "multi-cluster-experiment",
            tmp_filename,
        ]

        process = subprocess.run(
            cmd,
            cwd=self.search_dir,
            check=True,
            capture_output=True,
        )

        stdout = process.stdout
        output = stdout.decode("utf-8").splitlines()

        result = self._parse_go_output(output[-1], query_id)
        self.results.append(result)
        return result

    def _parse_go_output(self, output, query_id):
        """Parse the output from the Go program and return a structured dictionary."""

        results = json.loads(output.splitlines()[-1])
        structured_results = {"query_id": query_id}
        for cluster_res in results["all_results"]:
            structured_results[f"cluster_{cluster_res['cluster_rank']}_res"] = (
                cluster_res["results"]
            )
            structured_results[f"cluster_{cluster_res['cluster_rank']}_total_comm"] = (
                cluster_res["perf_up"] + cluster_res["perf_down"]
            )
        return structured_results

    def run_experiment(self, output_path: str | None = None):
        """Run the full search experiment."""
        _ = [
            self._single_query_search(query)
            for _, query in tqdm_sync(self.queries.iterrows())
        ]
        for result in self.results:
            self.results_df = pd.concat(
                [self.results_df, pd.DataFrame([result])], ignore_index=True
            )
        if output_path is not None:
            self._save_results(output_path)
        return self.results_df

    def _save_results(self, output_path: str):
        """Save the results DataFrame to a CSV file."""
        self.results_df.to_csv(output_path, index=False)
