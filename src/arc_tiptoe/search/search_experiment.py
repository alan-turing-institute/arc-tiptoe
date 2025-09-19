"""
Full search experiment class.
"""

import asyncio
import json
from pathlib import Path

import ir_datasets
import pandas as pd


class SearchExperiment:
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
        self.cluster_search_num = self.config.clustering.get("search_top_k", 4)
        self.results_df = pd.DataFrame(
            columns=["query_id"]
            + [f"cluster_{i+1}_res" for i in range(self.cluster_search_num)]
            + [f"cluster_{i+1}_total_comm" for i in range(self.cluster_search_num)]
        )

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
            "--searh_config",
            f"../{self.config_path}",
            "multi-cluster-experiment",
            query["text"],
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.search_dir,
        )

        stdout, _ = await process.communicate()

        output = stdout.decode().splitlines()[-1]
        output_dict = self._parse_go_output(output, query["id"])
        self.results_df = pd.concat(
            [self.results_df, pd.DataFrame(output_dict, index=[0])], ignore_index=True
        )
        return output_dict

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
        await asyncio.gather(*tasks)
        if output_path is not None:
            self._save_results(output_path)
        return self.results_df

    def _save_results(self, output_path: str):
        """Save the results DataFrame to a CSV file."""
        self.results_df.to_csv(output_path, index=False)
