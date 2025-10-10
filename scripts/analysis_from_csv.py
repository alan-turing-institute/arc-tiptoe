import logging
import os
from typing import NamedTuple

import ir_datasets
import jsonargparse

from arc_tiptoe.constants import RESULTS_DIR
from arc_tiptoe.eval.accuracy.analysis import evaluate_queries, run_analysis
from arc_tiptoe.eval.utils import parse_clustering_search_results
from arc_tiptoe.preprocessing.utils.tfidf import load_queries_from_ir_datasets
from arc_tiptoe.search.tfidf import get_top_relevance_level
from arc_tiptoe.utils import DATASET_SAVE_MAP, save_to_json


class clustering_args(NamedTuple):
    """
    Arguments for clustering analysis.

    Args:
        csv_filepath: Path to the CSV file containing search results.
        n_results: Number of results to consider.
        log_level: Logging level.
    """

    csv_filepath: str
    n_results: int
    log_level: str


def clustering_analysis(args: clustering_args):
    """
    Run analysis on search results from a CSV file.
    Args:
        args: Arguments for clustering analysis which contains:
            - csv_filepath: Path to the CSV file containing search results.
            - n_results: Number of results to consider.
            - log_level: Logging level.

    Returns:
        None
    """
    logging.getLogger().setLevel(args.log_level.upper())
    log_info = f"Parsing search results from {args.csv_filepath}..."
    logging.info(log_info)
    all_results = parse_clustering_search_results(args.csv_filepath)

    # parse dataset name from filepath and load IR dataset object and queries
    dataset_name = args.csv_filepath.split("/")[1]
    ir_dataset_name = DATASET_SAVE_MAP.get(dataset_name, dataset_name.replace("_", "/"))
    dataset = ir_datasets.load(ir_dataset_name)
    query_list = load_queries_from_ir_datasets(dataset=dataset)

    # create output directories
    outputs_dir = os.path.join(RESULTS_DIR, dataset_name, "embedding_model")

    # loop through n_clusters and save results and analysis
    for n_clusters, cluster_results in all_results.items():
        # evaluate and save results
        results = evaluate_queries(
            query_list,
            cluster_results,
            top_k_docs=args.n_results,
            target_relevance_level=get_top_relevance_level(dataset),
        )
        mean_results_path = os.path.join(
            outputs_dir,
            "mean_metrics",
            f"{args.n_results}_results",
            f"{n_clusters}_clusters.json",
        )
        mean_metrics = run_analysis(results, verbose=False)
        save_to_json(
            mean_metrics._asdict(),
            mean_results_path,
            indent=2,
        )


if __name__ == "__main__":
    arg_parser = jsonargparse.ArgumentParser()
    arg_parser.add_argument(
        "--csv_filepath",
        type=str,
        default="results/msmarco-document_trec-dl-2019/distilbert/search_results/test.csv",
        help="Path to the CSV file containing search results.",
    )
    arg_parser.add_argument(
        "--n_results", type=int, default=100, help="Number of results to consider."
    )
    arg_parser.add_argument(
        "--log_level", type=str, default="WARNING", help="Logging level."
    )
    args = arg_parser.parse_args()

    clustering_analysis(args)
