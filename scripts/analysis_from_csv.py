import logging
import os

import ir_datasets

from arc_tiptoe.constants import RESULTS_DIR
from arc_tiptoe.eval.accuracy.analysis import evaluate_queries, run_analysis
from arc_tiptoe.eval.utils import parse_clustering_search_results
from arc_tiptoe.preprocessing.utils.tfidf import load_queries_from_ir_datasets
from arc_tiptoe.utils import save_to_json


def clustering_analysis(csv_filepath: str) -> dict:
    """Run analysis on search results from a CSV file."""
    log_info = f"Parsing search results from {csv_filepath}..."
    logging.info(log_info)
    all_results = parse_clustering_search_results(csv_filepath)
    dataset_name = "msmarco-document/trec-dl-2019"

    dataset = ir_datasets.load(dataset_name)
    query_list = load_queries_from_ir_datasets(dataset=dataset)

    outputs_dir = os.path.join(
        RESULTS_DIR, dataset_name.replace("/", "_"), "embedding_model"
    )

    for n_clusters, cluster_results in all_results.items():
        all_results_path = os.path.join(
            outputs_dir,
            "all_results",
            f"{n_clusters}_clusters.json",
        )
        results = evaluate_queries(query_list, cluster_results)

        save_to_json(
            results,
            all_results_path,
            indent=2,
        )
        mean_results_path = os.path.join(
            outputs_dir,
            "mean_results",
            f"{n_clusters}_clusters.json",
        )
        mean_metrics = run_analysis(all_results_path, verbose=True)
        save_to_json(
            mean_metrics._asdict(),
            mean_results_path,
            indent=2,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    clustering_analysis(
        "results/msmarco-document_trec-dl-2019/distilbert/search_results/test.csv"
    )
