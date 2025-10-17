import logging
import os

import ir_datasets
import jsonargparse

from arc_tiptoe.constants import RESULTS_DIR
from arc_tiptoe.eval.accuracy.analysis import evaluate_queries, run_analysis
from arc_tiptoe.model.tfidf import (
    check_existing_model,
    load_tfidf_model,
    train_tfidf_model,
)
from arc_tiptoe.preprocessing.utils.tfidf import (
    load_doc_ids_only,
    load_documents_from_ir_datasets,
    load_queries_from_ir_datasets,
)
from arc_tiptoe.search.tfidf import (
    SearchConfig,
    check_existing_results,
    get_top_relevance_level,
    save_to_json,
    search_queries_batch,
)
from arc_tiptoe.utils import parse_json


def main(args):
    log_level = args.log_level.upper()
    logging.getLogger().setLevel(log_level)
    """Main function to run TF-IDF search using ir_datasets."""
    max_docs = args.max_documents  # Number of documents to use
    n_results = args.num_results  # Number of search results to return
    batch_size = args.batch_size  # Process documents in batches

    dataset = ir_datasets.load(args.dataset_name)
    dataset_name = args.dataset_name.replace("/", "_")

    n_docs = max_docs if max_docs else dataset.docs_count()

    # Check for existing results before loading documents
    preliminary_results_dir = check_existing_results(
        dataset_name=dataset_name,
        doc_count=n_docs,
    )
    if preliminary_results_dir:
        msg = f"""Found existing search results for:
        \tDATASET = {dataset_name},
        \tMAX_DOCUMENTS = {n_docs},
        """
        logging.info(msg)
        preliminary_results = parse_json(preliminary_results_dir)

    else:
        # Check for existing model before loading full documents
        model_path_stem = check_existing_model(
            dataset_name=dataset_name, doc_count=n_docs
        )

        if model_path_stem:
            msg = f"""Found existing TF-IDF model for:
            DATASET = {dataset_name},
            MAX_DOCUMENTS = {n_docs}.
            """
            logging.info(msg)
            model = load_tfidf_model(model_path_stem)
            doc_ids = load_doc_ids_only(
                dataset=dataset,
                max_documents=n_docs,
            )
        else:
            msg = f"""Did not find existing TF-IDF model for:
            DATASET = {dataset_name},
            MAX_DOCUMENTS = {n_docs}.
            """
            logging.info(msg)
            doc_ids, doc_contents = load_documents_from_ir_datasets(
                dataset=dataset,
                max_documents=n_docs,
                batch_size=batch_size,
            )
            model = train_tfidf_model(
                dataset_name=dataset_name, doc_contents=doc_contents
            )

    # Run queries
    query_list = load_queries_from_ir_datasets(dataset=dataset)

    outputs_dir = os.path.join(RESULTS_DIR, dataset_name, "tfidf")

    if "preliminary_results" not in locals():
        search_config = SearchConfig(
            max_documents=n_docs, results_per_query=n_results, save_path=outputs_dir
        )
        preliminary_results = search_queries_batch(
            query_list, doc_ids, model, search_config
        )

    # Perform evaluation
    results = evaluate_queries(
        query_list,
        preliminary_results,
        target_relevance_level=get_top_relevance_level(dataset),
        top_k_docs=n_results,
    )

    mean_results = run_analysis(results)
    save_to_json(
        mean_results._asdict(),
        save_path=os.path.join(
            outputs_dir, "mean_metrics", f"{n_docs}_docs", f"{n_results}_results.json"
        ),
        indent=2,
    )


if __name__ == "__main__":
    arg_parser = jsonargparse.ArgumentParser()
    arg_parser.add_argument(
        "--dataset_name",
        type=str,
        default="msmarco-document/trec-dl-2019",
        help="Name of the ir_datasets dataset to use. "
        "Default is 'msmarco-document/trec-dl-2019'.",
    )
    arg_parser.add_argument(
        "--max_documents",
        type=int,
        default=None,
        help="Maximum number of documents to process (for testing). "
        "Default is None (all documents).",
    )
    arg_parser.add_argument(
        "--num_results",
        type=int,
        default=100,
        help="Number of search results to return. Default is 100.",
    )
    arg_parser.add_argument(
        "--batch_size",
        type=int,
        default=5000,
        help="Batch size for processing documents. Default is 5000.",
    )
    arg_parser.add_argument(
        "--log_level",
        type=str,
        default="WARNING",
        help="Logging level. Default is WARNING.",
    )

    args = arg_parser.parse_args()
    main(args)
