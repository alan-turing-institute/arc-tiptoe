import logging

import jsonargparse

from arc_tiptoe.eval.accuracy.analysis import run_analysis
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
from arc_tiptoe.search.tfidf import save_results_to_json, search_queries_and_evaluate


def main(args):
    log_level = args.log_level.upper()
    logging.getLogger().setLevel(log_level)
    """Main function to run TF-IDF search using ir_datasets."""
    dataset_name = args.dataset_name
    max_docs = args.max_documents  # Number of documents to use
    n_results = args.num_results  # Number of search results to return
    batch_size = args.batch_size  # Process documents in batches

    # Check for existing model before loading documents
    n_docs = max_docs if max_docs else "full"
    model_name = check_existing_model(n_docs)
    if model_name:
        msg = f"Found existing TF-IDF model for: MAX_DOCUMENTS = {n_docs}."
        logging.info(msg)
        vectorizer, tfidf_matrix = load_tfidf_model(model_name)
        doc_ids = load_doc_ids_only(
            max_documents=n_docs,
            dataset_name=dataset_name,
        )
    else:
        doc_ids, doc_contents = load_documents_from_ir_datasets(
            dataset_name=dataset_name,
            max_documents=n_docs,
            batch_size=batch_size,
        )
        vectorizer, tfidf_matrix = train_tfidf_model(doc_contents)

    # Run queries and display results
    query_list = load_queries_from_ir_datasets()
    results = search_queries_and_evaluate(
        query_list,
        doc_ids,
        vectorizer,
        tfidf_matrix,
        n_results,
    )
    filename = f"tfidf_all_results_{n_docs}_docs.json"
    save_path = save_results_to_json(results, filename=filename)

    mean_results = run_analysis(save_path)
    save_results_to_json(
        mean_results, filename=f"tfidf_mean_results_{n_docs}_docs.json"
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
        default="full",
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
