import os
from typing import NamedTuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from arc_tiptoe.constants import RESULTS_DIR
from arc_tiptoe.model.tfidf import TFIDFModel
from arc_tiptoe.utils import save_to_json


class SearchConfig(NamedTuple):
    """
    Configuration for the search process.

    Attributes:
        max_documents (int): Maximum number of documents to consider.
        results_per_query (int): Number of results to return per query.
        save_path (str): Path to save the search results.
    """

    max_documents: int
    results_per_query: int
    save_path: str


class QueryResult(NamedTuple):
    """
    Summary of the query results.

    Attributes:
        original_query (str): The original query string.
        processed_query (str): The processed query string.
        retrieved_docs (list): List of retrieved document IDs.
        scores (list): List of scores for the retrieved documents.
    """

    original_query: str
    processed_query: str
    retrieved_docs: list[str]
    scores: list[float | int]


def get_top_relevance_level(dataset):
    meta_data = dataset.metadata()
    rel_counts = meta_data["qrels"]["fields"]["relevance"]["counts_by_value"]
    vals = [int(k) for k in rel_counts if k.isdigit()]
    return max(vals) if vals else None


def check_existing_results(dataset_name, doc_count):
    """Check if search results already exist for given parameters."""
    data_save_name = dataset_name.replace("/", "_")  # should have happened already
    results_dir = os.path.join(RESULTS_DIR, data_save_name, "tfidf", "search_results")
    results_name = f"{doc_count}_docs.json"
    results_path = os.path.join(results_dir, results_name)
    if os.path.exists(results_path):
        return results_path
    return None


def search_queries(query_list, doc_ids, model: TFIDFModel, search_config: SearchConfig):
    """Execute queries using TF-IDF similarity and save results."""
    search_results = {}

    for qid, original_query, processed_query, _ in tqdm(query_list, desc="Query No."):
        # Transform query to TF-IDF vector
        query_vector = model.vectorizer.transform([processed_query])

        # Calculate cosine similarity with all documents
        similarities = cosine_similarity(query_vector, model.tfidf_matrix).flatten()

        # Get top results
        top_indices = np.argsort(similarities)[::-1][: search_config.results_per_query]

        # Store results for this query
        query_outputs = QueryResult(original_query, processed_query, [], [])

        for doc_idx in top_indices:
            if (
                similarities[doc_idx] > 0
            ):  # Only include documents with positive similarity
                query_outputs.retrieved_docs.append(doc_ids[doc_idx])
                query_outputs.scores.append(float(similarities[doc_idx]))
            search_results[qid] = query_outputs._asdict()

    # Save results to JSON
    save_path = os.path.join(
        RESULTS_DIR,
        search_config.save_path,
        "search_results",
        f"{search_config.max_documents}_docs.json",
    )
    save_to_json(results=search_results, save_path=save_path)
    return search_results


def search_queries_batch(
    query_list,
    doc_ids,
    model: TFIDFModel,
    search_config: SearchConfig,
    batch_size: int = 32,
):
    """
    Execute queries using TF-IDF similarity in batches for improved performance.

    Args:
        query_list: List of tuples containing (qid, original_query, processed_query, _)
        doc_ids: List of document IDs corresponding to the TF-IDF matrix
        model: TFIDFModel containing the vectorizer and TF-IDF matrix
        search_config: SearchConfig containing search parameters
        batch_size: Number of queries to process in each batch (default: 32)

    Returns:
        Dictionary containing search results for all queries
    """
    search_results = {}

    # Process queries in batches
    for i in tqdm(range(0, len(query_list), batch_size), desc="Batch No."):
        batch = query_list[i : i + batch_size]
        batch_qids = [item[0] for item in batch]
        batch_original_queries = [item[1] for item in batch]
        batch_processed_queries = [item[2] for item in batch]

        # Transform all queries in batch to TF-IDF vectors
        batch_query_vectors = model.vectorizer.transform(batch_processed_queries)

        # Calculate cosine similarity for all queries in batch at once
        batch_similarities = cosine_similarity(batch_query_vectors, model.tfidf_matrix)

        # Process results for each query in the batch
        for j, (qid, original_query, processed_query) in enumerate(
            zip(
                batch_qids,
                batch_original_queries,
                batch_processed_queries,
                strict=True,
            )
        ):
            similarities = batch_similarities[j]

            # Get top results
            top_indices = np.argsort(similarities)[::-1][
                : search_config.results_per_query
            ]

            # Store results for this query
            query_outputs = QueryResult(original_query, processed_query, [], [])

            for doc_idx in top_indices:
                if similarities[doc_idx] > 0:  # Only positive similarity
                    query_outputs.retrieved_docs.append(doc_ids[doc_idx])
                    query_outputs.scores.append(float(similarities[doc_idx]))
            search_results[qid] = query_outputs._asdict()

    # Save results to JSON
    save_path = os.path.join(
        RESULTS_DIR,
        search_config.save_path,
        "search_results",
        f"{search_config.max_documents}_docs.json",
    )
    save_to_json(results=search_results, save_path=save_path)
    return search_results
