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
