import json
import logging
import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from arc_tiptoe.constants import RESULTS_DIR
from arc_tiptoe.eval.accuracy.dcg import (
    cumulative_gain,
    discounted_cumulative_gain,
    normalized_discounted_cumulative_gain,
)
from arc_tiptoe.eval.accuracy.f1 import f1, precision, recall
from arc_tiptoe.eval.accuracy.mrr import reciprocal_rank
from arc_tiptoe.preprocessing.utils.tfidf import get_relevant_docs


def save_results_to_json(results, filename):
    """Save search results to a JSON file."""
    save_path = os.path.join(RESULTS_DIR, "tfidf", filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    msg = f"Results saved to {save_path}"
    logging.info(msg)

    return save_path


def search_queries(query_list, doc_ids, vectorizer, tfidf_matrix, n_results):
    """Execute queries using TF-IDF similarity and save results."""

    search_results = {}

    for qid, original_query, processed_query, _ in query_list:
        # Transform query to TF-IDF vector
        query_vector = vectorizer.transform([processed_query])

        # Calculate cosine similarity with all documents
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

        # Get top results
        top_indices = np.argsort(similarities)[::-1][:n_results]

        # Store results for this query
        query_outputs = {
            "original_query": original_query,
            "processed_query": processed_query,
            "retrieved_docs": [],
            "scores": [],
        }

        for doc_idx in top_indices:
            if (
                similarities[doc_idx] > 0
            ):  # Only include documents with positive similarity
                query_outputs["retrieved_docs"].append(doc_ids[doc_idx])
                query_outputs["scores"].append(float(similarities[doc_idx]))
            search_results[qid] = query_outputs

    return search_results


def evaluate_queries(query_list, search_results, print_results=False):
    all_results = {}
    for qid, _, _, qrels in query_list:
        query_results = {}
        query_search_results = search_results[qid]
        relevant_document_ids = get_relevant_docs(qrels, target_relevance_level=None)
        query_results["precision"] = precision(
            query_search_results["retrieved_docs"], relevant_document_ids
        )
        query_results["recall"] = recall(
            query_search_results["retrieved_docs"], relevant_document_ids
        )
        query_results["f1"] = f1(query_results["precision"], query_results["recall"])
        query_results["CG"] = cumulative_gain(
            query_search_results["retrieved_docs"], qrels
        )
        query_results["DCG"] = discounted_cumulative_gain(
            query_search_results["retrieved_docs"], qrels
        )
        query_results["nDCG"] = normalized_discounted_cumulative_gain(
            query_search_results["retrieved_docs"], qrels
        )
        target_relevance_level = 3
        while target_relevance_level > 1:
            top_relevant_docs = get_relevant_docs(
                qrels, target_relevance_level=target_relevance_level
            )
            if len(top_relevant_docs) > 1:
                break
            msg = (
                f"No top relevant docs (rel=={target_relevance_level}) for query {qid}."
                f"Trying rel=={target_relevance_level - 1} docs."
            )
            logging.info(msg)
            target_relevance_level -= 1

        if target_relevance_level == 1:
            err_msg = f"No relevant docs (rel>0) for query {qid}."
            logging.warning(err_msg)
            top_relevant_docs = []
        query_results["RR"] = reciprocal_rank(
            query_search_results["retrieved_docs"],
            top_relevant_docs,
            eval_most_relevant=False,  # get_relevant_docs returns multiple targets here
        )
        query_results.update(query_results)
        all_results[qid] = query_results

        if print_results:
            # Display results (keeping original display format)
            print(f"Query: {qid}")
            print("----- F1 score metrics -----")
            print(f"Precision: {query_results['precision']:.4f}")
            print(f"Recall: {query_results['recall']:.4f}")
            print(f"F1 Score: {query_results['f1']:.4f}")
            print("----------------------------")
            print("")
            print("-- Cumulative Gain Metrics --")
            print(f"Cumulative Gain: {query_results['CG']:.4f}")
            print(f"Discounted Cumulative Gain: {query_results['DCG']:.4f}")
            print(f"Normalized Discounted Cumulative Gain: {query_results['nDCG']:.4f}")
            print("-----------------------------")
            print("")
            print("-- Reciprocal Rank Metrics --")
            print(f"Reciprocal Rank: {query_results['RR']:.4f}")
            print("-----------------------------")
            print("")

    # Save results to JSON
    return all_results


def search_queries_and_evaluate(
    query_list, doc_ids, vectorizer, tfidf_matrix, n_results
):
    preliminary_results = search_queries(
        query_list, doc_ids, vectorizer, tfidf_matrix, n_results
    )
    return evaluate_queries(query_list, preliminary_results)
