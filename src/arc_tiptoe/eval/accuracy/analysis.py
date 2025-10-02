import logging
from typing import NamedTuple

from tqdm import tqdm

from arc_tiptoe.eval.accuracy.dcg import (
    cumulative_gain,
    discounted_cumulative_gain,
    normalized_discounted_cumulative_gain,
)
from arc_tiptoe.eval.accuracy.f1 import f1, precision, recall
from arc_tiptoe.eval.accuracy.mrr import reciprocal_rank
from arc_tiptoe.preprocessing.utils.tfidf import get_relevant_docs
from arc_tiptoe.utils import parse_json


class EvalMetrics(NamedTuple):
    precision: float
    recall: float
    f1: float
    CG: float
    DCG: float
    nDCG: float
    MRR: float


def mean_f1_metrics(all_results: dict[str, dict]) -> tuple[float, float, float]:
    """Calculate mean precision, recall, and F1-score from all query results."""
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    num_queries = len(all_results)

    for _, metrics in all_results.items():
        total_precision += metrics.get("precision", 0.0)
        total_recall += metrics.get("recall", 0.0)
        total_f1 += metrics.get("f1", 0.0)

    mean_precision = total_precision / num_queries if num_queries > 0 else 0.0
    mean_recall = total_recall / num_queries if num_queries > 0 else 0.0
    mean_f1 = total_f1 / num_queries if num_queries > 0 else 0.0

    return mean_precision, mean_recall, mean_f1


def mean_dcg_metrics(all_results: dict[str, dict]) -> float:
    """Calculate mean Discounted Cumulative Gain (DCG) from all query results."""
    total_dcg = 0.0
    total_cg = 0.0
    total_ndcg = 0.0
    num_queries = len(all_results)

    for _, metrics in all_results.items():
        total_cg += metrics.get("CG", 0.0)
        total_dcg += metrics.get("DCG", 0.0)
        total_ndcg += metrics.get("nDCG", 0.0)

    mean_dcg = total_dcg / num_queries if num_queries > 0 else 0.0
    mean_cg = total_cg / num_queries if num_queries > 0 else 0.0
    mean_ndcg = total_ndcg / num_queries if num_queries > 0 else 0.0

    return mean_cg, mean_dcg, mean_ndcg


def mean_rr_metrics(all_results: dict[str, dict]) -> float:
    """Calculate mean Reciprocal Rank (MRR) from all query results."""
    total_rr = 0.0
    num_queries = len(all_results)

    for _, metrics in all_results.items():
        total_rr += metrics.get("RR", 0.0)

    return total_rr / num_queries if num_queries > 0 else 0.0


def run_analysis(results_pth: str, verbose: bool = False) -> EvalMetrics:
    """Run analysis on retrieval results and print mean metrics.
    Args:
        results_pth (str): Path to the JSON file containing all query results.
        verbose (bool): Whether to print detailed metrics. Defaults to False.
    Returns:
        EvalMetrics: NamedTuple containing mean precision, recall, F1, CG, DCG, nDCG,
        and MRR.
    """
    all_results = parse_json(results_pth)

    mean_precision, mean_recall, mean_f1 = mean_f1_metrics(all_results)
    mean_cg, mean_dcg, mean_ndcg = mean_dcg_metrics(all_results)
    mrr = mean_rr_metrics(all_results)

    if verbose:
        print("\n=== Mean Metrics ===")
        print(f"Mean Precision: {mean_precision:.4f}")
        print(f"Mean Recall: {mean_recall:.4f}")
        print(f"Mean F1-Score: {mean_f1:.4f}")
        print(f"Mean CG: {mean_cg:.4f}")
        print(f"Mean DCG: {mean_dcg:.4f}")
        print(f"Mean nDCG: {mean_ndcg:.4f}")
        print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")

    return EvalMetrics(
        precision=mean_precision,
        recall=mean_recall,
        f1=mean_f1,
        CG=mean_cg,
        DCG=mean_dcg,
        nDCG=mean_ndcg,
        MRR=mrr,
    )


def _evaluate(
    query_results: dict,
    qid: str,
    query_search_results: dict,
    qrels: dict,
    relevant_document_ids: list,
    target_relevance_level: int = 3,
) -> dict:
    query_results["precision"] = precision(
        query_search_results["retrieved_docs"], relevant_document_ids
    )
    query_results["recall"] = recall(
        query_search_results["retrieved_docs"], relevant_document_ids
    )
    query_results["f1"] = f1(query_results["precision"], query_results["recall"])
    query_results["CG"] = cumulative_gain(query_search_results["retrieved_docs"], qrels)
    query_results["DCG"] = discounted_cumulative_gain(
        query_search_results["retrieved_docs"], qrels
    )
    query_results["nDCG"] = normalized_discounted_cumulative_gain(
        query_search_results["retrieved_docs"], qrels
    )
    while target_relevance_level > 0:
        top_relevant_docs = get_relevant_docs(
            qrels, target_relevance_level=target_relevance_level
        )
        if len(top_relevant_docs) >= 1:
            break
        msg = (
            f"No top relevant docs (rel=={target_relevance_level}) for query {qid}."
            f"Trying rel=={target_relevance_level - 1} docs."
        )
        logging.info(msg)
        target_relevance_level -= 1

    if target_relevance_level == 0:
        err_msg = f"No relevant docs (rel>0) for query {qid}."
        logging.warning(err_msg)
        top_relevant_docs = []
    query_results["RR"] = reciprocal_rank(
        query_search_results["retrieved_docs"],
        top_relevant_docs,
        eval_most_relevant=False,  # get_relevant_docs returns multiple targets here
    )
    return query_results


def evaluate_queries(
    query_list, search_results, print_results=False, target_relevance_level=None
):
    """
    Evaluate retrieval results for a list of queries.

    Args:
        query_list: List of queries with their IDs and relevance judgments.
        search_results: Dictionary of search results keyed by query ID.
        print_results: Whether to print results. Defaults to False.
        target_relevance_level: The relevance level to evaluate. Defaults to None.

    Returns:
        Dictionary of evaluation results for each query.
    """
    all_results = {}
    for qid, _, _, qrels in tqdm(query_list, desc="Query No."):
        query_results = {}
        query_search_results = search_results[qid]
        relevant_document_ids = get_relevant_docs(
            qrels, target_relevance_level=target_relevance_level
        )

        query_results.update(
            _evaluate(
                query_results=query_results,
                qid=qid,
                query_search_results=query_search_results,
                qrels=qrels,
                relevant_document_ids=relevant_document_ids,
                target_relevance_level=target_relevance_level,
            )
        )

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
