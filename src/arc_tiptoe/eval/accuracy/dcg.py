import numpy as np


def cumulative_gain(retrieved_docs: list[str], qrels: dict[str, int]) -> float:
    """
    Calculate Cumulative Gain (CG) for retrieved documents.

    Args:
        retrieved_docs: List of retrieved document IDs.
        qrels: Dictionary of relevant document IDs with their relevance scores.

    Returns:
        CG score as a float.
    """
    cg = 0.0
    for doc_id in retrieved_docs:
        if doc_id in qrels:
            cg += qrels.get(doc_id, 0)
    return cg


def discounted_cumulative_gain(
    retrieved_docs: list[str], qrels: dict[str, int]
) -> float:
    """
    Calculate Discounted Cumulative Gain (DCG) for retrieved documents.

    Args:
        retrieved_docs: List of retrieved document IDs.
        qrels: Dictionary of relevant document IDs with their relevance scores.

    Returns:
        DCG score as a float.
    """
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in qrels:
            dcg += qrels.get(doc_id, 0) / np.log2(i + 2)
    return dcg


def normalized_discounted_cumulative_gain(
    retrieved_docs: list[str], qrels: dict[str, int]
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (nDCG) for retrieved documents.

    Args:
        retrieved_docs: List of retrieved document IDs.
        qrels: Dictionary of relevant document IDs with their relevance scores.

    Returns:
        nDCG score as a float.
    """
    dcg = discounted_cumulative_gain(retrieved_docs, qrels)
    ideal_dcg = discounted_cumulative_gain(
        sorted(qrels.keys(), key=lambda x: qrels[x], reverse=True), qrels
    )
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
