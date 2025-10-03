"""
Methods required for Mean Reciprocal Rank (MRR) calculation.
"""


def reciprocal_rank(
    retrieved_docs: list[str],
    relevant_docs: list[str],
    eval_most_relevant: bool = True,
) -> float:
    """
    Calculate Reciprocal Rank (RR) for a single query.

    Args:
        retrieved_docs: List of retrieved document IDs.
        relevant_docs: List of relevant document IDs.
        eval_most_relevant: If True, only the most relevant document is considered.

    Returns:
        Reciprocal Rank (RR) as a float.
    """
    if eval_most_relevant:
        relevant_docs = [relevant_docs[0]]
    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in relevant_docs:
            return 1.0 / (i + 1)
    return 0.0
