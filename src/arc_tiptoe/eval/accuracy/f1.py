"""
Methods required to compute precision, recall, and F1 score.
"""


def get_relevant_docs(doc_dict: dict[str, int]) -> list[str]:
    """
    Get relevant document IDs for a given query ID.

    Args:
        qrels_ref: A reference to the qrels dictionary.
        query_id: The ID of the query.

    Returns:
        A list of relevant document IDs.
    """
    # return [doc_id for doc_id, rel in doc_dict.items() if rel > 0]
    return [doc_id for doc_id, rel in doc_dict.items() if rel > 0]


def recall(retrieved_docs: list[str], relevant_docs: list[str]) -> float:
    """
    Calculate recall of retrieved documents against relevant documents.

    Args:
        retrieved_docs: List of retrieved document IDs.
        relevant_docs: List of relevant document IDs.

    Returns:
        Recall as a float.
    """
    retrieved_set = set(retrieved_docs)
    relevant_set = set(relevant_docs)
    true_positives = len(retrieved_set.intersection(relevant_set))
    return true_positives / len(relevant_set)


def precision(retrieved_docs: list[str], relevant_docs: list[str]) -> float:
    """
    Calculate precision of retrieved documents against relevant documents.

    Args:
        retrieved_docs: List of retrieved document IDs.
        relevant_docs: List of relevant document IDs.

    Returns:
        Precision as a float.
    """
    retrieved_set = set(retrieved_docs)
    relevant_set = set(relevant_docs)
    true_positives = len(retrieved_set.intersection(relevant_set))
    return true_positives / len(retrieved_set)


def f1(precision: float, recall: float) -> float:
    """
    Calculate F1 score given precision and recall.

    Args:
        precision: Precision value.
        recall: Recall value.

    Returns:
        F1 score as a float.
    """
    return (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
