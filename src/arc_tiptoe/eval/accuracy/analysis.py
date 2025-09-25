import json


def load_results(results_pth: str) -> dict:
    """Load retrieval results from a JSON file."""
    with open(results_pth) as f:
        return json.load(f)


def mean_f1_metrics(all_results: dict) -> tuple[float, float, float]:
    """Calculate mean precision, recall, and F1-score from all query results."""
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    num_queries = len(all_results)

    for _, metrics in all_results.items():
        total_precision += metrics["precision"]
        total_recall += metrics["recall"]
        total_f1 += metrics["f1"]

    mean_precision = total_precision / num_queries if num_queries > 0 else 0.0
    mean_recall = total_recall / num_queries if num_queries > 0 else 0.0
    mean_f1 = total_f1 / num_queries if num_queries > 0 else 0.0

    return mean_precision, mean_recall, mean_f1


def mean_dcg_metrics(all_results: dict) -> float:
    """Calculate mean Discounted Cumulative Gain (DCG) from all query results."""
    total_dcg = 0.0
    total_cg = 0.0
    total_ndcg = 0.0
    num_queries = len(all_results)

    for _, metrics in all_results.items():
        total_cg += metrics["CG"]
        total_dcg += metrics["DCG"]
        total_ndcg += metrics["nDCG"]

    mean_dcg = total_dcg / num_queries if num_queries > 0 else 0.0
    mean_cg = total_cg / num_queries if num_queries > 0 else 0.0
    mean_ndcg = total_ndcg / num_queries if num_queries > 0 else 0.0

    return mean_cg, mean_dcg, mean_ndcg


def mean_rr_metrics(all_results: dict) -> float:
    """Calculate mean Reciprocal Rank (MRR) from all query results."""
    total_rr = 0.0
    num_queries = len(all_results)

    for _, metrics in all_results.items():
        total_rr += metrics["RR"]

    return total_rr / num_queries if num_queries > 0 else 0.0


def run_analysis(results_pth: str, verbose: bool = False) -> None:
    """Run analysis on retrieval results and print mean metrics."""
    all_results = load_results(results_pth)

    mean_precision, mean_recall, mean_f1 = mean_f1_metrics(all_results)
    mean_cg, mean_dcg, mean_ndcg = mean_dcg_metrics(all_results)
    mean_rr = mean_rr_metrics(all_results)

    if verbose:
        print("\n=== Mean Metrics ===")
        print(f"Mean Precision: {mean_precision:.4f}")
        print(f"Mean Recall: {mean_recall:.4f}")
        print(f"Mean F1-Score: {mean_f1:.4f}")
        print(f"Mean CG: {mean_cg:.4f}")
        print(f"Mean DCG: {mean_dcg:.4f}")
        print(f"Mean nDCG: {mean_ndcg:.4f}")
        print(f"Mean Reciprocal Rank (MRR): {mean_rr:.4f}")

    return {
        "Mean Precision": mean_precision,
        "Mean Recall": mean_recall,
        "Mean F1-Score": mean_f1,
        "Mean CG": mean_cg,
        "Mean DCG": mean_dcg,
        "Mean nDCG": mean_ndcg,
        "Mean Reciprocal Rank (MRR)": mean_rr,
    }
