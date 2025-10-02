import ast
import csv
import logging


def parse_clustering_search_results(filepath: str) -> dict[int, dict[str, dict]]:
    """
    Parse search results CSV with number of clusters as outer key.

    Returns:
        dict: {
            num_clusters: {
                query_id: {
                    'retrieved_docs': [...],  # concatenated results from clusters
                    'scores': [...],          # corresponding scores
                    'total_comm': float       # sum of cluster communications
                }
            }
        }
    """
    # First pass: organize data by query_id and cluster
    raw_data: dict[str, dict[int, dict]] = {}

    with open(filepath) as file:
        reader = csv.DictReader(file)
        headers = reader.fieldnames

        # Extract cluster numbers
        cluster_nums = []
        if headers:
            for header in headers:
                if header.startswith("cluster_") and header.endswith("_res"):
                    cluster_num = int(header.split("_")[1])
                    cluster_nums.append(cluster_num)

        cluster_nums.sort()

        for row in reader:
            query_id = row["query_id"]
            raw_data[query_id] = {}

            for cluster_num in cluster_nums:
                res_key = f"cluster_{cluster_num}_res"
                comm_key = f"cluster_{cluster_num}_total_comm"

                try:
                    parsed_cluster_results = ast.literal_eval(row[res_key])
                    sorted_results = sorted(
                        parsed_cluster_results, key=lambda x: x["score"], reverse=True
                    )
                    cluster_results = {
                        "retrieved_docs": [item["url"] for item in sorted_results],
                        "scores": [item["score"] for item in sorted_results],
                    }
                except (ValueError, SyntaxError):
                    log_msg = (
                        f"Failed to parse results for query {query_id}, cluster "
                        f"{cluster_num}. Setting empty list."
                    )
                    logging.warning(log_msg)
                    cluster_results = {"retrieved_docs": [], "scores": []}

                try:
                    total_comm = float(row[comm_key])
                except (ValueError, TypeError):
                    log_msg = (
                        f"Failed to parse total_comm for query {query_id}, cluster "
                        f"{cluster_num}. Setting to 0.0."
                    )
                    logging.warning(log_msg)
                    total_comm = 0.0

                raw_data[query_id][cluster_num] = {
                    "results": cluster_results,
                    "total_comm": total_comm,
                }

    # Second pass: reorganize with num_clusters as outer key and concatenate results
    results: dict[int, dict[str, dict]] = {}

    for num_clusters in range(1, len(cluster_nums) + 1):
        results[num_clusters] = {}

        for query_id, query_data in raw_data.items():
            # Get clusters 1 through num_clusters for this query
            clusters_to_use = [i for i in range(1, num_clusters + 1) if i in query_data]

            # Concatenate results from all clusters up to num_clusters
            all_docs = []
            all_scores = []
            total_comm_sum = 0.0

            for cluster_num in clusters_to_use:
                cluster_data = query_data[cluster_num]
                all_docs.extend(cluster_data["results"]["retrieved_docs"])
                all_scores.extend(cluster_data["results"]["scores"])
                total_comm_sum += cluster_data["total_comm"]

            # Sort combined results by score (descending)
            if all_docs and all_scores:
                combined = list(zip(all_docs, all_scores, strict=True))
                combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)
                all_docs_tuple, all_scores_tuple = zip(*combined_sorted, strict=True)
                all_docs = list(all_docs_tuple)
                all_scores = list(all_scores_tuple)

            results[num_clusters][query_id] = {
                "retrieved_docs": all_docs,
                "scores": all_scores,
                "total_comm": total_comm_sum,
            }

    return results
