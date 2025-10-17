import logging


def mean_communication_cost(all_results: dict[str, dict]) -> float | None:
    total_comm_cost = 0.0
    num_queries = 0

    for query_id, metrics in all_results.items():
        comm_cost = metrics.get("total_comm", None)
        if comm_cost is not None:
            total_comm_cost += comm_cost
            num_queries += 1
        else:
            warn_msg = f"Communication cost not found for query {query_id}. Skipping."
            logging.warning(warn_msg)
            # If not found skip
            continue

    return total_comm_cost / num_queries if num_queries > 0 else None
