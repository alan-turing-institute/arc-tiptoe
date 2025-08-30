"""
Functions for computing the MRR
"""

import re
import sys

MRR_RANK = 100
# MRR_RANK = 100


def read_top_results(filename):
    """Read the top result from the file"""
    score_dict = dict()
    with open(filename, encoding="utf-8") as file:
        lines = [line.rstrip() for line in file]

    query = ""
    for line in lines:
        m = re.match("Query: (.+)", line)
        if m:
            match = m.group(1)
            query_parts = match.split(" ")
            if query_parts:
                query = query_parts[0]  # First word is query ID
                # Skip if query ID is empty or invalid
                if not query or not query.isdigit():
                    query = ""
                    continue
        else:
            tokens = line.split()
            if (
                len(tokens) > 1
                and query  # Make sure query is not empty
                and query not in score_dict
                and "/home/ubuntu" not in tokens[0]
                and "/home/azureuser" not in tokens[0]  # Also filter azureuser paths
            ):
                try:
                    # Validate that first token is a number (score)
                    float(tokens[0])
                    score_dict[query] = [tokens[1]]
                except ValueError:
                    continue
            elif (
                len(tokens) > 1
                and query  # Make sure query is not empty
                and query in score_dict
                and len(score_dict[query]) < MRR_RANK
                and "/home/ubuntu" not in tokens[0]
                and "/home/azureuser" not in tokens[0]  # Also filter azureuser paths
            ):
                try:
                    # Validate that first token is a number (score)
                    float(tokens[0])
                    score_dict[query].append(tokens[1])
                except ValueError:
                    continue
    return score_dict


def read_ranked_qrel(filename):
    """
    Read the ranked qrels from the file
    """
    # PATCHED FOR MSMARCO - Handle tab-separated qrels format
    lines = open(filename, encoding="utf-8").read().splitlines()
    result_dict = dict()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Try tab-separated first (MSMARCO format)
        parts = line.split("\t")
        if len(parts) < 4:
            # Fall back to space-separated (original format)
            parts = line.split(" ")

        if len(parts) >= 4:
            try:
                qid = parts[0]
                docid = parts[2]
                relevance = int(parts[3])

                # Only keep relevant documents (relevance > 0)
                if relevance > 0 and qid not in result_dict:
                    result_dict[qid] = docid
            except (ValueError, IndexError):
                continue

    return result_dict


def compute_mrr(result_dict, real_dict):
    """Given the results and ground truth, compute the mrr"""
    mrr = 0.0
    num_ranked = 0
    for qid in result_dict:
        real = real_dict[qid]
        for i, result in enumerate(result_dict[qid]):
            if real == result and i < MRR_RANK:
                num_ranked += 1
                mrr += 1.0 / float(i + 1)
    print(f"Num ranked = {num_ranked}")
    print(f"Total = {len(result_dict)}")
    return mrr / float(len(result_dict))


def main():
    """Main entry point"""
    results = read_top_results(sys.argv[1])
    real = read_ranked_qrel("/home/azureuser/msmarco_data/msmarco-docdev-qrels.tsv")
    mrr = compute_mrr(results, real)
    print(f"MRR@{MRR_RANK}: {mrr}")


if __name__ == "__main__":
    main()
