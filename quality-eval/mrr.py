import re
import statistics
import sys

import numpy

MRR_RANK = 100
# MRR_RANK = 100


def read_top_results(filename):
    score_dict = dict()
    with open(filename) as file:
        lines = [line.rstrip() for line in file]

    query = ""
    for line in lines:
        m = re.match("Query: (.+)", line)
        if m:
            match = m.group(1)
            query = match.split(" ")[0]
        else:
            tokens = line.split()
            if (
                len(tokens) > 1
                and query not in score_dict
                and "/home/ubuntu" not in tokens[0]
            ):
                score_dict[query] = [tokens[1]]
            elif (
                len(tokens) > 1
                and query in score_dict
                and len(score_dict[query]) < MRR_RANK
                and "/home/ubuntu" not in tokens[0]
            ):
                score_dict[query].append(tokens[1])
    return score_dict


def read_ranked_qrel(filename):
    # PATCHED FOR MSMARCO - Handle tab-separated qrels format
    lines = open(filename).read().splitlines()
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
    mrr = 0.0
    num_ranked = 0
    for qid in result_dict:
        real = real_dict[qid]
        for i, result in enumerate(result_dict[qid]):
            if real == result and i < MRR_RANK:
                num_ranked += 1
                mrr += 1.0 / float(i + 1)
    print("Num ranked = %d" % num_ranked)
    print("Total = %d" % len(result_dict))
    return mrr / float(len(result_dict))


def main():
    results = read_top_results(sys.argv[1])
    real = read_ranked_qrel("/home/ubuntu/msmarco/msmarco-docdev-qrels.tsv")
    mrr = compute_mrr(results, real)
    print("MRR@%d: %f" % (MRR_RANK, mrr))


if __name__ == "__main__":
    main()
