"""Search quality experiment script"""

# import concurrent.futures
import glob
import json
import multiprocessing
import os
import sys
import threading

import faiss
import numpy
import requests

# from PIL import Image
from sentence_transformers import SentenceTransformer

# from sklearn.decomposition import PCA
# from sklearn.preprocessing import normalize
# from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

# import clip
# import torch


# from random import shuffle


MODEL_NAME = "msmarco-distilbert-base-tas-b"
NUM_CLUSTERS = 1

# Default values - will be overridden by config
CENTROIDS_FILE = None
PCA_COMPONENTS_FILE = None
QUERY_FILE = None
CLUSTER_FILE_LOCATION = None
URL_BUNDLE_BASE_DIR = None
IS_TEXT = True
RUN_PCA = True
RUN_URL_FILTER = True
URL_FILTER_BY_CLUSTER = False
RUN_MSMARCO_DEV_QUERIES = True
FILTER_BADWORDS = False
INDEX_FILE = None
BADWORDS_FILE = None
SHORT_EXP = False
IMG_RESULTS_DIR = None

lock = threading.Lock()


def load_config(config_file):
    """Load configuration from a JSON file."""
    with open(config_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Set default values for optional fields
    config = {
        "PCA_COMPONENTS_FILE": data.get("pca_components_file"),
        "QUERY_FILE": data.get("query_file"),
        "CLUSTER_FILE_LOCATION": data.get("cluster_file_location"),
        "URL_BUNDLE_BASE_DIR": data.get("url_bundle_base_dir", ""),
        "IS_TEXT": data.get("is_text", True),
        "RUN_PCA": data.get("run_pca", True),
        "RUN_URL_FILTER": data.get("run_url_filter", False),
        "URL_FILTER_BY_CLUSTER": data.get("url_filter_by_cluster", False),
        "RUN_MSMARCO_DEV_QUERIES": data.get("run_msmarco_dev_queries", True),
        "FILTER_BADWORDS": data.get("filter_badwords", False),
        "INDEX_FILE": data.get("index_file"),
        "SHORT_EXP": data.get("short_exp", False),
        "BADWORDS_FILE": data.get("badwords_file", None),
        "IMG_RESULTS_DIR": data.get("img_results_dir", "/tmp/img_res"),
    }

    # Create image results directory if it doesn't exist
    if config["IMG_RESULTS_DIR"]:
        os.makedirs(config["IMG_RESULTS_DIR"], exist_ok=True)

    # Load FAISS index
    if config["INDEX_FILE"] and os.path.exists(config["INDEX_FILE"]):
        index = faiss.read_index(config["INDEX_FILE"])
        config["index"] = index
    else:
        print(f"Warning: Index file not found: {config['INDEX_FILE']}", file=sys.stderr)
        config["index"] = None

    return config


def find_nearest_clusters_from_faiss(query_embed, config):
    """Find the nearest clusters using a FAISS index."""
    query_float = numpy.array(query_embed).astype("float32")

    if config["index"] is None:
        print("Error: No FAISS index loaded", file=sys.stderr)
        return [0]  # Default to cluster 0

    _, results = config["index"].search(query_float, 1)
    return results[0]


def embed(query):
    """Embed a query using the appropriate model."""
    return SentenceTransformer(MODEL_NAME).encode(query)


def find_nearest_clusters_from_file(query_embed, config):
    """Find the nearest clusters from a file."""
    if not config.get("CENTROIDS_FILE") or not os.path.exists(config["CENTROIDS_FILE"]):
        print(
            f"Error: Centroids file not found: {config.get('CENTROIDS_FILE')}",
            file=sys.stderr,
        )
        return [0]

    query_float = numpy.array(query_embed).astype("float32")
    centroids = numpy.loadtxt(config["CENTROIDS_FILE"])
    centroids = numpy.round((centroids) * (1 << 5))

    distances = numpy.asarray(
        numpy.matmul(centroids, numpy.transpose(numpy.asmatrix(query_float)))
    )
    res = numpy.argpartition(distances, -NUM_CLUSTERS, axis=0)
    res = sorted(res[-NUM_CLUSTERS:], key=lambda i: distances[i], reverse=True)
    topk = res[-NUM_CLUSTERS:]
    return topk


def find_nearest_clusters(cluster_index, query_embed, num_clusters):
    """Find the nearest clusters."""
    query_float = numpy.array(query_embed).astype("float32")
    results = cluster_index.search(query_float, num_clusters)
    return results[1][0]


def get_results_url_chunks(top_res, query_embed, cluster, num_results, config):
    """Get the URL chunks for the top results."""
    cluster_file_name = f"{config['CLUSTER_FILE_LOCATION']}/cluster_{cluster}.txt"

    if not os.path.exists(cluster_file_name):
        print(f"Warning: Cluster file not found: {cluster_file_name}", file=sys.stderr)
        return []

    with open(cluster_file_name, "r", encoding="utf-8") as f:
        lines = [line for line in f.readlines() if line.strip()]
    if len(lines) == 0:
        return []
    chunk = []
    matches = False
    done = False
    for line in lines:
        if done:
            break
        if "------------" in line:
            if not matches:
                chunk = list()
            else:
                done = True
        else:
            if line.split(" | ")[2].strip() == top_res.strip():
                matches = True
            chunk.append(line)
    return find_best_docs_from_lines(chunk, query_embed, num_results)


def filter_results_by_url_bundle(top_res, query_embed, cluster, num_results, config):
    """Filter results by url bundle"""
    bundle_dir = f"{config['URL_BUNDLE_BASE_DIR']}/{cluster}/clusters"
    print(f"{bundle_dir}", file=sys.stderr)

    if not os.path.exists(bundle_dir):
        print(f"Warning: Bundle directory not found: {bundle_dir}", file=sys.stderr)
        return []

    bundle_files = glob.glob(f"{bundle_dir}/*")
    for bundle_file in bundle_files:
        with open(bundle_file, "r", encoding="utf-8") as f:
            print(
                f"Checking {bundle_file} for {top_res['url'].strip()}",
                file=sys.stderr,
            )
            lines = [line for line in f.readlines() if line.strip()]
            match = False
            for line in lines:
                if line.split(" | ")[2].strip() == top_res["url"].strip():
                    match = True
                    print("Found MATCH", file=sys.stderr)
            if match:
                pool = multiprocessing.pool.ThreadPool()
                out = pool.map(lambda x: line_to_dist(x, query_embed), lines)
                out = list(zip(*out))
                urls = out[0]
                dists = out[1]
                print(f"[{bundle_file}] Parsed", file=sys.stderr)

                res_ids = find_nearest_docs(dists, num_results)
                print(f"[{bundle_file}] Found nearest", file=sys.stderr)

                ret = [{"score": dists[rid], "url": urls[rid]} for rid in res_ids]
                return ret

    print("ERROR: NO MATCH", file=sys.stderr)
    return []


def find_nearest_docs(dists, how_many):
    """Find the nearest documents based on distance."""
    res = None
    length = 1 if numpy.isscalar(dists) else len(dists)
    if length <= how_many:
        return range(length)
    # Get indexes of top-k
    res = numpy.argpartition(dists, -how_many, axis=0)

    res = sorted(res[-how_many:], key=lambda i: dists[i], reverse=True)
    topk = res[-how_many:]
    return topk


def line_to_dist(line, query_embed):
    """Line to distance."""
    (_, rest) = line.split(" | ", 1)
    parts = rest.split(",", len(query_embed) - 1)
    embed_vec = parts[0 : len(query_embed) - 1]
    (last, url) = parts[len(query_embed) - 1].split(" | ", 1)
    embed_vec.append(last)
    vec = [float(i) for i in embed_vec]
    if RUN_MSMARCO_DEV_QUERIES and not RUN_PCA:
        vec = numpy.clip(numpy.round(numpy.array(vec) * (1 << 5)), -16, 15)

    return (url, numpy.inner(vec, query_embed))


def find_best_docs_from_lines(lines, query_embed, num_results):
    """Find best docs from the lines"""
    pool = multiprocessing.pool.ThreadPool()
    out = pool.map(lambda x: line_to_dist(x, query_embed), lines)
    out = list(zip(*out))
    urls = out[0]
    dists = out[1]

    res_ids = find_nearest_docs(dists, num_results)

    return list(map(lambda rid: {"score": dists[rid], "url": urls[rid]}, res_ids))


def find_best_docs(cluster_file_name, query_embed, num_results):
    """Find the best documents from a cluster file."""
    if not os.path.exists(cluster_file_name):
        print(f"Warning: Cluster file not found: {cluster_file_name}", file=sys.stderr)
        return []

    print(f"[{cluster_file_name}] Starting read", file=sys.stderr)
    with open(cluster_file_name, "r", encoding="utf-8") as f:
        lines = [
            line
            for line in f.readlines()
            if (line.strip() and "--------------" not in line)
        ]
    if len(lines) == 0:
        return []
    print(f"[{cluster_file_name}] Have lines", file=sys.stderr)
    return find_best_docs_from_lines(lines, query_embed, num_results)


def find_one(results, cluster_id, query_embed, num_results, config):
    """Find one cluster and its best documents."""
    cluster_file_name = f"{config['CLUSTER_FILE_LOCATION']}/cluster_{cluster_id}.txt"
    print(f"Going to find best docs for {cluster_file_name}", file=sys.stderr)
    docs = find_best_docs(cluster_file_name, query_embed, num_results)
    print(docs, file=sys.stderr)

    lock.acquire()
    results += docs
    results.sort(key=lambda x: -int(x["score"]))
    lock.release()


def search(query, num_results, config):
    """Run search"""
    query_embed = embed(query)
    # Reduce precision to 5 bits
    query_embed = numpy.round(numpy.array(query_embed) * (1 << 5))
    print("\tHave embedding", file=sys.stderr)

    query_embed_pca = query_embed
    if config["RUN_PCA"] and config["PCA_COMPONENTS_FILE"]:
        if os.path.exists(config["PCA_COMPONENTS_FILE"]):
            pca_components = numpy.load(config["PCA_COMPONENTS_FILE"])
            if config["IS_TEXT"]:
                query_embed_pca = numpy.clip(
                    numpy.round(numpy.matmul(query_embed, pca_components) / 10), -16, 15
                )
            else:
                query_embed_pca = numpy.clip(
                    numpy.round(numpy.matmul(query_embed, pca_components)), -16, 15
                )
        else:
            print(
                f"Warning: PCA components file not found: {config['PCA_COMPONENTS_FILE']}",
                file=sys.stderr,
            )
    else:
        query_embed_pca = numpy.clip(query_embed_pca, -16, 15)

    res = []
    clusters = find_nearest_clusters_from_faiss([query_embed], config)
    print(f"\tNearest clusters: {clusters}", file=sys.stderr)
    for _, cluster_id in enumerate(clusters):
        find_one(res, cluster_id, query_embed_pca, num_results, config)
        if config["RUN_URL_FILTER"] and len(res) > 0:
            res = filter_results_by_url_bundle(
                res[0], query_embed_pca, cluster_id, num_results, config
            )
        if config["URL_FILTER_BY_CLUSTER"]:
            print("filter by cluster", file=sys.stderr)
            res = get_results_url_chunks(
                res[0]["url"], query_embed_pca, cluster_id, num_results, config
            )
    return res


def latex_format_queries(query_list, badwords, config):
    """Process queries in latex format"""
    for qid, query in enumerate(query_list[:100]):
        results = search(query, 20, config)[0:10]

        done = False
        for _, r in enumerate(results):
            if done:
                break
            safe = True
            url = r["url"].strip()
            for badword in badwords:
                if badword in url:
                    safe = False
            if safe:
                try:
                    img_data = requests.get(
                        url.split(".jpg", 1)[0] + ".jpg", timeout=5
                    ).content

                    # Use configurable image results directory
                    img_file = f"{config['IMG_RESULTS_DIR']}/result_{qid}.jpg"
                    with open(img_file, "wb") as handler:
                        handler.write(img_data)
                    print(f"\\QueryRes{{{query}}}{{fig/img_results/result_{qid}.jpg}}")
                    print("")
                    sys.stdout.flush()
                    done = True
                except requests.exceptions.RequestException:
                    print("Trying next...", file=sys.stderr)
        sys.stdout.flush()


def main():
    """Main entry point"""
    config_file = sys.argv[1]
    config = load_config(config_file)

    if config["RUN_MSMARCO_DEV_QUERIES"]:
        if not config["QUERY_FILE"] or not os.path.exists(config["QUERY_FILE"]):
            print(
                f"Error: Query file not found: {config.get('QUERY_FILE')}",
                file=sys.stderr,
            )
            return

        lines = open(config["QUERY_FILE"], encoding="utf-8").read().splitlines()
        query_data = [line.split("\t") for line in lines]
        query_list = [elem[1] for elem in query_data]

        qid_dict = dict()
        for elem in query_data:
            qid_dict[elem[1]] = int(elem[0])

        # Load badwords if enabled and file exists
        badwords = set()
        if config["FILTER_BADWORDS"] and config["BADWORDS_FILE"]:
            if os.path.exists(config["BADWORDS_FILE"]):
                with open(config["BADWORDS_FILE"], "r", encoding="utf-8") as f:
                    badwords = set([line.strip() for line in f.readlines()])
            else:
                print(
                    f"Warning: Badwords file not found: {config['BADWORDS_FILE']}",
                    file=sys.stderr,
                )

        if config["SHORT_EXP"]:
            query_list = query_list[:500]
        for query in query_list:
            print(f"Query: {qid_dict[query]}\n")
            results = search(query, 100, config)[0:100]
            for _, r in enumerate(results):
                safe = True
                url = r["url"].strip()
                url_lower = url.lower()
                if config["FILTER_BADWORDS"]:
                    for badword in badwords:
                        if badword in url_lower:
                            safe = False
                if safe:
                    print(f"{r['score']} {r['url'].strip()}")
                else:
                    print("[REDACTED]")
            print("---------------\n")
            sys.stdout.flush()
    else:
        query_list = ["chocolate chip cookie"]

        for query in query_list:
            print(f"Query: {query}\n")
            results = search(query, 20, config)
            print(results, file=sys.stderr)
            for _, r in enumerate(results):
                print(f"{r['score']} {r['url'].strip()}")
            print("\n----------")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
