"""
Modified search.py that supports configurable number of clusters to search
"""

import json
import os
import sys
from pathlib import Path

import faiss

# Import everything from the original search.py
sys.path.append(str(Path(__file__).parent))
from search import *


def find_nearest_clusters_from_faiss_multi(query_embeds, config):
    """Find nearest clusters with configurable count"""

    # Get number of clusters to search from config
    num_clusters_to_search = config.get("num_clusters_to_search", 1)

    print(f"🔍 Searching top {num_clusters_to_search} clusters", file=sys.stderr)

    index_file = config.get("INDEX_FILE") or config.get("index_file")
    if not index_file or not os.path.exists(index_file):
        print(f"❌ Index file not found: {index_file}", file=sys.stderr)
        return [0]  # Fallback to cluster 0

    try:
        cluster_index = faiss.read_index(index_file)

        # Search for top N clusters
        query_float = numpy.array(query_embeds[0]).astype("float32").reshape(1, -1)
        _, cluster_ids = cluster_index.search(query_float, num_clusters_to_search)

        # Return the cluster IDs
        valid_clusters = [int(cid) for cid in cluster_ids[0] if cid >= 0]

        print(f"🎯 Found clusters: {valid_clusters}", file=sys.stderr)
        return valid_clusters[:num_clusters_to_search]

    except Exception as e:
        print(f"❌ Error searching clusters: {e}", file=sys.stderr)
        return [0]  # Fallback


def search_multi_cluster(query, num_results, config):
    """Multi-cluster search function"""

    # Get embedding for query
    query_embed = embed(query)
    query_embed = numpy.round(numpy.array(query_embed) * (1 << 5))
    print("✅ Have embedding", file=sys.stderr)

    # Apply PCA if configured
    query_embed_pca = query_embed
    if config.get("run_pca", False) and config.get("pca_components_file"):
        pca_file = config["pca_components_file"]
        if os.path.exists(pca_file):
            try:
                pca_components = numpy.loadtxt(pca_file)
                if config.get("is_text", True):
                    query_embed_pca = numpy.clip(
                        numpy.round(numpy.matmul(query_embed, pca_components) / 10),
                        -16,
                        15,
                    )
                else:
                    query_embed_pca = numpy.clip(
                        numpy.round(numpy.matmul(query_embed, pca_components)), -16, 15
                    )
                print(f"✅ Applied PCA transformation", file=sys.stderr)
            except Exception as e:
                print(f"⚠️  PCA failed: {e}", file=sys.stderr)
                query_embed_pca = numpy.clip(query_embed_pca, -16, 15)
        else:
            query_embed_pca = numpy.clip(query_embed_pca, -16, 15)
    else:
        query_embed_pca = numpy.clip(query_embed_pca, -16, 15)

    # Find nearest clusters
    clusters = find_nearest_clusters_from_faiss_multi([query_embed], config)
    print(f"🔍 Searching {len(clusters)} clusters: {clusters}", file=sys.stderr)

    # Aggregate results from all clusters
    all_results = []

    for cluster_id in clusters:
        print(f"📂 Searching cluster {cluster_id}", file=sys.stderr)

        cluster_file_name = (
            f"{config['cluster_file_location']}/cluster_{cluster_id}.txt"
        )

        if os.path.exists(cluster_file_name):
            # Search this cluster
            cluster_results = find_best_docs(
                cluster_file_name, query_embed_pca, num_results
            )

            # Add cluster info to results
            for result in cluster_results:
                result["cluster_id"] = cluster_id

            all_results.extend(cluster_results)
            print(
                f"📊 Cluster {cluster_id}: {len(cluster_results)} results",
                file=sys.stderr,
            )
        else:
            print(f"⚠️  Cluster file not found: {cluster_file_name}", file=sys.stderr)

    # Sort all results by score and take top N
    all_results.sort(key=lambda x: -float(x.get("score", 0)))
    final_results = all_results[:num_results]

    print(
        f"🎯 Final results: {len(final_results)} from {len(clusters)} clusters",
        file=sys.stderr,
    )

    return final_results


def main():
    """Main function with multi-cluster support"""

    if len(sys.argv) < 2:
        print("Usage: python3 search_multi.py <config_file>")
        return

    config_file = sys.argv[1]

    # Load config
    with open(config_file, "r") as f:
        config = json.load(f)

    # Override global variables from config
    globals().update({k.upper(): v for k, v in config.items()})

    print(f"🚀 Multi-cluster search starting", file=sys.stderr)
    print(
        f"   Clusters to search: {config.get('num_clusters_to_search', 1)}",
        file=sys.stderr,
    )
    print(
        f"   Optimization: PCA={config.get('run_pca', False)}, URL={config.get('run_url_filter', False)}",
        file=sys.stderr,
    )

    if config.get("run_msmarco_dev_queries", True):
        query_file = config.get("query_file")
        if not query_file or not os.path.exists(query_file):
            print(f"❌ Query file not found: {query_file}", file=sys.stderr)
            return

        # Read queries
        with open(query_file, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        query_data = [line.split("\t") for line in lines if line.strip()]

        # Skip header if present
        if query_data and query_data[0][0] == "query_id":
            query_data = query_data[1:]

        print(f"📝 Processing {len(query_data)} queries", file=sys.stderr)

        for i, (qid, query_text) in enumerate(query_data):
            if config.get("short_exp", False) and i >= 10:
                break

            print(f"Query: {qid}")

            # Use multi-cluster search
            results = search_multi_cluster(query_text, 100, config)

            # Output results in expected format
            for result in results:
                score = result.get("score", 0)
                url = result.get("url", "").strip()
                cluster_id = result.get("cluster_id", -1)

                # Show which cluster each result came from (for debugging)
                print(f"{score} {url}")

            print("---------------\n")
            sys.stdout.flush()
    else:
        # Test with single query
        test_query = "chocolate chip cookie"
        print(f"Query: test")

        results = search_multi_cluster(test_query, 20, config)

        for result in results:
            score = result.get("score", 0)
            url = result.get("url", "").strip()
            cluster_id = result.get("cluster_id", -1)
            print(f"{score} {url} (cluster_{cluster_id})")

        print("---------------\n")


if __name__ == "__main__":
    main()
