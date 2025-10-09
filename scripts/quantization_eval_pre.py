import argparse
import csv
import glob
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from arc_tiptoe.preprocessing.embedding.tt_models import (
    distilbert_preprocess,
    load_sentence_transformer,
)
from arc_tiptoe.preprocessing.utils.quantization import quantize_query_embedding
from arc_tiptoe.preprocessing.utils.utils import parse_file


@dataclass
class Cluster:
    indices: np.ndarray
    embeddings: np.ndarray
    urls: list[str]
    _centroid: np.ndarray | None = None

    def centroid(self) -> np.ndarray:
        if self._centroid is None:
            self._centroid = np.mean(
                self.embeddings, axis=0, dtype=self.embeddings.dtype
            )
        return self._centroid


@dataclass
class ClusterResult:
    score: float
    url: str


@dataclass
class Result:
    query_id: str
    cluster_results: list[list[ClusterResult]]


@dataclass
class Results:
    results: list[Result]

    def to_csv(self, fpath: str) -> None:
        max_clusters = max(len(r.cluster_results) for r in self.results)

        with open(fpath, "w", newline="") as csvfile:
            fieldnames = (
                ["query_id"]
                + [f"cluster_{i}_res" for i in range(max_clusters)]
                + [f"cluster_{i}_total_comm" for i in range(max_clusters)]
            )
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in self.results:
                row = {"query_id": result.query_id}
                for i, cluster in enumerate(result.cluster_results):
                    row[f"cluster_{i}_res"] = str(
                        [{"score": cr.score, "url": cr.url} for cr in cluster]
                    )
                    row[f"cluster_{i}_total_comm"] = "-1"
                writer.writerow(row)


def get_orig_embeddings(data_dir):
    orig_fpath = f"{data_dir}/embedding/embeddings/embeddings_original.npy"
    return np.load(orig_fpath)


def load_query_embs(queries_fpath):
    return np.load(queries_fpath)


def get_cluster_fpaths(data_dir):
    return glob.glob(f"{data_dir}/clusters/cluster_*.txt")


def load_quantized_cluster(
    cluster_fpath: str,
) -> Cluster:
    lines = parse_file(cluster_fpath)
    indices, quantized_embs_str, urls = zip(*lines, strict=True)
    quantized_embs = [list(map(int, emb.split(","))) for emb in quantized_embs_str]
    return Cluster(
        indices=np.array(indices, dtype=np.int64),
        embeddings=np.array(quantized_embs, dtype=np.int8),
        urls=list(urls),
    )


def construct_non_quantized_cluster(
    original_embs: np.ndarray, quantized_cluster: Cluster
) -> Cluster:
    return Cluster(
        indices=quantized_cluster.indices,
        embeddings=original_embs[quantized_cluster.indices],
        urls=quantized_cluster.urls,
    )


def nearest_docs(
    embedding: np.ndarray, doc_embeddings: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:  # indices, distances
    distances = np.linalg.norm(doc_embeddings - embedding, axis=1)
    inds = np.argsort(distances)[:n]
    return inds, distances[inds]


def nearest_clusters(
    embedding: np.ndarray, clusters: list[Cluster], n: int
) -> tuple[np.ndarray, np.ndarray]:
    centroids = np.array([cluster.centroid() for cluster in clusters])
    return nearest_docs(embedding, centroids, n)


def search_for_query(
    query_id: str,
    query_embedding: np.ndarray,
    clusters: list[Cluster],
    top_k_clusters: int = 5,
    top_k_docs: int = 10,
) -> Result:
    nearest_cluster_indices, _ = nearest_clusters(
        query_embedding, clusters, top_k_clusters
    )

    clusters_results: list[list[ClusterResult]] = []
    for cluster_idx in nearest_cluster_indices:
        cluster = clusters[cluster_idx]
        inds, distances = nearest_docs(query_embedding, cluster.embeddings, top_k_docs)

        cluster_results: list[ClusterResult] = [
            ClusterResult(score=dist, url=cluster.urls[ind])
            for dist, ind in zip(distances, inds, strict=True)
        ]

        clusters_results.append(cluster_results)

    return Result(query_id=query_id, cluster_results=clusters_results)


def search(query_embeddings, clusters, top_k_clusters=5, top_k_docs=10) -> Results:
    return Results(
        results=[
            search_for_query(
                query_id=str(i),
                query_embedding=query_embedding,
                clusters=clusters,
                top_k_clusters=top_k_clusters,
                top_k_docs=top_k_docs,
            )
            for i, query_embedding in enumerate(query_embeddings)
        ]
    )


def embed_queries(queries: list[str]) -> np.ndarray:
    model = SentenceTransformer(
        "sentence-transformers/msmarco-distilbert-base-tas-b", "cpu"
    )
    qembs = model.encode(
        queries,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.array(qembs)


def main(data_dir: str, queries: np.ndarray):
    orig_embs = get_orig_embeddings(data_dir)
    print(f"Shape of original embeddings: {orig_embs.shape}")

    quantized_queries = np.array(
        [quantize_query_embedding(x) for x in queries], dtype=np.int8
    )
    print(f"Shape of quantized queries: {quantized_queries.shape}")

    cluster_fpaths = get_cluster_fpaths(data_dir)

    quantized_clusters = [load_quantized_cluster(fpath) for fpath in cluster_fpaths]

    nonquantized_clusters = [
        construct_non_quantized_cluster(orig_embs, qcluster)
        for qcluster in quantized_clusters
    ]

    nq_search_results = search(queries, nonquantized_clusters)
    print(f"Search results for {len(queries)} queries")
    nq_results_fpath = f"{data_dir}/non_quantized_search_results.csv"
    nq_search_results.to_csv(nq_results_fpath)

    q_search_results = search(quantized_queries, quantized_clusters)
    print(f"Search results for {len(queries)} queries")
    q_results_fpath = f"{data_dir}/quantized_search_results.csv"
    q_search_results.to_csv(q_results_fpath)

    print(f"saved nq and q results to {nq_results_fpath} and {q_results_fpath}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_dir", type=str, required=True)

    # either need the --query_embs_fpath or queries_text_fpath
    # if the latter is provided we will embed using msmarco-distilbert-base-tas-b

    argparser.add_argument(
        "--query_text_fpath",
        type=str,
        required=False,
        help="Path to file containing query texts",
    )
    argparser.add_argument(
        "--query_embs_fpath",
        type=str,
        required=False,
        help="Path to npy file containing query embeddings",
    )

    args = argparser.parse_args()

    if args.query_text_fpath is not None:
        if not args.query_text_fpath.endswith(".tsv"):
            raise ValueError("--queries_text_fpath must be a .tsv file")

        with open(args.query_text_fpath, "r") as f:
            text = [l.split("\t")[1].strip() for l in f.readlines()]

        queries = embed_queries(text)
    elif args.query_embs_fpath is not None:
        queries = load_query_embs(args.query_embs_fpath)
    else:
        raise ValueError("Must provide either --query_text_fpath or --query_embs_fpath")

    main(args.data_dir, queries=queries)


#  testing with
# --data_dir ./data/c1970a4c-46c4-8fbe-cf94-d099e24ba206-2000-192 --query_text_fpath ./test_data/test_queries.tsv
