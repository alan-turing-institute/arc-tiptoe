import argparse
import glob
import os
from dataclasses import dataclass

import numpy as np

from arc_tiptoe.preprocessing.utils.utils import parse_file, write_cluster_file


@dataclass
class Cluster:
    indices: np.ndarray
    embeddings: np.ndarray
    urls: list[str]
    _centroid: np.ndarray | None = None

    def centroid(self) -> np.ndarray:
        if self._centroid is None:
            self._centroid = np.mean(self.embeddings, axis=0)
        return self._centroid


def get_orig_embeddings(data_dir):
    orig_fpath = f"{data_dir}/embedding/embeddings/embeddings_original.npy"
    return np.load(orig_fpath)


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


def write_cluster(cluster_file_path: str, cluster: Cluster):
    ind_strings = [str(idx) for idx in cluster.indices]
    emb_strings = [",".join(map(str, emb.tolist())) for emb in cluster.embeddings]

    contents = zip(
        *[ind_strings, emb_strings, cluster.urls],
        strict=True,
    )
    write_cluster_file(cluster_file_path, contents)


def construct_non_quantized_cluster(
    original_embs: np.ndarray, quantized_cluster: Cluster
) -> Cluster:
    return Cluster(
        indices=quantized_cluster.indices,
        embeddings=original_embs[quantized_cluster.indices],
        urls=quantized_cluster.urls,
    )


def create_nonquantized_clusters(data_dir: str):
    nq_cluster_dir = f"{data_dir}/non_quantized_clusters"
    print(f"Saving non-quantized clusters to {nq_cluster_dir}")
    os.makedirs(nq_cluster_dir, exist_ok=False)

    orig_embs = get_orig_embeddings(data_dir)
    print(f"Shape of original embeddings: {orig_embs.shape}")

    cluster_fpaths = get_cluster_fpaths(data_dir)

    quantized_clusters = [load_quantized_cluster(fpath) for fpath in cluster_fpaths]

    nonquantized_clusters = [
        construct_non_quantized_cluster(orig_embs, qcluster)
        for qcluster in quantized_clusters
    ]

    for i, cluster in enumerate(nonquantized_clusters):
        out_fpath = f"{nq_cluster_dir}/cluster_{i}.txt"
        with open(out_fpath, "w") as f:
            write_cluster(out_fpath, cluster)

    # write centroids file
    centroids_fpath = f"{nq_cluster_dir}/centroids.txt"
    with open(centroids_fpath, "w") as f:
        for cluster in nonquantized_clusters:
            centroid_str = ", ".join(map(str, cluster.centroid().tolist()))
            f.write(centroid_str)

    print("Complete.")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_dir", type=str, required=True)
    args = argparser.parse_args()
    create_nonquantized_clusters(args.data_dir)


#  testing with
# --data_dir ./data/c1970a4c-46c4-8fbe-cf94-d099e24ba206-2000-192
