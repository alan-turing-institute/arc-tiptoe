import argparse
import glob

import numpy as np

from arc_tiptoe.preprocessing.utils.utils import parse_file


def get_orig_embeddings(data_dir):
    orig_fpath = f"{data_dir}/embedding/embeddings/embeddings_original.npy"
    return np.load(orig_fpath)


def get_cluster_fpaths(data_dir):
    return glob.glob(f"{data_dir}/clusters/cluster_*.txt")


def load_quantized_cluster(cluster_fpath: str) -> tuple[np.ndarray, np.ndarray]:
    lines = parse_file(cluster_fpath)
    indices, quantized_embs_str, _ = zip(*lines, strict=False)
    quantized_embs = [list(map(int, emb.split(","))) for emb in quantized_embs_str]
    return np.array(indices, dtype=np.int64), np.array(quantized_embs, dtype=np.int8)


def construct_non_quantized_cluster(
    original_embs: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return indices, original_embs[indices]


def main(data_dir: str):
    orig_embs = get_orig_embeddings(data_dir)
    print(f"Shape of original embeddings: {orig_embs.shape}")

    cluster_fpaths = get_cluster_fpaths(data_dir)

    quantized_clusters = [load_quantized_cluster(fpath) for fpath in cluster_fpaths]

    nonquantized_clusters = [
        construct_non_quantized_cluster(orig_embs, indices)
        for indices, _ in quantized_clusters
    ]

    print("now we need to do some searching")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_dir", type=str, required=True)
    args = argparser.parse_args()
    main(args.data_dir)


#  testing with
# --data_dir ./data/c1970a4c-46c4-8fbe-cf94-d099e24ba206-2000-192
