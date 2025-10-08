import argparse
import glob

import numpy as np

from arc_tiptoe.preprocessing.utils.utils import parse_file


def get_orig_embeddings(data_dir):
    orig_fpath = f"{data_dir}/embedding/embeddings/embeddings_original.npy"
    return np.load(orig_fpath)


def get_cluster_fpaths(data_dir):
    return glob.glob(f"{data_dir}/clusters/cluster_*.txt")


def main(data_dir: str):
    orig_embs = get_orig_embeddings(data_dir)
    print(f"Shape of original embeddings: {orig_embs.shape}")

    # TODO more efficient to do a pre-allocation
    all_lines: list[tuple[str]] = []

    cluster_fpaths = get_cluster_fpaths(data_dir)

    for cluster_fpath in cluster_fpaths:
        lines = parse_file(cluster_fpath)
        lines = [
            (*line, cluster_fpath) for line in lines
        ]  # convert to tuples for hashing and add fpath for debugging
        all_lines.extend(lines)

    all_lines = sorted(set(all_lines), key=lambda row: row[0])

    # TODO remove:
    print("example overlaps that are breaking this right now:")
    print(all_lines[2])
    print(all_lines[3])

    # TODO this fails at the moment
    assert len(all_lines) == orig_embs.shape[0], (
        f"Number of unique lines in clusters ({len(all_lines)}) does not match "
        f"number of original embeddings ({orig_embs.shape[0]})"
    )

    inds, quantized_embs, _ = zip(*all_lines, strict=True)

    print(f"Number of quantized embeddings: {len(quantized_embs)}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_dir", type=str, required=True)
    args = argparser.parse_args()
    main(args.data_dir)


#  testing with
# --data_dir ./data/c1970a4c-46c4-8fbe-cf94-d099e24ba206-2000-192
