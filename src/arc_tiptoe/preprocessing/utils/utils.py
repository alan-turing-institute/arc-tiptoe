"""
Utility functions for preprocessing that are used jointly across the different methods.
"""

import zlib

from tqdm import tqdm


def write_cluster_file(cluster_file_path: str, contents: list[list[str]]):
    with open(cluster_file_path, "w") as f:
        for line in contents:
            f.write(" | ".join(line) + "\n")


def parse_file(cluster_file_path: str) -> list[list[str]]:
    """Parse a cluster file and return its contents.

    The expected format is:

       ['doc_id | embedding (comma separated) | url']

    for each embedding in the cluster.

    Args:
        cluster_file_path (str): Path to the cluster file.

    Returns:
        list[list[str]]: Parsed contents of the cluster file.

    """
    contents = []
    with open(cluster_file_path, encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc=f"Parsing {cluster_file_path}"):
            if len(line) <= 1:
                continue
            contents.append(line.split(" | "))
    return contents


def get_size(contents):
    """Calculate the size of the contents."""
    out = zlib.compress(
        bytes(" ".join([content[2] for content in contents]), "utf-8"), level=9
    )
    return len(out)
