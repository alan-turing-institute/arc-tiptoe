# # import concurrent
# import glob
# import concurrent.futures
import logging
import math
import os
import zlib

# import sys
import faiss
import numpy as np
from tqdm import tqdm

# DIM=256
DIM = 768
AVG_BUNDLE_SIZE = 4000
MAX_SIZE = 4000
URLS_PER_BUNDLE = 160
CLUSTER_DIR = "clustering/assignments"
BASE_DIR = "clustering/"
NUM_CLUSTERS = 4 * 32 * 10


def parse_file(filename):
    """Parse a cluster file and extract relevant information.

    Args:
        filename: The path to the cluster file.

    Returns:
        A list of parsed contents from the cluster file.
    """
    contents = []
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc=f"Parsing {filename}"):
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


def create_bundles(contents, logger=None):
    """Create bundles of URLs based on clustering."""

    # If logger is provided, use it; otherwise, print to console
    if logger:
        logger.info("Creating bundles")
    else:
        print("Creating bundles")
    # Calculate the number of bundles based on the size of contents
    num_bundles = int(math.ceil(float(get_size(contents)) / float(AVG_BUNDLE_SIZE)))
    if logger:
        logger.info("Num_bundles = %d" % num_bundles)
    else:
        print(f"Num_bundles = {num_bundles}")

    # Embed contents and perform clustering
    if logger:
        logger.info("Embedding contents")
    else:
        print("Embedding contents")
    embed_contents = [elem[1] for elem in contents]
    data = np.loadtxt(embed_contents, delimiter=",")
    kmeans = faiss.Kmeans(DIM, num_bundles, nredo=3)
    if len(data) >= 1 and len(np.shape(data)) == 2:
        kmeans.train(data.astype(np.float32))
        centroids = kmeans.centroids
        _, assignments = kmeans.index.search(data.astype(np.float32), 1)
    else:
        centroids = np.zeros((1, DIM))
        assignments = [[0]]

    # Create a dictionary to hold the assignments
    if logger:
        logger.info("Creating assignment dictionary")
    else:
        print("Creating assignment dictionary")
    assignment_dict = {}
    for i, cluster_pair in tqdm(
        enumerate(assignments), desc="Creating assignment dictionary"
    ):
        cluster = cluster_pair[0]
        if cluster not in assignment_dict:
            assignment_dict[cluster] = [contents[i]]
        else:
            assignment_dict[cluster].append(contents[i])

    # Divide arbitrarily when all documents are the same
    if len(assignment_dict) == 1 and num_bundles > 1:
        if logger:
            logger.info("All documents are the same, dividing arbitrarily")
        else:
            print("All documents are the same, dividing arbitrarily")
        for i in tqdm(
            range(int(math.ceil(float(len(contents)) / float(URLS_PER_BUNDLE)))),
            desc="Dividing arbitrarily",
        ):
            centroids[i] = centroids[list(assignment_dict.keys())[0]]
            upper_bound = min((i + 1) * URLS_PER_BUNDLE, len(contents))
            assignment_dict[i] = [
                contents[j] for j in range(i * URLS_PER_BUNDLE, upper_bound)
            ]
            return (centroids, assignment_dict)

    # If documents are not same, proceed with further processing
    if logger:
        logger.info("Documents are not the same, proceeding with further processing")
    else:
        print("Documents are not the same, proceeding with further processing")
    init_clusters = list(assignment_dict.keys()).copy()
    for cluster in tqdm(init_clusters, desc="Processing clusters"):
        # causing key error?
        if get_size(assignment_dict[cluster]) > MAX_SIZE:
            if logger:
                logger.info("Cluster %d exceeds max size, creating bundles" % cluster)
            else:
                print(f"Cluster {cluster} exceeds max size, creating bundles")
            (sub_centroids, sub_assignment_dict) = create_bundles(
                assignment_dict[cluster]
            )
            replace_idx = sorted(list(sub_assignment_dict.keys()))[0]
            centroids[cluster] = sub_centroids[replace_idx]
            assignment_dict[cluster] = sub_assignment_dict[replace_idx]
            offset = len(centroids)
            centroids = np.vstack((centroids, sub_centroids[replace_idx + 1 :]))
            # centroids = centroids + sub_centroids[1:]
            # Note: can have some centroids for empty clusters here
            for sub_cluster in sub_assignment_dict:
                if sub_cluster != replace_idx:
                    assignment_dict[sub_cluster + offset] = sub_assignment_dict[
                        sub_cluster
                    ]

    return (centroids, assignment_dict)


def process_cluster(cluster_file, cluster_idx, logger=None):
    """Process a single cluster file to create bundles and centroids."""
    if logger:
        logger.info("**** PROCESS CLUSTER *****")
    contents = parse_file(cluster_file)
    if logger:
        logger.info("LEN = %d", len(contents))
    if len(contents) == 0:
        return
    centroids, assignment_dict = create_bundles(contents, logger=logger)
    if logger:
        logger.info("Writing to file -- %s/%d", BASE_DIR, cluster_idx)
    if not os.path.exists(("%s/%d/") % (BASE_DIR, cluster_idx)):
        os.mkdir("%s/%d/" % (BASE_DIR, cluster_idx))
    if not os.path.exists(("%s/%d/clusters") % (BASE_DIR, cluster_idx)):
        os.mkdir("%s/%d/clusters/" % (BASE_DIR, cluster_idx))
    np.savetxt("%s/%d/centroids.npy" % (BASE_DIR, cluster_idx), centroids)

    for bundle in tqdm(assignment_dict, desc="Writing bundles"):
        with open(
            "%s/%d/clusters/bundle_%d.txt" % (BASE_DIR, cluster_idx, bundle),
            "w",
            encoding="utf-8",
        ) as f:
            print(
                "Writing to %s/%d/clusters/bundle_%d.txt"
                % (BASE_DIR, cluster_idx, bundle)
            )
            for elem in assignment_dict[bundle]:
                if len(elem) == 3:
                    f.write("%s | %s | %s\n" % (elem[0], elem[1], elem[2]))
                else:
                    print("ERORR: elem != 3")
                    print(len(elem))
    print("Finished write")


def main():
    """Main function to process all cluster files."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.info("Starting cluster processing")
    logger.info("Base directory: %s", BASE_DIR)

    if not os.path.exists(BASE_DIR):
        os.mkdir(BASE_DIR)
    cluster_files = [
        f"{CLUSTER_DIR}/msmarco_cluster_{i}.txt" for i in range(NUM_CLUSTERS)
    ]
    # ctr = 0
    for i in range(len(cluster_files)):
        process_cluster(cluster_files[i], i)


if __name__ == "__main__":
    main()
