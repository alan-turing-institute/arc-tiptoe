"""
This module provides utility functions for converting data between different formats.
"""

import glob
import os
import pickle
import shutil

import numpy as np


def convert_npy_to_txt():
    """Convert .npy files to text format expected by Go code."""

    # Create target directories
    os.makedirs("/home/azureuser/data/embeddings/clusters", exist_ok=True)
    os.makedirs("/home/azureuser/data/urls/clusters", exist_ok=True)

    # 1. Convert PCA components
    if os.path.exists("dim_reduce/dim_reduced/pca_192.npy"):
        pca_components = np.load(
            "dim_reduce/dim_reduced/pca_192.npy", allow_pickle=False
        )
        np.savetxt(
            "/home/azureuser/data/embeddings/pca_components_192.txt",
            pca_components,
            fmt="%.6f",
        )
        print(f"Converted PCA components: {pca_components.shape}")

    # 2. Convert centroids - handle both .npy and text formats
    centroids_file = "clustering/centroids/msmarco_centroids.npy"
    if os.path.exists(centroids_file):
        try:
            # Try loading as .npy first
            centroids = np.load(centroids_file, allow_pickle=False)
            print(f"Loaded centroids as .npy: {centroids.shape}")
        except (ValueError, pickle.UnpicklingError):
            try:
                # If that fails, try as text file
                centroids = np.loadtxt(centroids_file)
                print(f"Loaded centroids as text file: {centroids.shape}")
            except Exception as e:
                print(f"Error loading centroids: {e}")
                print("Checking file format...")
                with open(centroids_file, "r") as f:
                    first_lines = f.readlines()[:3]
                    for i, line in enumerate(first_lines):
                        print(f"Line {i+1}: {line.strip()[:100]}...")
                return

        # Save as text
        np.savetxt(
            "/home/azureuser/data/embeddings/centroids.txt", centroids, fmt="%.6f"
        )
        print(f"Converted centroids: {centroids.shape}")


def convert_cluster_files():
    """Convert cluster assignment files to the format expected by Go."""

    # Copy embedding cluster files
    source_dir = "clustering/assignments/"
    target_dir = "/home/azureuser/data/embeddings/clusters/"

    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} not found")
        # Try alternative location
        source_dir = "clustering/dim_red_assignments/pca_192/"
        if not os.path.exists(source_dir):
            print(f"Alternative source directory {source_dir} also not found")
            return

    cluster_files = glob.glob(f"{source_dir}*.txt")
    print(f"Found {len(cluster_files)} cluster files in {source_dir}")

    for source_file in cluster_files:
        # Extract cluster number from filename
        filename = os.path.basename(source_file)
        if "cluster_" in filename:
            cluster_num = filename.split("cluster_")[-1].split(".")[0]
        elif "msmarco_cluster_" in filename:
            cluster_num = filename.split("msmarco_cluster_")[-1].split(".")[0]
        else:
            continue

        target_file = f"{target_dir}cluster_{cluster_num}.txt"
        shutil.copy2(source_file, target_file)
        print(f"Copied {filename} -> cluster_{cluster_num}.txt")

    print(f"Converted {len(cluster_files)} cluster files")


if __name__ == "__main__":
    convert_npy_to_txt()
    convert_cluster_files()
