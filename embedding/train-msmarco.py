import os
import pickle
import sys
import time

import jsonargparse
import numpy
import torch
import torch.backends
import torch.backends.mps
from config import *

# from datasets import Dataset, load_dataset, load_from_disk
from sentence_transformers import SentenceTransformer

NTHREADS = 1
# NTHREADS = max(1, multiprocessing.cpu_count() - 1)
DATA_FILE = "data/ms_marco_docs.tsv"


def save(entries, outfile):
    """Save the entries to a file.

    Args:
        entries: entries to save
        outfile: output file path
    """
    with open(outfile, "wb") as f:
        pickle.dump(entries, f)


def compute_embeddings(model):
    model = SentenceTransformer(model)
    model.max_seq_length = SEQ_LEN
    lines = open(DATA_FILE).read().splitlines()
    print("Read all lines")
    sys.stdout.flush()
    new_data = [line.split("\t") for line in lines]
    print("Split all lines by tabs")
    sys.stdout.flush()
    print([elem[3] for elem in new_data])
    chunked_data = [" ".join(elem[1].split()[0:SEQ_LEN]) for elem in new_data]
    embeddings = numpy.array(
        model.encode(chunked_data, batch_size=32, convert_to_numpy=True)
    )
    print("Encoded all")
    sys.stdout.flush()
    docids = [elem[0] for elem in new_data]

    return (embeddings, docids)


# def chunked_data(data, seq_len=SEQ_LEN):
#     """Chunk the data into sequences of length seq_len.

#     Args:
#         data: list of strings to chunk
#         seq_len: maximum sequence length

#     Returns:
#         List of chunked strings.
#     """
#     data["body"] = " ".join(data["body"].split()[0:seq_len])
#     return data


# def compute_embeddings_hf(model: str, dataset: str) -> tuple[numpy.ndarray, list[str]]:
#     """Compute embeddings using Hugging Face datasets.

#     Args:
#         model: model name to use for embeddings

#     Returns:
#         Tuple of embeddings and document IDs.
#     """
#     device = "mps" if torch.backends.mps.is_available() else "cpu"
#     print(f"Using device: {device}")
#     sys.stdout.flush()
#     model = SentenceTransformer(model, device=device)
#     model.max_seq_length = SEQ_LEN
#     dataset = load_dataset(dataset, "docs")
#     dataset = dataset.map(
#         chunked_data, batched=True, batch_size=1000, args={"seq_len": SEQ_LEN}
#     )
#     print("Loaded and preprocessed dataset")
#     sys.stdout.flush()
#     embeddings = numpy.array(
#         model.encode(dataset, batch_size=32, convert_to_numpy=True)
#     )
#     print("Encoded all")
#     sys.stdout.flush()

#     docids = dataset["doc_id"]

#     return (embeddings, docids)


def process_embeddings(embeddings, docids, out_docids, out_embeddings):
    numpy.save(out_docids, numpy.array(docids))

    embeddings_mat = numpy.asmatrix(embeddings)
    numpy.save(out_embeddings, embeddings_mat)


def main(idx: int, file_prefix: str):
    # if len(sys.argv) != 2:
    #     raise ValueError("Usage: python %s idx file-prefix\n" % sys.argv[0])

    prefix = file_prefix + str(idx) + "_"
    model = "msmarco-distilbert-base-tas-b"
    embeddings, docids = compute_embeddings(model)
    docids_file = ("%s_url.npy") % (prefix)
    embedding_file = ("%s_embeddings.npy") % (prefix)
    process_embeddings(embeddings, docids, docids_file, embedding_file)
    print(("Output to %s and %s") % (docids_file, embedding_file))


if __name__ == "__main__":
    jsonargparse.auto_cli(main)
