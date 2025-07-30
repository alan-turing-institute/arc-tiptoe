"""
Refactoring the MS MARCO embedding code to improve readability and maintainability,
including using the huggingface datasets library for data handling.
"""

import gc
import glob
import logging
import os

import ir_datasets
import numpy as np
import psutil
import torch
from config import SEQ_LEN
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_msmarco_dataset(max_docs: int = None) -> Dataset:
    """Load the MS MARCO dataset and convert it to a Hugging Face dataset."""
    dataset = ir_datasets.load("msmarco-document")

    def convert_to_hf_format():
        docs = []
        for doc in tqdm(dataset.docs_iter(), desc="Loading MS MARCO documents"):
            if max_docs and len(docs) >= max_docs:
                break
            docs.append(
                {
                    "doc_id": doc.doc_id,
                    "body": doc.body,
                    "title": doc.title,
                    "url": doc.url,
                }
            )
        return Dataset.from_list(docs)

    return convert_to_hf_format()


def chunk_text(text: str, seq_len: int = SEQ_LEN) -> str:
    """Chunk the text into smaller segments of a specified length."""
    # Fix: return a string, not a list
    if not text:
        return ""
    return " ".join(text.split()[:seq_len])


def extract_embeddings(
    texts,
    doc_ids,
    model_name: str = "msmarco-distilbert-base-tas-b",
):
    """Extract embeddings from the dataset using a specified model."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = SentenceTransformer(model_name, device=device)

    # Increase batch size for GPU
    batch_size = 128 if device == "cuda" else 32

    # Process in smaller batches to avoid memory issues
    embeddings = np.array(
        model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True
        )
    )

    return embeddings, doc_ids


def save_embeddings(
    embeddings: np.ndarray,
    doc_ids: list[str],
    out_embeddings_file: str,
    out_docids_file: str,
):
    """Save the embeddings and document IDs to files."""
    np.save(f"embedding/embeddings/{out_docids_file}", np.array(doc_ids))
    np.save(f"embedding/embeddings/{out_embeddings_file}", embeddings)
    print(
        f"Embeddings saved to {out_embeddings_file} and document IDs to {out_docids_file}"
    )


def process_embeddings(
    embeddings: np.ndarray,
    doc_ids: list[str],
    model_name: str = "msmarco-distilbert-base-tas-b",
):
    """Process embeddings and save for the MS MARCO dataset."""
    # Save the embeddings and document IDs
    out_embeddings_file = f"{model_name}_embeddings.npy"
    out_docids_file = f"{model_name}_url.npy"
    save_embeddings(embeddings, doc_ids, out_embeddings_file, out_docids_file)


def main(max_docs: int = None):
    """Main function"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("msmarco_refactor.log")],
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting MS MARCO embedding process...")

    # Check memory usage before loading the dataset
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 * 1024)  # in MB
    logger.info("Memory usage before loading dataset: %.2f MB", memory_usage)

    # Load the dataset
    hf_dataset = load_msmarco_dataset(max_docs=max_docs)

    # Print some information about the dataset
    print(f"Loaded {len(hf_dataset)} documents from MS MARCO.")
    print("Sample document:", hf_dataset[0])

    # Process the dataset to chunk text - remove batched=True for now
    hf_dataset = hf_dataset.map(
        lambda x: {"body": chunk_text(x["body"])}, batch_size=32
    )

    # Extract embeddings
    embeddings, doc_ids = extract_embeddings(hf_dataset["body"], hf_dataset["doc_id"])

    # Process and save the embeddings
    process_embeddings(embeddings, doc_ids)


### Refactor for processing in chunks
def process_in_batches(chunk_size: int = 50000):
    """Process the dataset in batches to avoid memory issues."""

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("msmarco_embeddings.log"),
        ],
    )
    logger = logging.getLogger(__name__)

    # load the dataset iterator
    dataset = ir_datasets.load("msmarco-document")

    # Initialise model once
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("msmarco-distilbert-base-tas-b", device=device)
    batch_size = 128 if device == "cuda" else 32

    logger.info(f"Using device: {device}, batch size: {batch_size}")

    # Process the dataset in chunks
    chunk_num = 0
    docs_buffer = []

    for doc in tqdm(dataset.docs_iter(), desc="Processing documents"):
        docs_buffer.append(
            {
                "doc_id": doc.doc_id,
                "body": chunk_text(doc.body),
                "title": doc.title,
                "url": doc.url,
            }
        )

        # Process the buffer when it reaches the chunk size
        if len(docs_buffer) >= chunk_size:
            process_chunk(docs_buffer, model, batch_size, chunk_num, logger)
            docs_buffer = []  # Clear the buffer
            chunk_num += 1
            gc.collect()  # Force garbage collection to free memory

    # Process any remaining documents in the buffer
    if docs_buffer:
        process_chunk(docs_buffer, model, batch_size, chunk_num, logger)

    logger.info(f"completed processing {chunk_num + 1} chunks of documents.")


def process_chunk(docs_buffer, model, batch_size, chunk_num, logger):
    "process a single chunk of documents and save embeddings"
    logger.info(f"Processing chunk {chunk_num} with {len(docs_buffer)} documents.")

    # Extract texts and doc_ids
    texts = [doc["body"] for doc in docs_buffer]
    doc_ids = [doc["doc_id"] for doc in docs_buffer]

    # Extract embeddings
    embeddings = np.array(
        model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True
        )
    )

    # Save chunk embeddings
    embeddings_file = f"msmarco_chunk_{chunk_num}_embeddings.npy"
    docids_file = f"msmarco_chunk_{chunk_num}_docids.npy"

    os.makedirs("embedding/embeddings", exist_ok=True)
    np.save(f"embedding/embeddings/{docids_file}", np.array(doc_ids))
    np.save(f"embedding/embeddings/{embeddings_file}", embeddings)

    logger.info(f"Saved chunk {chunk_num}: {embeddings_file}, {docids_file}")

    # Check memory usage after processing the chunk
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 * 1024)
    logger.info(
        f"Memory usage after processing chunk {chunk_num}: {memory_usage:.2f} MB"
    )


def combine_chunks():
    """Combine all chunked embeddings into a single file."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("msmarco_refactor.log")],
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting MS MARCO embedding chunk combination...")

    # Find all chunk files
    embeddings_files = sorted(
        glob.glob("embedding/embeddings/msmarco_chunk_*_embeddings.npy")
    )
    docids_files = sorted(glob.glob("embedding/embeddings/msmarco_chunk_*_docids.npy"))

    logger.info(f"Found {len(embeddings_files)} chunk files to combine.")

    # Combine embeddings and doc_ids
    all_embeddings = []
    all_doc_ids = []

    for emb_file, docid_file in zip(embeddings_files, docids_files):
        embeddings = np.load(emb_file)
        doc_ids = np.load(docid_file)

        all_embeddings.append(embeddings)
        all_doc_ids.append(doc_ids)

    # Save combined files
    combined_embeddings = np.vstack(all_embeddings)
    combined_doc_ids = np.array(all_doc_ids)

    logger.info("Saving combined embeddings and document IDs...")
    np.save("embedding/embeddings/msmarco_combined_embeddings.npy", combined_embeddings)
    np.save("embedding/embeddings/msmarco_combined_docids.npy", combined_doc_ids)


if __name__ == "__main__":
    # main(max_docs=1000000)  # Adjust max_docs as needed

    # Process in batches to avoid memory issues
    process_in_batches(chunk_size=50000)

    # Combine all chunked embeddings into a single file
    combine_chunks()
    print("MS MARCO embedding process completed.")
