"""
Refactoring the MS MARCO embedding code to improve readability and maintainability,
including using the huggingface datasets library for data handling.
"""

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


if __name__ == "__main__":
    main(max_docs=1000000)  # Adjust max_docs as needed
