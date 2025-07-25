"""
Refactoring the MS MARCO embedding code to improve readability and maintainability,
including using the huggingface datasets library for data handling.
"""

import ir_datasets
import numpy as np
from config import SEQ_LEN
from datasets import Dataset
from sentence_transformers import SentenceTransformer


def load_msmarco_dataset():
    """Load the MS MARCO dataset and convert it to a Hugging Face dataset."""
    dataset = ir_datasets.load("msmarco-doc-dev")

    def convert_to_hf_format():
        docs = []
        for doc in dataset.docs_iter():
            docs.append(
                {
                    "doc_id": doc.doc_id,
                    "body": doc.text,
                    "title": doc.title,
                    "url": doc.url,
                }
            )
        return Dataset.from_dict({"docs": docs})

    return convert_to_hf_format()


def chunk_text(text: str, seq_len: int = SEQ_LEN) -> str:
    """Chunk the text into smaller segments of a specified length."""
    words = text.split()
    return " ".join(words[:seq_len]) if len(words) > seq_len else text


def extract_embeddings(
    dataset: Dataset, model_name: str = "msmarco-distilbert-base-tas-b"
) -> tuple[np.ndarray, list[str]]:
    """Extract embeddings from the dataset using a specified model."""
    model = SentenceTransformer(model_name)

    # Prepare the text data for embedding
    embeddings = np.array(model.encode(dataset, batch_size=32, convert_to_numpy=True))

    return embeddings, [doc["doc_id"] for doc in dataset]


def save_embeddings(
    embeddings: np.ndarray,
    doc_ids: list[str],
    out_embeddings_file: str,
    out_docids_file: str,
):
    """Save the embeddings and document IDs to files."""
    np.save(out_docids_file, np.array(doc_ids))
    np.save(out_embeddings_file, embeddings)
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


def main():
    """Main function"""
    # Load the dataset
    hf_dataset = load_msmarco_dataset()

    # Print some information about the dataset
    print(f"Loaded {len(hf_dataset)} documents from MS MARCO.")
    print("Sample document:", hf_dataset[0])

    # Process the dataset to chunk text
    hf_dataset = hf_dataset.map(
        lambda x: {"body": chunk_text(x["body"])}, batched=True, batch_size=32
    )

    # Extract embeddings
    embeddings, doc_ids = extract_embeddings(hf_dataset["body"])

    # Process and save the embeddings
    process_embeddings(embeddings, doc_ids)


if __name__ == "__main__":
    main()
