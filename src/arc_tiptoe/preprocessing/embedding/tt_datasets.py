"""
Load MSMARCO dataset
"""

import ir_datasets
from datasets import Dataset
from tqdm import tqdm


def load_msmarco_dataset_hf(max_docs: int | None = None) -> Dataset:
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


def load_msmarco_dataset_ir() -> ir_datasets.Dataset:
    """Load MSMARCO using ir_datasets."""
    return ir_datasets.load("msmarco-document")


### IR_Datasets version ###
def load_ir_dataset(dataset_name: str) -> ir_datasets.Dataset:
    """Load an IR dataset using ir_datasets.

    Args:
        dataset_name (str): The name of the dataset to load.

    Returns:
        ir_datasets.Dataset: The loaded dataset.
    """
    return ir_datasets.load(dataset_name)


def load_ir_dataset_hf(dataset_name: str, max_docs: int | None = None) -> Dataset:
    """Load an IR dataset using ir_datasets and convert it to a Hugging Face dataset.

    Args:
        dataset_name (str): The name of the dataset to load.
        max_docs (int | None): The maximum number of documents to load.
                                If None, load all documents.

    Returns:
        Dataset: The loaded dataset in Hugging Face format.
    """
    dataset = ir_datasets.load(dataset_name)

    def convert_to_hf_format():
        docs = []
        for doc in tqdm(dataset.docs_iter(), desc="Loading documents"):
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
