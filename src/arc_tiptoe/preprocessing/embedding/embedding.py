"""
Embedding methods for preprocessing pipeline
"""

import gc
import glob
import logging
import os
from abc import ABC, abstractmethod
from copy import copy

import numpy as np
import torch
from tqdm import tqdm

import arc_tiptoe.preprocessing.embedding.tt_datasets as tt_ds
import arc_tiptoe.preprocessing.embedding.tt_models as tt_models
from arc_tiptoe.preprocessing.utils.config import PreProcessConfig


class Embedder(ABC):
    """Base class for all embedding methods. The embeddings are placed in the
    subdirectory 'embeddings' within the embedding directory."""

    def __init__(self, config: PreProcessConfig, within_pipeline: bool = False):
        self.config = config
        self.within_pipeline = within_pipeline
        # replace datasetname slashes with underscores for file paths
        dataset_name_safe = copy(self.config.data["dataset"]).replace("/", "_")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    f"data/{self.config.uuid}/"
                    f"{dataset_name_safe}_"
                    f"{self.config.data['data_subset_size']}_embedding.log"
                ),
            ],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Starting dataset embedding process, checking for precomputed")

        if config.embedding_done:
            self.logger.info("Initial embedding complete")
            if within_pipeline:
                self._return_config_in_pipeline()
        else:
            self.dataset = None
            self.device = "cpu"
            if self.config.embed_pars.get("use_gpu", True):
                if torch.cuda.is_available():
                    self.device = "cuda"
                    self.logger.info("Using cuda GPU")
                elif torch.backends.mps.is_available():
                    self.device = "mps"
            self.model = self.load_model(self.config.embed_model, self.device)
            if self.config.embed_pars["preprocessing_required"]:
                print(f"Preprocessing documents")

    def _return_config_in_pipeline(self):
        if self.within_pipeline:
            return self.config
        return self.config.save_config()

    @abstractmethod
    def load_model(self, model_name: str, device: str):
        """Load the embedding model specified in the config."""
        raise NotImplementedError()

    def gen_directory_structure(self):
        """
        Generate the directory structure for the embedding. See README for details.
        """
        os.makedirs(
            os.path.join("data", self.config.uuid, "embedding", "embeddings"),
            exist_ok=True,
        )
        self.logger.info(
            "Directory structure created at: %s", self.config.embeddings_path
        )

    def load_dataset(self):
        """Load the dataset into either a Hugging Face dataset or an ir_datasets
        dataset depending on whether chunking is required.

        Sets self.dataset to the loaded dataset.
        """
        if not self.config.embed_pars.get("chunk_data"):
            self.dataset = tt_ds.load_ir_dataset_hf(
                self.config.data["dataset"],
                max_docs=self.config.data["data_subset_size"],
            )
            self.logger.info(
                "Dataset %s loaded with %d documents",
                self.config.data["dataset"],
                self.dataset.num_rows,
            )
            return 1
        self.dataset = tt_ds.load_ir_dataset(self.config.data["dataset"])
        self.logger.info(
            "Dataset %s loaded with %d documents",
            self.config.data["dataset"],
            self.dataset.docs_count(),
        )
        return 1

    def embed(self):
        """Embed the dataset using the specified embedding model."""
        if self.dataset is None:
            self.logger.error("Dataset not loaded")
            return 1
        self.load_dataset()

        self.logger.info("Starting embedding process")

        if self.config.embed_pars["chunk_data"]:
            self.logger.info("Processing dataset in chunks")
            chunk_path = f"{self.config.embeddings_path}/chunks"
            if not os.path.exists(chunk_path):
                os.makedirs(chunk_path, exist_ok=True)
            self._process_in_batches(chunk_path=chunk_path)
            self._combine_chunks(chunk_path=chunk_path)
        else:
            self.logger.info("Processing entire dataset at once")
            self._embed_entire_dataset()

        self.config.embedding_done = True
        self.logger.info("Embedding process complete")
        self.config.save_config()

        if self.within_pipeline:
            return self.config

        return self.config.save_config()

    def _embed_entire_dataset(self):
        """Emnbed the entire dataset"""
        hf_dataset = self.dataset
        if self.config.embed_pars["preprocessing_required"]:
            hf_dataset = self.dataset.map(
                lambda x: {
                    "body": self._preprocess_data(
                        x["body"], max_length=self.config.embed_pars["sequence_length"]
                    )
                },
                batch_size=32,
            )

        embeddings = np.array(
            self.model.encode_document(
                hf_dataset["body"],
                batch_size=self.config.embed_pars.get("batch_size", 256),
                convert_to_numpy=True,
                show_progress_bar=True,
            )
        )
        doc_ids = hf_dataset["doc_id"]
        np.save(f"{self.config.embeddings_path}/embeddings.npy", embeddings)
        np.save(f"{self.config.embeddings_path}/doc_ids.npy", np.array(doc_ids))
        self.logger.info(
            "Embeddings and doc ids saved to %s", self.config.embeddings_path
        )

    def _process_in_batches(self, chunk_path: str):
        """Process the dataset in batches to avoid memory issues."""
        chunk_size = self.config.embed_pars.get("chunk_size", 50000)
        batch_size = self.config.embed_pars.get("batch_size", 256)
        self.logger.info(
            "Processing dataset in chunks of size %d with batch size %d",
            chunk_size,
            batch_size,
        )

        total_chunks = (self.dataset.docs_count() + chunk_size - 1) // chunk_size
        self.logger.info("Total chunks to process: %d", total_chunks)

        chunk_num = 0
        docs_buffer = []

        # Use subset of dataset if given
        if self.config.data["data_subset_size"] is not None:
            self.logger.info(
                "Using subset of dataset: %d documents",
                self.config.data["data_subset_size"],
            )
            ds_iter = self.dataset.docs_iter()[: self.config.data["data_subset_size"]]
        else:
            self.logger.info("Using full dataset")
            ds_iter = self.dataset.docs_iter()

        for doc in tqdm(ds_iter, desc="Processing documents"):
            doc_body = doc.body if "body" in dir(doc) else doc.text
            docs_buffer.append(
                {
                    "doc_id": doc.doc_id,
                    "body": (
                        self._preprocess_data(
                            doc_body,
                            max_length=self.config.embed_pars["sequence_length"],
                        )
                        if self.config.embed_pars["preprocessing_required"]
                        else doc_body
                    ),
                    "title": doc.title if "title" in dir(doc) else "",
                    "url": doc.url if "url" in dir(doc) else "",
                }
            )

            if len(docs_buffer) >= chunk_size:
                self._process_chunk(
                    docs_buffer,
                    chunk_num,
                    self.config.embed_pars["batch_size"],
                    chunk_path,
                )
                docs_buffer = []
                chunk_num += 1
                gc.collect()

        if docs_buffer:
            self._process_chunk(
                docs_buffer,
                chunk_num,
                self.config.embed_pars["batch_size"],
                chunk_path,
            )

        self.logger.info("Completed processing %d chunks of documents.", chunk_num + 1)

    def _preprocess_data(self, text, **kwargs):
        """Preprocess the data as required for models"""
        return tt_models.PREPROCESSING_METHODS[self.config.embed_model](text, **kwargs)

    def _process_chunk(
        self, docs_buffer: list[dict], chunk_num: int, batch_size: int, chunk_path: str
    ):
        """Process a single chunk of documents and save embeddings"""
        self.logger.info(
            "Processing chunk %d with %s documents.", chunk_num, len(docs_buffer)
        )

        texts = [doc["body"] for doc in docs_buffer]
        doc_ids = [doc["doc_id"] for doc in docs_buffer]

        embeddings = np.array(
            self.model.encode_document(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=True,
            )
        )

        embeddings_file = f"msmarco_chunk_{chunk_num}_embeddings.npy"
        docids_file = f"msmarco_chunk_{chunk_num}_docids.npy"

        os.makedirs(chunk_path, exist_ok=True)
        np.save(os.path.join(chunk_path, docids_file), np.array(doc_ids))
        np.save(os.path.join(chunk_path, embeddings_file), embeddings)

        self.logger.info(
            "Saved chunk %d: %s, %s",
            chunk_num,
            embeddings_file,
            docids_file,
        )

    def _combine_chunks(self, chunk_path: str):
        """Combine all chunked embeddings into a single file."""
        self.logger.info("Combining chunked embeddings from %s", chunk_path)

        embedding_files = sorted(
            glob.glob(os.path.join(chunk_path, "msmarco_chunk_*_embeddings.npy"))
        )
        docid_files = sorted(
            glob.glob(os.path.join(chunk_path, "msmarco_chunk_*_docids.npy"))
        )
        self.logger.info(
            "Found %d embedding files and %d docid files",
            len(embedding_files),
            len(docid_files),
        )

        all_embeddings = []
        all_doc_ids = []

        for emb_file, docid_file in zip(embedding_files, docid_files, strict=True):
            self.logger.info("Loading %s and %s", emb_file, docid_file)
            embeddings = np.load(emb_file)
            doc_ids = np.load(docid_file)

            all_embeddings.append(embeddings)
            all_doc_ids.extend(doc_ids)

        combined_embeddings = np.vstack(all_embeddings)
        combined_doc_ids = np.array(all_doc_ids)

        self.logger.info("saving combined embeddings and doc ids")
        np.save(f"{self.config.embeddings_path}/embeddings.npy", combined_embeddings)
        np.save(f"{self.config.embeddings_path}/doc_ids.npy", combined_doc_ids)


class SentenceTransformerEmbedder(Embedder):
    """Embedder using SentenceTransformer models."""

    def load_model(self, model_name: str, device: str):
        """Load the SentenceTransformer model."""
        model = tt_models.load_sentence_transformer(model_name, device=device)
        print(f"Model is using device {model.device}")
        return model
