"""
Embedding methods for preprocessing pipeline
"""

import gc
import glob
import logging
import os
from abc import ABC, abstractmethod

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
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    f"{self.config.data['dataset']}_"
                    f"{self.config.data['data_subset_size']}.log"
                ),
            ],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Starting dataset embedding process, checking for precomputed")

        if config.embedding_done:
            self.logger.info("Initial embedding complete")
        else:
            self.gen_directory_structure()
            self.config.embeddings_path = os.path.join(
                "data", self.config.uuid, "embedding", "embeddings"
            )
            self.dataset = None
            if self.config.embed_pars.get("use_gpu", True):
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.load_model(self.config.embed_model, self.device)

    @abstractmethod
    def load_model(self, model_name: str, device: str):
        """Load the embedding model specified in the config."""
        raise NotImplementedError()

    def gen_directory_structure(self):
        """
        Generate the directory structure for the embedding.

        TODO: update when finalised directory structure
        """
        os.makedirs(
            os.path.join("data", self.config.uuid, "embedding", "embeddings"),
            exist_ok=True,
        )
        self.logger.info(
            "Directory structure created at: %s", self.config.embeddings_path
        )

    def load_dataset(self):
        """Load the dataset. TODO: refactor for additional datasets"""
        if self.config.data["dataset"] == "msmarco":
            if self.config.embed_pars.get("chunk_data", False):
                self.dataset = tt_ds.load_msmarco_dataset_ir()
            else:
                self.dataset = tt_ds.load_msmarco_dataset_hf(
                    max_docs=self.config.data["data_subset_size"]
                )
            self.logger.info(
                "MS MARCO dataset loaded with %d documents", self.dataset.docs_count()
            )
        else:
            error_msg = f"Dataset {self.config.data['dataset']} not implemented"
            raise NotImplementedError(error_msg)

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
        hf_dataset = self.dataset.map(
            lambda x: {"body": self._chunk_text(x["body"])}, batch_size=32
        )
        embeddings = np.array(
            self.model.encode(
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

        for doc in tqdm(self.dataset.docs_iter(), desc="Processing documents"):
            docs_buffer.append(
                {
                    "doc_id": doc.doc_id,
                    "body": self._chunk_text(doc.body),
                    "title": doc.title,
                    "url": doc.url,
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

    def _chunk_text(self, text: str, max_length: int = 512) -> list[str]:
        """chunk the text into smaller segments of a specified maximum length"""
        if not text:
            return ""
        return " ".join(text.split()[:max_length])

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
            self.model.encode(
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
        np.save(f"{self.config.embedding_path}/embeddings.npy", combined_embeddings)
        np.save(f"{self.config.embedding_path}/doc_ids.npy", combined_doc_ids)


class SentenceTransformerEmbedder(Embedder):
    """Embedder using SentenceTransformer models."""

    def load_model(self, model_name: str, device: str):
        """Load the SentenceTransformer model."""
        return tt_models.load_sentence_transformer(model_name, device=device)
