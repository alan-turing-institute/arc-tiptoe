"""
Configuration utilities for the preprocessing pipeline.
"""

import hashlib
import json
import os
import uuid

# from known_methods import (
#     CLUSTERING_METHODS,
#     DIMENSIONALITY_REDUCTION_METHODS,
#     EMBEDDING_MODELS,
# )

BASE_DIR = "data/"


class PreProcessConfig:
    """
    Preprocessing config class. Either generates or reads a config JSON, generating a
    uuid for the preprocessing pipeline based on the config, and checks what stages have
    already been completed.

    config structure:
    {
        uuid: <UUID> - generated from rest of config
        embed_model: <EMBED_MODEL> - str
        data: {
            dataset: <DATASET> - str
            data_subset_size: <DATA_SUBSET_SIZE> - int
        }
        cluster: {
            apply_clustering: <APPLY_CLUSTERING> - bool
            clustering_method: <CLUSTERING_METHOD> - str
        }
        dim_red: {
            apply_dim_red: <APPLY_DIM_RED> - bool
            dim_red_before_clustering: <DIM_RED_BEFORE_CLUSTERING> - bool
            dim_red_method: <DIM_RED_METHOD> - str
            dim_red_dimension: <DIM_RED_DIMENSION> - int
        }
    }

    It will save the config to a JSON file and return the UUID if generated. It will
    also generate the following data directory structure, or check if present:
    data/
    |-- uuid/
        |-- config.json
        |-- embedding/
        |-- clusters/
        |-- logs/
        |-- artifact/
        TODO: finish off data directory structure

    If a config overlaps with an existing one in earlier stages, e.g. the same embedding
    model is used, this will be checked for and used as a checkpoint for the later
    stages to avoid additional computation.
    """

    def __init__(self, config_path: str | None):
        self.uuid = None
        self.embed_model = None
        self.embed_pars = {
            "chunk_data": True,
            "chunk_size": 50000,
            "batch_size": 256,
            "sequence_length": 512,
            "use_gpu": True,
        }
        self.data = {"dataset": None, "data_subset_size": None}
        self.cluster = {
            "apply_clustering": None,
            "clustering_method": None,
            "avg_bundle_size": 4000,
            "urls_per_bundle": 160,
            "max_size": 4000,
        }
        self.dim_red = {
            "apply_dim_red": None,
            "dim_red_before_clustering": None,
            "dim_red_method": None,
            "dim_red_dimension": None,
        }

        # flags for completed preprocessing
        self.embedding_done = False
        self.clustering_done = False
        self.dim_red_done = False

        # paths to computed objects
        self.embeddings_path = None
        self.clustering_path = None
        self.dim_red_path = None

        # load config from existing or create new config
        if config_path is not None:
            self.load_config(config_path)
        else:
            self.gen_config_from_cli()

        # create directory structure
        self.gen_directory_structure()

        # save updated config
        self.save_config()

    def load_config(self, config_path: str):
        """
        Load the config from a pre-existing JSON file. If the uuid has not been
        generated, it will be created based on the current config.
        """
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
            self.uuid = config.get("uuid", None)
            self.embed_model = config.get("embed_model", None)
            self.embed_pars = config.get(
                "embed_pars",
                {
                    "chunk_data": None,
                    "chunk_size": None,
                    "batch_size": None,
                    "use_gpu": None,
                    "sequence_length": None,
                },
            )
            self.data = config.get("data", {"dataset": None, "data_subset_size": None})
            self.cluster = config.get(
                "cluster",
                {
                    "apply_clustering": None,
                    "clustering_method": None,
                    "avg_bundle_size": 4000,
                    "urls_per_bundle": 160,
                    "max_size": 4000,
                },
            )
            self.dim_red = config.get(
                "dim_red",
                {
                    "apply_dim_red": None,
                    "dim_red_before_clustering": None,
                    "dim_red_method": None,
                    "dim_red_dimension": None,
                },
            )

        if self.uuid is None:
            self.uuid = self.gen_uuid()

    def gen_uuid(self):
        """Generate the uuid from the config. For simpler comparison it appends
        numerical values in the following way:

        final_id = uuid_<DATA_SUBSET_SIZE>_<DIM_RED_DIMENSION>
        """
        config_str = (
            f"{self.embed_model}-"
            f"{self.data['dataset']}-"
            f"{self.cluster['apply_clustering']}-"
            f"{self.cluster['clustering_method']}-"
            f"{self.dim_red['apply_dim_red']}-"
            f"{self.dim_red['dim_red_before_clustering']}-"
            f"{self.dim_red['dim_red_method']}"
        )
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        return (
            f"{uuid.UUID(config_hash)!s}-"
            f"{self.data['data_subset_size']}-{self.dim_red['dim_red_dimension']}"
        )

    def gen_config_from_cli(self):
        """Gen the config from a cli input"""
        return NotImplementedError()

    def gen_directory_structure(self):
        """
        Generate the directory structure for the data.

        TODO: update when finalised directory structure
        """
        os.makedirs(os.path.join(BASE_DIR, self.uuid), exist_ok=True)
        os.makedirs(os.path.join(BASE_DIR, self.uuid, "embedding"), exist_ok=True)
        os.makedirs(os.path.join(BASE_DIR, self.uuid, "clusters"), exist_ok=True)
        os.makedirs(os.path.join(BASE_DIR, self.uuid, "logs"), exist_ok=True)
        os.makedirs(os.path.join(BASE_DIR, self.uuid, "artifact"), exist_ok=True)

    def check_for_previous_compute(self):
        """Check for previously computed objects, copy/reference if done"""
        return NotImplementedError()

    def save_config(self):
        """Save the config to a JSON file in the data directory"""
        config = {
            "uuid": self.uuid,
            "embed_model": self.embed_model,
            "data": self.data,
            "cluster": self.cluster,
            "dim_red": self.dim_red,
        }
        config_path = os.path.join(BASE_DIR, self.uuid, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
