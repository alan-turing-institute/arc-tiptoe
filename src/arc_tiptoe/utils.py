import json
import logging
import os

DATASET_IR_MAP = {
    # IR Datasets name : filesystem-safe name
    "msmarco-document/trec-dl-2019": "msmarco-document_trec-dl-2019",
    "msmarco-document/dev": "msmarco-document_dev",
    "wikir_en1k_test": "wikir/en1k/test",
}

DATASET_SAVE_MAP = {
    # Filesystem-safe name : IR Datasets name
    value: key
    for key, value in DATASET_IR_MAP.items()
}


def save_to_json(results: dict, save_path: str, **json_kwargs) -> str:
    """Save outputs to a JSON file.
    Args:
        results (dict): The results to save.
        save_path (str): The path where the JSON file will be saved.
        **json_kwargs: Additional keyword arguments for json.dump().
    Returns:
        str: The path where the JSON file was saved."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, **json_kwargs)

    msg = f"Outputs saved to {save_path}"
    logging.info(msg)

    return save_path


def parse_json(results_pth: str) -> dict:
    """Load from a JSON file."""
    with open(results_pth) as f:
        return json.load(f)
