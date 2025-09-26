import json
import logging
import os


def save_to_json(results, save_path, **json_kwargs):
    """Save outputs to a JSON file."""
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
