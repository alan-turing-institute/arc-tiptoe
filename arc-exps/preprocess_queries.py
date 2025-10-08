import argparse
import json
import os
from pathlib import Path

import pandas as pd
from ir_datasets import load
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def main(config: dict):
    model_name = config["embed_model"]
    model = SentenceTransformer(model_name, device="mps")
    dataset = load(config["data"]["dataset"])

    output_dir = (
        Path("processed_queries")
        / config["data"]["dataset"].replace("/", "_")
        / f"{model_name.replace('/', '_')}.csv"
    )
    os.makedirs(output_dir.parent.absolute(), exist_ok=True)

    embeddings_file = pd.DataFrame(columns=["query_id", "embedding", "text"])

    for query in tqdm(dataset.queries_iter(), total=dataset.queries_count()):
        output = model.encode_query([query.text], convert_to_numpy=True)[0]
        embeddings_file = pd.concat(
            [
                embeddings_file,
                pd.DataFrame(
                    {
                        "query_id": [query.query_id],
                        "embedding": [output.tolist()],
                        "text": [query.text],
                    }
                ),
            ],
            ignore_index=True,
        )

    embeddings_file.to_csv(output_dir, index=False)
    print(f"Embeddings saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process queries with embedding model")
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to JSON config file"
    )
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = json.load(f)

    main(config)
