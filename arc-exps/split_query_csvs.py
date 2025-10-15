import argparse
import os
from os import path
from typing import NamedTuple

import pandas as pd


class ScriptArgs(NamedTuple):
    target_csv_path: str
    chunk_size: int = 500


def main(args: ScriptArgs) -> None:
    full_csv_df = pd.read_csv(args.target_csv_path)
    save_directory = f"{args.target_csv_path.rstrip('.csv')}_shards"
    os.makedirs(save_directory, exist_ok=True)

    # Split the DataFrame into smaller chunks
    chunk_size = args.chunk_size
    for i in range(0, len(full_csv_df), chunk_size):
        chunk = full_csv_df.iloc[i : i + chunk_size]
        chunk.to_csv(
            path.join(save_directory, f"shard_{i // chunk_size}.csv"), index=False
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target_csv_path",
        type=str,
        help="Path to the target CSV file to be split.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500,
        help="Number of rows per chunk.",
    )
    args = parser.parse_args()
    main(args)
