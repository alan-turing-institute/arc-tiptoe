import argparse
import os
from pathlib import Path

import pandas as pd


def collate_csvs(input_dir: str):
    dir_components = input_dir.split("/")
    model = dir_components[-1]
    dataset = dir_components[-3]

    output_dir = f"results/{dataset}/{model}/search_results/"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "combined_results.csv")

    input_path = Path(input_dir)
    all_files = list(input_path.glob("result*.csv"))
    df_list = []

    for file in all_files:
        df = pd.read_csv(file)
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collate CSV files from a directory.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing CSV files to collate.",
    )
    args = parser.parse_args()
    collate_csvs(args.input_dir)
