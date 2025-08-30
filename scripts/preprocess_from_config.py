"""
Run full preprocessing pipeline from config file.
"""

import jsonargparse

from arc_tiptoe.preprocessing.pipeline.full_pipeline import PreprocessingPipeline
from arc_tiptoe.preprocessing.utils.config import PreProcessConfig


def main(json_config_path: str = "config/example_preprocess_config.json"):
    """Main function to run preprocessing from config."""
    config = PreProcessConfig(json_config_path)
    pipeline = PreprocessingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    arg_parser = jsonargparse.ArgumentParser()
    arg_parser.add_argument(
        "--json_config_path", type=str, default="config/example_preprocess_config.json"
    )
    args = arg_parser.parse_args()
    main(args.json_config_path)
