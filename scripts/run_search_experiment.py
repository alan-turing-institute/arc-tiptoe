"""
Run a search experiment with specified parameters.
"""

import asyncio

import jsonargparse

from arc_tiptoe.search.search_experiment import (
    SearchExperimentAsync,
    SearchExperimentSingleThread,
)
from arc_tiptoe.search.servers import TiptoeServerManager, aggressive_cleanup


async def main(
    config_path: str,
    search_dir: str = "search",
    dataset_name: str = "msmarco-passage/dev/small",
):
    """Main entry point for the script."""

    # Clean up previous processes and start servers
    aggressive_cleanup()
    server_manager = TiptoeServerManager(config_path, search_dir)
    if not server_manager.start_servers():
        print("Error: Failed to start servers.")
        return 1

    # Run the search experiment
    experiment = SearchExperimentAsync(config_path, search_dir, dataset_name)
    await experiment.run_experiment(output_path="search_results.csv")

    return 1


def main_sync(
    config_path: str,
    search_dir: str = "search",
    queries_path: str = "scratch/msmarco-queries-embedded.csv",
):
    """Main entry point for the script."""

    # Clean up previous processes and start servers
    aggressive_cleanup()
    server_manager = TiptoeServerManager(config_path, search_dir)
    if not server_manager.start_servers():
        print("Error: Failed to start servers.")
        return 1

    # Run the search experiment
    experiment = SearchExperimentSingleThread(config_path, queries_path, search_dir)
    experiment.run_experiment(output_path="search_results.csv")

    return 1


if __name__ == "__main__":
    arg_parser = jsonargparse.ArgumentParser()
    arg_parser.add_argument(
        "--json_search_config_path",
        type=str,
        default="config/example_preprocess_config.json",
    )
    arg_parser.add_argument("--queries_path", type=str, default=None)
    args = arg_parser.parse_args()
    async_mode = False
    if async_mode:
        asyncio.run(main(config_path=args.json_search_config_path))
    else:
        main_sync(
            config_path=args.json_search_config_path,
            queries_path=args.queries_path,
        )
