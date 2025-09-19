"""
Run a search experiment with specified parameters.
"""

import asyncio

import jsonargparse

from arc_tiptoe.search.search_experiment import SearchExperiment
from arc_tiptoe.search.servers import TiptoeServerManager, aggressive_cleanup


async def main(config_path: str, search_dir: str = "search"):
    """Main entry point for the script."""

    # Clean up previous processes and start servers
    aggressive_cleanup()
    server_manager = TiptoeServerManager(config_path, search_dir)
    if not server_manager.start_servers():
        print("Error: Failed to start servers.")
        return 1

    # Run the search experiment
    experiment = SearchExperiment(config_path, search_dir)
    await experiment.run_experiment(output_path="search_results.csv")

    return 1


if __name__ == "__main__":
    arg_parser = jsonargparse.ArgumentParser()
    arg_parser.add_argument(
        "--json_search_config_path",
        type=str,
        default="config/example_preprocess_config.json",
    )
    args = arg_parser.parse_args()
    asyncio.run(main(args.json_search_config_path))
