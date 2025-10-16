"""
Run a search experiment with specified parameters.
"""

import jsonargparse

from arc_tiptoe.search.search_experiment import SearchExperimentSingleThread
from arc_tiptoe.search.servers import TiptoeServerManager, aggressive_cleanup


def main(
    config_path: str,
    search_dir: str = "search",
    queries_path: str = "scratch/msmarco-queries-embedded.csv",
    save_path: str = "search_results.csv",
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
    experiment.run_experiment(output_path=save_path)

    return 1


if __name__ == "__main__":
    arg_parser = jsonargparse.ArgumentParser()
    arg_parser.add_argument(
        "--json_search_config_path",
        type=str,
        default="config/example_preprocess_config.json",
    )
    arg_parser.add_argument("--queries_path", type=str, default=None)
    arg_parser.add_argument("--save_path", type=str, default="search_results.csv")
    arg_parser.add_argument("--search_dir", type=str, default="search")
    args = arg_parser.parse_args()
    main(
        config_path=args.json_search_config_path,
        search_dir=args.search_dir,
        queries_path=args.queries_path,
        save_path=args.save_path,
    )
