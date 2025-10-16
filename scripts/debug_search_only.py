import argparse

from arc_tiptoe.search.search_experiment import SearchExperimentSingleThread


def main(config_path: str, search_dir: str, queries_path: str, save_path: str) -> int:
    # Run the search experiment
    experiment = SearchExperimentSingleThread(config_path, queries_path, search_dir)
    experiment.run_experiment(output_path=save_path)

    return 1


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--json_search_config_path",
        type=str,
        required=True,
    )
    arg_parser.add_argument("--queries_path", type=str, required=True)
    arg_parser.add_argument("--save_path", type=str, required=True)
    arg_parser.add_argument("--search_dir", type=str, required=True)
    args = arg_parser.parse_args()

    main(
        config_path=args.json_search_config_path,
        search_dir=args.search_dir,
        queries_path=args.queries_path,
        save_path=args.save_path,
    )

#
# --json_search_config_path /Users/jbishop/dev/workspace/concealed_beagle/tiptoe/configs/test_search.json --save_path /Users/jbishop/dev/workspace/concealed_beagle/tiptoe/output.csv --queries_path ./queries_2.tsv --search_dir ./search
# --json_search_config_path /Users/jbishop/dev/workspace/concealed_beagle/tiptoe/configs/test_search.json --save_path /Users/jbishop/dev/workspace/concealed_beagle/tiptoe/output.csv --queries_path ./queries_1.tsv --search_dir ./search
