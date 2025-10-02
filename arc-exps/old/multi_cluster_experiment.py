"""
Full multi-cluster experiments on dataset
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import runExp_refactor
from analyse_results import TiptoeAnalyser
from use_qrels_queries import (
    create_subset_with_qrels,
    find_qrels_file,
    get_queries_with_qrels,
)


class FullMultiClusterExperiment:
    """Run full multi-cluster experiments on a dataset."""

    def __init__(self, base_dir: str = "/home/azureuser"):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "full_multi_cluster_results"
        self.results_dir.mkdir(exist_ok=True)

    def create_full_query_set(self):
        """Create query set using ALL queries that have qrels"""
        print("🔍 Finding all queries with qrels...")

        # Find qrels file
        qrels_file = find_qrels_file(str(self.base_dir))
        if not qrels_file:
            raise FileNotFoundError("No qrels file found")

        print("📊 Counting all queries with qrels...")
        # Get ALL queries with qrels (no limit)
        all_query_ids = get_queries_with_qrels(qrels_file, max_queries=999999)

        print(f"✅ Found {len(all_query_ids)} total queries with qrels")

        # Create full query file using the existing function
        query_file = create_subset_with_qrels(str(self.base_dir), len(all_query_ids))

        if not query_file:
            raise RuntimeError("Failed to create full query file")

        print(f"📁 Full query file created: {query_file}")
        return query_file, len(all_query_ids)

    def create_experiment_config(
        self, query_file: str, num_clusters: int, optimisation: str
    ) -> str:
        """Create the experiment configuration for multi-cluster experiments."""

        # Check what data structure actually exists
        data_dir = self.base_dir / "data"

        print(f"🔍 Checking data structure in {data_dir}")
        if data_dir.exists():
            subdirs = [d.name for d in data_dir.iterdir() if d.is_dir()]
            print(f"   Available subdirectories: {subdirs}")

        # Try to find the correct paths based on your workspace structure
        possible_paths = {
            "pca_components": [
                self.base_dir / "data" / "embeddings" / "pca_components_192.txt",
                self.base_dir / "data" / "pca_components_192.txt",
                self.base_dir / "pca_components_192.txt",
            ],
            "centroids": [
                self.base_dir / "data" / "embeddings" / "centroids.txt",
                self.base_dir / "data" / "centroids.txt",
                self.base_dir / "centroids.txt",
            ],
            "index": [
                self.base_dir / "data" / "artifact" / "dim192" / "index.faiss",
                self.base_dir / "data" / "index.faiss",
                self.base_dir / "index.faiss",
            ],
            "clusters": [
                self.base_dir / "data" / "clusters",
                self.base_dir / "clusters",
            ],
        }

        # Find existing paths
        actual_paths = {}
        for key, paths in possible_paths.items():
            for path in paths:
                if path.exists():
                    actual_paths[key] = str(path)
                    print(f"   ✅ Found {key}: {path}")
                    break
            if key not in actual_paths:
                print(
                    f"   ❌ Could not find {key} in any of: {[str(p) for p in paths]}"
                )

        # Use the EXACT config keys that clustering/search.py expects
        config = {
            # These are the exact keys from clustering/search.py
            "pca_components_file": actual_paths.get(
                "pca_components"
            ),  # Note: lowercase!
            "query_file": str(query_file),  # Note: lowercase!
            "cluster_file_location": actual_paths.get(
                "clusters", str(self.base_dir / "data" / "clusters")
            )
            + "/",
            "url_bundle_base_dir": actual_paths.get(
                "clusters", str(self.base_dir / "data" / "clusters")
            )
            + "/",
            "index_file": actual_paths.get("index"),  # Note: lowercase!
            "is_text": True,  # Note: lowercase!
            "run_msmarco_dev_queries": True,  # Note: lowercase!
            "filter_badwords": False,
            "short_exp": False,  # Full experiment, not short
            "num_clusters": num_clusters,  # Note: lowercase!
            "centroids_file": actual_paths.get("centroids"),  # Note: lowercase!
            "badwords_file": None,
            "img_results_dir": None,
        }

        # Apply optimization settings with correct keys
        if optimisation == "basic":
            config.update(
                {
                    "run_pca": False,
                    "run_url_filter": False,
                    "url_filter_by_cluster": False,
                }
            )
        elif optimisation == "url_cluster":
            config.update(
                {
                    "run_pca": False,
                    "run_url_filter": True,
                    "url_filter_by_cluster": True,
                }
            )
        elif optimisation == "pca":
            config.update(
                {
                    "run_pca": True,
                    "run_url_filter": True,
                    "url_filter_by_cluster": True,
                }
            )

        # Save config
        config_name = f"full_config_{num_clusters}c_{optimisation}.json"
        config_file = self.results_dir / config_name

        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        print(f"✅ Config created: {config_file}")
        print(f"   Clusters to search: {num_clusters}")
        print(f"   Optimization: {optimisation}")

        # Debug: Print the actual config that was saved
        print(f"📋 Config contents:")
        for key, value in config.items():
            if value is not None:
                print(f"   {key}: {value}")
            else:
                print(f"   ❌ {key}: None (missing!)")

        return str(config_file)

    def run_quality_experiment(self, config_file: str, experiment_name: str) -> str:
        """Run quality experiment using clustering/search.py"""

        print(f"🔍 Running quality experiment: {experiment_name}")

        # First, let's validate the configuration
        try:
            with open(config_file, "r") as f:
                config = json.load(f)

            print(f"📋 Validating configuration:")
            print(f"   Query file: {config.get('QUERY_FILE')}")
            print(f"   Clusters: {config.get('NUM_CLUSTERS')}")
            print(f"   Index file: {config.get('INDEX_FILE')}")
            print(f"   Cluster location: {config.get('CLUSTER_FILE_LOCATION')}")

            # Check if files exist
            query_file = config.get("QUERY_FILE")
            if query_file and Path(query_file).exists():
                # Check query file format
                with open(query_file, "r") as f:
                    lines = f.readlines()[:5]
                    print(f"   Query file exists with {len(lines)} sample lines:")
                    for i, line in enumerate(lines):
                        print(f"     {i+1}: {line.strip()[:100]}")
            else:
                print(f"   ❌ Query file not found: {query_file}")

            index_file = config.get("INDEX_FILE")
            if index_file and Path(index_file).exists():
                print(
                    f"   ✅ Index file exists: {Path(index_file).stat().st_size} bytes"
                )
            else:
                print(f"   ❌ Index file not found: {index_file}")

            cluster_dir = config.get("CLUSTER_FILE_LOCATION")
            if cluster_dir and Path(cluster_dir).exists():
                cluster_files = list(Path(cluster_dir).glob("*.txt"))
                print(
                    f"   ✅ Cluster directory exists with {len(cluster_files)} .txt files"
                )
            else:
                print(f"   ❌ Cluster directory not found: {cluster_dir}")

        except Exception as e:
            print(f"   ❌ Config validation error: {e}")

        quality_log = self.results_dir / f"{experiment_name}_quality.log"

        cmd = ["python3", "search.py", config_file]
        cwd = "clustering"

        try:
            print(f"   Executing: {' '.join(cmd)} (in {cwd})")
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout for full experiments
                check=False,
            )

            print(f"   Return code: {result.returncode}")
            print(f"   Stdout length: {len(result.stdout)} chars")
            print(f"   Stderr length: {len(result.stderr)} chars")

            # Show stderr for debugging
            if result.stderr:
                print(f"   Stderr preview: {result.stderr[:500]}")

            # Show first few lines of stdout
            if result.stdout:
                stdout_lines = result.stdout.split("\n")[:10]
                print(f"   Stdout preview:")
                for i, line in enumerate(stdout_lines):
                    if line.strip():
                        print(f"     {i+1}: {line}")

            if result.returncode == 0:
                # Save quality results
                with open(quality_log, "w") as f:
                    f.write(result.stdout)

                # Parse results summary
                lines = result.stdout.split("\n")
                query_count = len([line for line in lines if "Query:" in line])
                result_count = len(
                    [
                        line
                        for line in lines
                        if line.strip()
                        and not line.startswith("Query:")
                        and not line.startswith("---------------")
                    ]
                )

                print(f"   ✅ Quality: {query_count} queries, {result_count} results")

                if query_count == 0:
                    print("   ⚠️  No queries processed - checking configuration...")
                    return None

                return str(quality_log)
            else:
                print(f"   ❌ Quality failed: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            print(f"   ❌ Quality experiment timed out")
            return None
        except Exception as e:
            print(f"   ❌ Quality experiment error: {e}")
            return None

    def run_performance_experiment(
        self, num_clusters: int, experiment_name: str
    ) -> str:
        """Run performance experiment"""

        print(f"⏱️  Running performance experiment: {experiment_name}")

        # Scale servers based on cluster count for full experiments
        num_embed_servers = min(8, max(4, num_clusters * 2))
        num_url_servers = min(4, max(2, num_clusters))

        print(f"   Servers: {num_embed_servers} embed + {num_url_servers} URL")

        cluster = runExp_refactor.LocalTiptoeCluster(
            num_embed_servers=num_embed_servers,
            num_url_servers=num_url_servers,
            preamble=str(self.base_dir),
            image_search=False,
        )

        try:
            cluster.start_embedding_servers()
            cluster.start_url_servers()
            cluster.start_coordinator()

            latency_results = cluster.run_latency_experiment()

            if latency_results:
                latency_log = self.results_dir / f"{experiment_name}_latency.log"
                with open(latency_log, "w") as f:
                    f.write(latency_results)

                print(f"   ✅ Performance: completed")
                return str(latency_log)
            else:
                print(f"   ❌ Performance: failed")
                return None

        except Exception as e:
            print(f"   ❌ Performance error: {e}")
            return None
        finally:
            cluster.cleanup()

    def run_full_experiments(self, cluster_configs: list, optimization_levels: list):
        """Run full experiments across all configurations"""

        print("🚀 Starting full multi-cluster experiments...")
        print(f"   Cluster configs: {cluster_configs}")
        print(f"   Optimizations: {optimization_levels}")

        # Create full query set
        query_file, total_queries = self.create_full_query_set()
        print(f"📊 Using {total_queries} queries with qrels")

        all_results = []

        for num_clusters in cluster_configs:
            for optimization in optimization_levels:
                experiment_name = f"{num_clusters}c_{optimization}"

                print(f"\n🔬 Running experiment: {experiment_name}")

                try:
                    # Create config
                    config_file = self.create_experiment_config(
                        query_file, num_clusters, optimization
                    )

                    # Run quality experiment
                    quality_log = self.run_quality_experiment(
                        config_file, experiment_name
                    )

                    # Run performance experiment
                    latency_log = self.run_performance_experiment(
                        num_clusters, experiment_name
                    )

                    # Analyze results
                    if quality_log and latency_log:
                        # Create temporary results directory for analyzer
                        temp_results_dir = self.results_dir / f"{experiment_name}_temp"
                        temp_results_dir.mkdir(exist_ok=True)

                        # Copy logs to temp directory with expected names
                        import shutil

                        shutil.copy(
                            quality_log,
                            temp_results_dir / f"{experiment_name}_quality.log",
                        )
                        shutil.copy(
                            latency_log,
                            temp_results_dir / f"{experiment_name}_latency.log",
                        )

                        # Analyze
                        analyzer = TiptoeAnalyser(str(temp_results_dir))
                        analysis = analyzer.analyze_all_results()

                        # Extract metrics
                        latency_metrics = analysis.get("single_cluster", {}).get(
                            "latency", {}
                        )
                        quality_metrics = analysis.get("single_cluster", {}).get(
                            "quality", {}
                        )

                        result = {
                            "experiment_name": experiment_name,
                            "num_clusters": num_clusters,
                            "optimization": optimization,
                            "total_queries": total_queries,
                            "mrr_100": quality_metrics.get("mrr_100", 0),
                            "total_comm_mb": latency_metrics.get("total_comm_mb", 0),
                            "avg_latency": latency_metrics.get("avg_latency", 0),
                            "upload_mb": latency_metrics.get("upload_mb", 0),
                            "download_mb": latency_metrics.get("download_mb", 0),
                            "queries_processed": quality_metrics.get(
                                "queries_evaluated", 0
                            ),
                        }

                        all_results.append(result)

                        print(
                            f"   📊 Results: MRR={result['mrr_100']:.4f}, Comm={result['total_comm_mb']:.2f}MB"
                        )

                        # Cleanup temp directory
                        shutil.rmtree(temp_results_dir)
                    else:
                        print(f"   ❌ Experiment {experiment_name} failed")

                    # Brief pause between experiments
                    time.sleep(10)

                except Exception as e:
                    print(f"   ❌ Experiment {experiment_name} error: {e}")
                    continue

        # Save all results
        results_file = self.results_dir / "full_experiment_results.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n🎉 Full experiments completed!")
        print(f"📁 Results saved to: {results_file}")

        return all_results

    def plot_mrr_vs_communication(self, results: list, save_plot: bool = True):
        """Create MRR vs Communication Cost plot (Figure 9 style)"""

        print("\n📈 Creating MRR vs Communication Cost plot...")

        # Group results by optimization level
        optimization_groups = {}
        for result in results:
            opt = result["optimization"]
            if opt not in optimization_groups:
                optimization_groups[opt] = []
            optimization_groups[opt].append(result)

        # Create plot
        plt.figure(figsize=(10, 8))

        colors = {"basic": "red", "url_cluster": "blue", "pca": "green"}
        markers = {"basic": "o", "url_cluster": "s", "pca": "^"}

        for opt_name, opt_results in optimization_groups.items():
            # Sort by communication cost
            opt_results.sort(key=lambda x: x["total_comm_mb"])

            comm_costs = [r["total_comm_mb"] for r in opt_results if r["mrr_100"] > 0]
            mrr_scores = [r["mrr_100"] for r in opt_results if r["mrr_100"] > 0]
            cluster_counts = [
                r["num_clusters"] for r in opt_results if r["mrr_100"] > 0
            ]

            if comm_costs and mrr_scores:
                plt.scatter(
                    comm_costs,
                    mrr_scores,
                    color=colors.get(opt_name, "black"),
                    marker=markers.get(opt_name, "o"),
                    s=100,
                    alpha=0.7,
                    label=f"{opt_name.replace('_', ' ').title()}",
                )

                # Add cluster count annotations
                for i, (comm, mrr, clusters) in enumerate(
                    zip(comm_costs, mrr_scores, cluster_counts)
                ):
                    plt.annotate(
                        f"{clusters}c",
                        (comm, mrr),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.7,
                    )

        plt.xlabel("Total Communication Cost (MB)", fontsize=12)
        plt.ylabel("MRR@100", fontsize=12)
        plt.title(
            "Quality vs Communication Cost Trade-off\n(Full Multi-Cluster Search)",
            fontsize=14,
        )
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add summary text
        total_queries = results[0]["total_queries"] if results else 0
        plt.figtext(
            0.02, 0.02, f"Total Queries: {total_queries}", fontsize=10, alpha=0.7
        )

        if save_plot:
            plot_file = self.results_dir / "mrr_vs_communication_plot.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            print(f"📊 Plot saved to: {plot_file}")

            # Also save as PDF
            pdf_file = self.results_dir / "mrr_vs_communication_plot.pdf"
            plt.savefig(pdf_file, bbox_inches="tight")
            print(f"📄 PDF saved to: {pdf_file}")

        plt.show()

        # Print numerical results
        print("\n📋 Full Multi-Cluster Results Summary:")
        print(
            f"{'Config':<15} {'Clusters':<8} {'MRR@100':<8} {'Comm(MB)':<10} {'Efficiency':<10}"
        )
        print("-" * 65)

        for result in sorted(results, key=lambda x: x["total_comm_mb"]):
            if result["mrr_100"] > 0 and result["total_comm_mb"] > 0:
                efficiency = result["mrr_100"] / result["total_comm_mb"]
                print(
                    f"{result['optimization']:<15} {result['num_clusters']:<8} "
                    f"{result['mrr_100']:<8.4f} {result['total_comm_mb']:<10.2f} {efficiency:<10.4f}"
                )


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Full multi-cluster experiments")
    parser.add_argument(
        "--base_dir", type=str, default="/home/azureuser", help="Base directory"
    )
    parser.add_argument(
        "--clusters",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Cluster counts to test",
    )
    parser.add_argument(
        "--optimizations",
        type=str,
        nargs="+",
        default=["basic", "pca"],
        help="Optimization levels to test",
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Only create plot from existing results",
    )

    args = parser.parse_args()

    experiment = FullMultiClusterExperiment(base_dir=args.base_dir)

    if args.plot_only:
        # Load existing results and plot
        results_file = experiment.results_dir / "full_experiment_results.json"
        if results_file.exists():
            with open(results_file, "r") as f:
                results = json.load(f)
            experiment.plot_mrr_vs_communication(results)
        else:
            print(f"❌ No existing results found at {results_file}")
    else:
        # Run full experiments
        results = experiment.run_full_experiments(
            cluster_configs=args.clusters, optimization_levels=args.optimizations
        )

        if results:
            experiment.plot_mrr_vs_communication(results)
        else:
            print("❌ No results to plot")


if __name__ == "__main__":
    main()
