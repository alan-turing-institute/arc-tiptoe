"""
Simple subset pipeline - supports single and multi-cluster experiments
"""

import argparse
import json
import subprocess
import time
from pathlib import Path


class SimpleQuickPipeline:
    """Simple pipeline that just reduces query/doc count"""

    def __init__(
        self,
        base_dir: str = "/home/azureuser",
        subset_queries: int = 10,
        docs_per_cluster: int = 50,
        num_clusters: int = 1,
    ):
        self.base_dir = Path(base_dir)
        self.subset_queries = subset_queries
        self.docs_per_cluster = docs_per_cluster
        self.num_clusters = num_clusters
        self.quick_dir = self.base_dir / "quick_data"

    def create_subset_queries_only(self):
        """Create smaller query file - use queries that have qrels"""
        print(f"Creating subset with {self.subset_queries} queries that have qrels...")

        self.quick_dir.mkdir(exist_ok=True)

        # Use the qrels-based query creation
        from use_qrels_queries import create_subset_with_qrels

        query_file = create_subset_with_qrels(str(self.base_dir), self.subset_queries)

        if query_file and Path(query_file).exists():
            print("✅ Created queries with guaranteed qrels entries")
            return Path(query_file)
        else:
            # Fallback to original method
            print("⚠️  Falling back to original query creation...")
            return self._create_original_queries()

    def _create_original_queries(self):
        """Original query creation method (fallback)"""
        target_queries = self.quick_dir / "quick_queries.tsv"

        # Create dummy queries for testing (but warn about MRR)
        with open(target_queries, "w", encoding="utf-8") as f:
            for i in range(self.subset_queries):
                f.write(f"{i+1000}\theart disease symptoms test query {i+1}\n")
        print(f"⚠️  Created {self.subset_queries} dummy queries (MRR will be 0)")

        return target_queries

    def create_quick_config(self, query_file: str, optimization: str = "basic") -> str:
        """Create config that uses existing data with fewer queries"""

        # Base config
        config = {
            "pca_components_file": str(
                self.base_dir / "data" / "embeddings" / "pca_components_192.txt"
            ),
            "query_file": str(query_file),
            "cluster_file_location": str(self.base_dir / "data" / "clusters") + "/",
            "url_bundle_base_dir": str(self.base_dir / "data" / "clusters") + "/",
            "index_file": str(
                self.base_dir / "data" / "artifact" / "dim192" / "index.faiss"
            ),
            "is_text": True,
            "run_msmarco_dev_queries": True,
            "filter_badwords": False,
            "short_exp": True,
            "max_docs_per_cluster": self.docs_per_cluster,
            # THIS is the key parameter that controls how many clusters to search
            "num_clusters_to_search": self.num_clusters,  # NEW: Number of clusters to search per query
            # These are different - total clusters available in the system
            "total_num_clusters": 1000,  # Total clusters in the dataset
        }

        # Apply optimization settings
        if optimization == "basic":
            config.update(
                {
                    "run_pca": False,
                    "run_url_filter": False,
                    "url_filter_by_cluster": False,
                }
            )
        elif optimization == "url_cluster":
            config.update(
                {
                    "run_pca": False,
                    "run_url_filter": True,
                    "url_filter_by_cluster": True,
                }
            )
        elif optimization == "pca":
            config.update(
                {
                    "run_pca": True,
                    "run_url_filter": True,
                    "url_filter_by_cluster": True,
                }
            )

        # Save config
        config_name = f"quick_config_{self.num_clusters}c_{optimization}.json"
        config_file = self.quick_dir / config_name

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        print(f"✅ Config created: {config_file}")
        print(f"   Clusters to search per query: {self.num_clusters}")
        print(f"   Optimization: {optimization}")
        print(f"   PCA: {config['run_pca']}")
        print(f"   URL clustering: {config['run_url_filter']}")

        return str(config_file)

    def setup(self, optimization: str = "basic"):
        """Simple setup - create query subset and config"""
        print(
            f"=== Simple Quick Setup ({self.num_clusters} clusters, {optimization}) ==="
        )

        query_file = self.create_subset_queries_only()
        config_file = self.create_quick_config(query_file, optimization)

        print("\n✅ Quick setup complete!")
        print(f"   Queries: {self.subset_queries}")
        print(f"   Docs per cluster: {self.docs_per_cluster}")
        print(f"   Clusters: {self.num_clusters}")
        print(f"   Optimization: {optimization}")
        print("   Using existing: embeddings, clusters, PCA, FAISS index")

        return config_file


def run_quick_performance_test(
    preamble: str,
    num_embed_servers: int = 2,
    num_url_servers: int = 1,
    results_dir: str = None,
    experiment_name: str = "quick",
):
    """Run performance test with existing preprocessing"""
    print(f"=== Quick Performance Test ({experiment_name}) ===")

    import runExp_refactor

    cluster = runExp_refactor.LocalTiptoeCluster(
        num_embed_servers=num_embed_servers,
        num_url_servers=num_url_servers,
        preamble=preamble,
        image_search=False,
    )

    try:
        print("Starting servers...")
        cluster.start_embedding_servers()
        cluster.start_url_servers()
        cluster.start_coordinator()

        print("Running latency test...")
        latency_results = cluster.run_latency_experiment()

        print("✅ Performance test completed")

        # Save results
        if results_dir:
            results_path = Path(results_dir)
            results_path.mkdir(exist_ok=True)

            latency_file = results_path / f"{experiment_name}_latency.log"
            with open(latency_file, "w", encoding="utf-8") as f:
                f.write(latency_results or "No results captured")

            print(f"📁 Results saved to: {latency_file}")

        return latency_results

    except (OSError, subprocess.SubprocessError, ConnectionError, TimeoutError) as e:
        print(f"❌ Performance test failed: {e}")
        return None
    finally:
        cluster.cleanup()


def run_quick_quality_test(
    config_file: str, results_dir: str = None, experiment_name: str = "quick"
):
    """Run quality test with subset queries"""
    print(f"=== Quick Quality Test ({experiment_name}) ===")

    # Find search.py
    search_dirs = ["cluster/kmeans", "clustering"]
    search_py_dir = None

    for dir_path in search_dirs:
        if (Path(dir_path) / "search.py").exists():
            search_py_dir = dir_path
            break

    if not search_py_dir:
        print("❌ Could not find search.py")
        return None

    cmd = ["python3", "search.py", config_file]

    try:
        result = subprocess.run(
            cmd,
            cwd=search_py_dir,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

        if result.returncode == 0:
            print("✅ Quality test completed")

            # Save results
            if results_dir:
                results_path = Path(results_dir)
                results_path.mkdir(exist_ok=True)

                quality_file = results_path / f"{experiment_name}_quality.log"
                with open(quality_file, "w", encoding="utf-8") as f:
                    f.write(result.stdout)

                print(f"📁 Quality results saved to: {quality_file}")

                # Show summary
                lines = result.stdout.split("\n")
                query_count = len([line for line in lines if "Query:" in line])
                result_count = len(
                    [
                        line
                        for line in lines
                        if line.strip() and not line.startswith("Query:")
                    ]
                )
                print(
                    f"📊 Processed {query_count} queries, {result_count} total results"
                )

            return result.stdout
        else:
            print(f"❌ Quality test failed: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        print("❌ Quality test timed out")
        return None


def run_multi_cluster_experiments(
    base_dir: str,
    subset_queries: int = 10,
    cluster_configs: list = None,
    optimization_levels: list = None,
):
    """Run experiments across multiple cluster configurations"""

    if cluster_configs is None:
        cluster_configs = [1, 4]  # Default quick test

    if optimization_levels is None:
        optimization_levels = ["basic", "pca"]  # Default quick test

    print("🚀 Starting quick multi-cluster experiments...")
    print(f"   Cluster configs: {cluster_configs}")
    print(f"   Optimizations: {optimization_levels}")

    results_summary = []
    base_path = Path(base_dir)
    multi_results_dir = base_path / "quick_data" / "multi_cluster_results"
    multi_results_dir.mkdir(parents=True, exist_ok=True)

    for num_clusters in cluster_configs:
        for optimization in optimization_levels:
            experiment_name = f"{num_clusters}c_{optimization}"

            print(f"\n🔬 Running experiment: {experiment_name}")

            try:
                # Setup pipeline for this configuration
                pipeline = SimpleQuickPipeline(
                    base_dir=base_dir,
                    subset_queries=subset_queries,
                    num_clusters=num_clusters,
                )

                config_file = pipeline.setup(optimization)

                # Create experiment-specific results directory
                exp_results_dir = multi_results_dir / experiment_name
                exp_results_dir.mkdir(exist_ok=True)

                # Scale servers based on cluster count
                num_embed_servers = min(4, max(2, num_clusters))
                num_url_servers = min(2, max(1, num_clusters // 2))

                experiment_result = {
                    "experiment_name": experiment_name,
                    "num_clusters": num_clusters,
                    "optimization": optimization,
                    "num_embed_servers": num_embed_servers,
                    "num_url_servers": num_url_servers,
                }

                # Run performance test
                print("  📊 Running performance test...")
                latency_results = run_quick_performance_test(
                    preamble=base_dir,
                    num_embed_servers=num_embed_servers,
                    num_url_servers=num_url_servers,
                    results_dir=str(exp_results_dir),
                    experiment_name=experiment_name,
                )

                if latency_results:
                    print("  ✅ Performance: completed")
                else:
                    print("  ⚠️  Performance: failed")

                # Run quality test
                print("  🔍 Running quality test...")
                quality_results = run_quick_quality_test(
                    config_file=config_file,
                    results_dir=str(exp_results_dir),
                    experiment_name=experiment_name,
                )

                if quality_results:
                    print("  ✅ Quality: completed")
                else:
                    print("  ⚠️  Quality: failed")

                # Save experiment metadata
                experiment_result["success"] = bool(latency_results and quality_results)

                results_summary.append(experiment_result)

                # Brief pause between experiments
                time.sleep(2)

            except (
                OSError,
                subprocess.SubprocessError,
                ConnectionError,
                TimeoutError,
                FileNotFoundError,
                PermissionError,
            ) as e:
                print(f"  ❌ Experiment {experiment_name} failed: {e}")
                experiment_result["success"] = False
                experiment_result["error"] = str(e)
                results_summary.append(experiment_result)

    # Save summary
    summary_file = multi_results_dir / "experiments_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    print("\n🎉 Multi-cluster experiments complete!")
    print(f"📁 Results saved to: {multi_results_dir}")
    print(f"📄 Summary: {summary_file}")

    # Quick analysis
    print("\n📊 Quick Results Summary:")
    print(f"{'Experiment':<15} {'Success':<8} {'Servers':<12}")
    print("-" * 40)

    for result in results_summary:
        name = result["experiment_name"]
        success = "✅" if result["success"] else "❌"
        servers = (
            f"{result.get('num_embed_servers', 0)}e+{result.get('num_url_servers', 0)}u"
        )
        print(f"{name:<15} {success:<8} {servers:<12}")

    return results_summary, multi_results_dir


def analyze_multi_cluster_results(results_dir: str):
    """Analyze multi-cluster experiment results"""
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return

    print(f"\n📊 Analyzing multi-cluster results from: {results_dir}")

    # Use the existing analyzer
    from analyse_results import TiptoeAnalyser

    # Find all experiment directories
    exp_dirs = [d for d in results_path.iterdir() if d.is_dir() and "c_" in d.name]

    if not exp_dirs:
        print("❌ No experiment directories found")
        return

    print(f"Found {len(exp_dirs)} experiments:")

    all_results = []

    for exp_dir in sorted(exp_dirs):
        print(f"\n🔍 Analyzing: {exp_dir.name}")

        # Analyze this experiment
        analyzer = TiptoeAnalyser(str(exp_dir))
        exp_results = analyzer.analyze_all_results()

        # Extract key metrics
        latency = exp_results.get("single_cluster", {}).get("latency", {})
        quality = exp_results.get("single_cluster", {}).get("quality", {})

        summary = {
            "name": exp_dir.name,
            "mrr_100": quality.get("mrr_100", 0),
            "total_comm_mb": latency.get("total_comm_mb", 0),
            "avg_latency": latency.get("avg_latency", 0),
            "queries": quality.get("queries_evaluated", 0),
        }

        all_results.append(summary)

        print(f"  MRR@100: {summary['mrr_100']:.4f}")
        print(f"  Communication: {summary['total_comm_mb']:.2f} MB")
        print(f"  Latency: {summary['avg_latency']:.3f}s")

    # Summary table
    print("\n📋 Multi-Cluster Results Summary:")
    print(f"{'Experiment':<15} {'MRR@100':<8} {'Comm(MB)':<10} {'Latency(s)':<10}")
    print("-" * 50)

    for result in sorted(all_results, key=lambda x: x["total_comm_mb"]):
        name = result["name"]
        mrr = result["mrr_100"]
        comm = result["total_comm_mb"]
        latency = result["avg_latency"]
        print(f"{name:<15} {mrr:<8.4f} {comm:<10.2f} {latency:<10.3f}")

    # Quality vs Communication insights
    print("\n💡 Quality vs Communication Trade-offs:")
    valid_results = [
        r for r in all_results if r["mrr_100"] > 0 and r["total_comm_mb"] > 0
    ]

    if valid_results:
        for result in sorted(valid_results, key=lambda x: x["total_comm_mb"]):
            efficiency = (
                result["mrr_100"] / result["total_comm_mb"]
                if result["total_comm_mb"] > 0
                else 0
            )
            print(
                f"  {result['name']}: {result['mrr_100']:.4f} MRR @ {result['total_comm_mb']:.2f} MB (efficiency: {efficiency:.4f})"
            )
    else:
        print("  ⚠️  No valid results for comparison")


def show_quick_results(results_dir: str):
    """Show a quick summary of results"""
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"❌ Results directory doesn't exist: {results_dir}")
        return

    print("\n=== Quick Results Summary ===")

    # Check latency results
    latency_files = list(results_path.glob("*latency*.log"))
    if latency_files:
        print("📊 Performance Results:")
        for latency_file in latency_files[:3]:  # Show first 3
            print(f"   📄 {latency_file.name}")

    # Check quality results
    quality_files = list(results_path.glob("*quality*.log"))
    if quality_files:
        print("🔍 Quality Results:")
        for quality_file in quality_files[:3]:  # Show first 3
            with open(quality_file, "r", encoding="utf-8") as f:
                content = f.read()

            lines = content.split("\n")
            query_count = len([line for line in lines if "Query:" in line])
            result_lines = [
                line for line in lines if line.strip() and not line.startswith("Query:")
            ]

            print(
                f"   📄 {quality_file.name}: {query_count} queries, {len(result_lines)} results"
            )


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Simple quick experiments with multi-cluster support"
    )
    parser.add_argument(
        "--subset_queries", type=int, default=10, help="Number of queries (default: 10)"
    )
    parser.add_argument(
        "--docs_per_cluster",
        type=int,
        default=50,
        help="Max docs per cluster (default: 50)",
    )
    parser.add_argument(
        "--num_clusters", type=int, default=1, help="Number of clusters (default: 1)"
    )
    parser.add_argument(
        "--optimization",
        type=str,
        choices=["basic", "url_cluster", "pca"],
        default="basic",
        help="Optimization level (default: basic)",
    )
    parser.add_argument("--performance_only", action="store_true")
    parser.add_argument("--quality_only", action="store_true")
    parser.add_argument("--preamble", type=str, default="/home/azureuser")

    # Multi-cluster experiment options
    parser.add_argument(
        "--multi_cluster", action="store_true", help="Run multi-cluster experiments"
    )
    parser.add_argument(
        "--clusters",
        type=int,
        nargs="+",
        default=[1, 4],
        help="Cluster configurations to test",
    )
    parser.add_argument(
        "--optimizations",
        type=str,
        nargs="+",
        default=["basic", "pca"],
        help="Optimizations to test",
    )
    parser.add_argument(
        "--analyze_multi",
        action="store_true",
        help="Analyze existing multi-cluster results",
    )

    args = parser.parse_args()

    # Multi-cluster experiments
    if args.multi_cluster:
        _, results_dir = run_multi_cluster_experiments(
            base_dir=args.preamble,
            subset_queries=args.subset_queries,
            cluster_configs=args.clusters,
            optimization_levels=args.optimizations,
        )

        # Auto-analyze results
        analyze_multi_cluster_results(str(results_dir))
        return

    # Analyze existing multi-cluster results
    if args.analyze_multi:
        results_dir = Path(args.preamble) / "quick_data" / "multi_cluster_results"
        analyze_multi_cluster_results(str(results_dir))
        return

    # Single experiment (original functionality)
    pipeline = SimpleQuickPipeline(
        base_dir=args.preamble,
        subset_queries=args.subset_queries,
        docs_per_cluster=args.docs_per_cluster,
        num_clusters=args.num_clusters,
    )
    config_file = pipeline.setup(args.optimization)

    # Create results directory
    results_dir = pipeline.quick_dir / "results"
    results_dir.mkdir(exist_ok=True)

    experiment_name = f"{args.num_clusters}c_{args.optimization}"

    # Run experiments
    if not args.quality_only:
        print("\n" + "=" * 50)
        run_quick_performance_test(
            args.preamble, results_dir=str(results_dir), experiment_name=experiment_name
        )

    if not args.performance_only:
        print("\n" + "=" * 50)
        run_quick_quality_test(
            config_file, results_dir=str(results_dir), experiment_name=experiment_name
        )

    print("\n🎉 Quick experiments complete!")
    print(f"📁 Results in: {results_dir}")

    # Show what was created
    if results_dir.exists():
        print("\nFiles created:")
        for file in results_dir.glob("*"):
            size = file.stat().st_size
            print(f"  📄 {file.name} ({size} bytes)")

    print("\nQuick commands:")
    print("  # Single experiments:")
    print(
        "  python3 subset_test.py --subset_queries 5 --num_clusters 4 --optimization pca"
    )
    print("  python3 subset_test.py --performance_only --num_clusters 1")
    print("")
    print("  # Multi-cluster experiments:")
    print(
        "  python3 subset_test.py --multi_cluster --clusters 1 4 --optimizations basic pca"
    )
    print("  python3 subset_test.py --analyze_multi")
    print("")
    print("  # Analysis:")
    print(f"  python3 analyse_results.py --results_dir {results_dir}")

    show_quick_results(str(results_dir))


if __name__ == "__main__":
    main()
