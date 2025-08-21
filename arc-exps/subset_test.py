"""
Simple subset pipeline - supports single and multi-cluster experiments
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import runExp_refactor
from analyse_results import TiptoeAnalyser
from use_qrels_queries import create_subset_with_qrels


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

        # Base config - match the format expected by clustering/search.py
        config = {
            "PCA_COMPONENTS_FILE": str(
                self.base_dir / "data" / "embeddings" / "pca_components_192.txt"
            ),
            "QUERY_FILE": str(query_file),
            "CLUSTER_FILE_LOCATION": str(self.base_dir / "data" / "clusters") + "/",
            "URL_BUNDLE_BASE_DIR": str(self.base_dir / "data" / "clusters") + "/",
            "INDEX_FILE": str(
                self.base_dir / "data" / "artifact" / "dim192" / "index.faiss"
            ),
            "IS_TEXT": True,
            "RUN_MSMARCO_DEV_QUERIES": True,
            "FILTER_BADWORDS": False,
            "SHORT_EXP": True,
            "MAX_DOCS_PER_CLUSTER": self.docs_per_cluster,
            # KEY: This controls how many clusters to search per query
            "NUM_CLUSTERS": self.num_clusters,  # This is what clustering/search.py uses
            # Add other required globals that clustering/search.py expects
            "CENTROIDS_FILE": str(
                self.base_dir / "data" / "embeddings" / "centroids.txt"
            ),
            "BADWORDS_FILE": None,
            "IMG_RESULTS_DIR": None,
        }

        # Apply optimization settings using the correct keys for clustering/search.py
        if optimization == "basic":
            config.update(
                {
                    "RUN_PCA": False,
                    "RUN_URL_FILTER": False,
                    "URL_FILTER_BY_CLUSTER": False,
                }
            )
        elif optimization == "url_cluster":
            config.update(
                {
                    "RUN_PCA": False,
                    "RUN_URL_FILTER": True,
                    "URL_FILTER_BY_CLUSTER": True,
                }
            )
        elif optimization == "pca":
            config.update(
                {
                    "RUN_PCA": True,
                    "RUN_URL_FILTER": True,
                    "URL_FILTER_BY_CLUSTER": True,
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
        print(f"   PCA: {config['RUN_PCA']}")
        print(f"   URL clustering: {config['RUN_URL_FILTER']}")

        return str(config_file)


def run_quick_quality_test(
    config_file: str, results_dir: str = None, experiment_name: str = "quick"
):
    """Run quality test using clustering/search.py directly"""
    print(f"=== Quick Quality Test ({experiment_name}) ===")

    # Use clustering/search.py directly - this is the real multi-cluster search
    search_script = Path("clustering") / "search.py"

    if not search_script.exists():
        print("❌ Could not find clustering/search.py")
        return None

    print("🔍 Using clustering/search.py for multi-cluster search")
    print(f"   Config: {config_file}")

    # Verify config file exists and is valid
    if not Path(config_file).exists():
        print(f"❌ Config file not found: {config_file}")
        return None

    try:
        with open(config_file, "r") as f:
            config = json.load(f)
        print(f"   Clusters to search: {config.get('NUM_CLUSTERS', 'unknown')}")
        print(f"   Query file: {config.get('QUERY_FILE', 'unknown')}")
        print(f"   PCA enabled: {config.get('RUN_PCA', 'unknown')}")
    except Exception as e:
        print(f"⚠️  Could not read config: {e}")

    cmd = ["python3", "search.py", config_file]
    cwd = "clustering"

    try:
        print(f"   Running: {' '.join(cmd)} (in {cwd})")
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300,  # Longer timeout for multi-cluster
            check=False,
        )

        print(f"   Return code: {result.returncode}")
        print(f"   Stdout length: {len(result.stdout)} chars")
        print(f"   Stderr length: {len(result.stderr)} chars")

        if result.returncode == 0:
            if result.stdout.strip():
                print("✅ Quality test completed with results")

                # Show first few lines of output for debugging
                lines = result.stdout.split("\n")
                print("   First 10 lines of output:")
                for i, line in enumerate(lines[:10]):
                    if line.strip():
                        print(f"     {i+1}: {line[:100]}")

                # Save results
                if results_dir:
                    results_path = Path(results_dir)
                    results_path.mkdir(exist_ok=True)

                    quality_file = results_path / f"{experiment_name}_quality.log"
                    with open(quality_file, "w", encoding="utf-8") as f:
                        f.write(result.stdout)

                    print(f"📁 Quality results saved to: {quality_file}")
                    print(f"   File size: {quality_file.stat().st_size} bytes")

                    # Enhanced summary for multi-cluster
                    lines = result.stdout.split("\n")
                    query_count = len([line for line in lines if "Query:" in line])
                    result_count = len(
                        [
                            line
                            for line in lines
                            if line.strip()
                            and not line.startswith("Query:")
                            and not line.startswith("---------------")
                            and len(line.split()) >= 2
                        ]
                    )

                    # Parse cluster information from stderr
                    stderr_lines = result.stderr.split("\n")
                    cluster_info = [
                        line for line in stderr_lines if "Nearest clusters:" in line
                    ]

                    print(
                        f"📊 Processed {query_count} queries, {result_count} total results"
                    )
                    if cluster_info:
                        print(
                            f"🔍 Multi-cluster info: {cluster_info[0] if cluster_info else 'N/A'}"
                        )

                return result.stdout
            else:
                print("❌ Quality test completed but produced no output")
                print("   This suggests the search isn't finding any results")
                if result.stderr:
                    print(f"   Stderr: {result.stderr[:500]}")
                return None
        else:
            print(f"❌ Quality test failed: return code {result.returncode}")
            print(f"   Stderr: {result.stderr}")
            print(f"   Stdout: {result.stdout}")
            return None

    except subprocess.TimeoutExpired:
        print("❌ Quality test timed out")
        return None
    except Exception as e:
        print(f"❌ Quality test error: {e}")
        return None


def run_quick_performance_test(
    preamble: str,
    num_embed_servers: int = 2,
    num_url_servers: int = 1,
    results_dir: str = None,
    experiment_name: str = "quick",
):
    """Run performance test with existing preprocessing"""
    print(f"=== Quick Performance Test ({experiment_name}) ===")

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

                # Create subset queries and config
                query_file = pipeline.create_subset_queries_only()
                config_file = pipeline.create_quick_config(
                    str(query_file), optimization
                )

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
    # Single experiment (original functionality)
    pipeline = SimpleQuickPipeline(
        base_dir=args.preamble,
        subset_queries=args.subset_queries,
        docs_per_cluster=args.docs_per_cluster,
        num_clusters=args.num_clusters,
    )
    query_file = pipeline.create_subset_queries_only()
    config_file = pipeline.create_quick_config(str(query_file), args.optimization)

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
