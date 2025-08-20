"""
Simple subset pipeline - just fewer queries and docs, reuse existing preprocessing
"""

import argparse
import json
import subprocess
from pathlib import Path


class SimpleQuickPipeline:
    """Simple pipeline that just reduces query/doc count"""

    def __init__(
        self,
        base_dir: str = "/home/azureuser",
        subset_queries: int = 10,
        docs_per_cluster: int = 50,
    ):
        self.base_dir = Path(base_dir)
        self.subset_queries = subset_queries
        self.docs_per_cluster = docs_per_cluster
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
        # Your existing create_subset_queries_only code here
        target_queries = self.quick_dir / "quick_queries.tsv"

        # Create dummy queries for testing (but warn about MRR)
        with open(target_queries, "w", encoding="utf-8") as f:
            for i in range(self.subset_queries):
                f.write(f"{i+1000}\theart disease symptoms test query {i+1}\n")
        print(f"⚠️  Created {self.subset_queries} dummy queries (MRR will be 0)")

        return target_queries

    def create_quick_config(self, query_file: str) -> str:
        """Create config that uses existing data with fewer queries"""
        config = {
            # Use all existing preprocessing - just point to original files
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
            "run_pca": True,
            "run_url_filter": False,
            "url_filter_by_cluster": False,
            "run_msmarco_dev_queries": True,
            "filter_badwords": False,
            "short_exp": True,
            # Limit documents per cluster for faster quality tests
            "max_docs_per_cluster": self.docs_per_cluster,
        }

        config_file = self.quick_dir / "quick_config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        print(f"✅ Config created: {config_file}")
        return str(config_file)

    def setup(self):
        """Simple setup - just create query subset"""
        print("=== Simple Quick Setup ===")

        query_file = self.create_subset_queries_only()
        config_file = self.create_quick_config(query_file)

        print("\n✅ Quick setup complete!")
        print(f"   Queries: {self.subset_queries}")
        print(f"   Docs per cluster: {self.docs_per_cluster}")
        print("   Using existing: embeddings, clusters, PCA, FAISS index")

        return config_file


def run_quick_performance_test(
    preamble: str,
    num_embed_servers: int = 2,
    num_url_servers: int = 1,
    results_dir: str = None,
):
    """Run performance test with existing preprocessing"""
    print("=== Quick Performance Test ===")

    import runExp_refactor

    cluster = runExp_refactor.LocalTiptoeCluster(
        num_embed_servers=num_embed_servers,
        num_url_servers=num_url_servers,
        preamble=preamble,  # Use original data directory
        image_search=False,
    )

    # Set custom results directory if provided
    if results_dir:
        cluster.results_dir = results_dir

    try:
        print("Starting servers...")
        cluster.start_embedding_servers()
        cluster.start_url_servers()
        cluster.start_coordinator()

        print("Running latency test...")
        latency_results = cluster.run_latency_experiment()

        print("✅ Performance test completed")

        # Also save to quick results directory
        if results_dir:
            results_path = Path(results_dir)
            results_path.mkdir(exist_ok=True)

            # Copy the latency results to quick directory
            latency_file = results_path / "quick_latency.log"
            with open(latency_file, "w", encoding="utf-8") as f:
                f.write(latency_results or "No results captured")

            print(f"📁 Results saved to: {latency_file}")

        return latency_results

    except (OSError, subprocess.SubprocessError, ConnectionError, TimeoutError) as e:
        print(f"❌ Performance test failed: {e}")
        return None
    finally:
        cluster.cleanup()


def run_quick_quality_test(config_file: str, results_dir: str = None):
    """Run quality test with subset queries"""
    print("=== Quick Quality Test ===")

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
            timeout=120,  # Shorter timeout for quick test
            check=False,
        )

        if result.returncode == 0:
            print("✅ Quality test completed")

            # Save results if directory provided
            if results_dir:
                results_path = Path(results_dir)
                results_path.mkdir(exist_ok=True)

                quality_file = results_path / "quick_quality.log"
                with open(quality_file, "w", encoding="utf-8") as f:
                    f.write(result.stdout)

                print(f"📁 Quality results saved to: {quality_file}")

                # Show summary of what was processed
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


def show_quick_results(results_dir: str):
    """Show a quick summary of results"""
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"❌ Results directory doesn't exist: {results_dir}")
        return

    print("\n=== Quick Results Summary ===")

    # Check latency results
    latency_file = results_path / "quick_latency.log"
    if latency_file.exists():
        with open(latency_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract key metrics
        lines = content.split("\n")
        print("📊 Performance Results:")

        # Look for timing info
        for line in lines[:10]:  # First 10 lines usually have summary
            if any(
                word in line.lower() for word in ["latency", "time", "ms", "seconds"]
            ):
                print(f"   {line.strip()}")

    # Check quality results
    quality_file = results_path / "quick_quality.log"
    if quality_file.exists():
        with open(quality_file, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.split("\n")
        query_count = len([line for line in lines if "Query:" in line])
        result_lines = [
            line for line in lines if line.strip() and not line.startswith("Query:")
        ]

        print("🔍 Quality Results:")
        print(f"   Queries processed: {query_count}")
        print(f"   Results returned: {len(result_lines)}")

        # Show first few results
        print("   Sample results:")
        for line in result_lines[:5]:
            if line.strip():
                print(f"     {line.strip()}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Simple quick experiments")
    parser.add_argument(
        "--subset_queries", type=int, default=10, help="Number of queries (default: 10)"
    )
    parser.add_argument(
        "--docs_per_cluster",
        type=int,
        default=50,
        help="Max docs per cluster (default: 50)",
    )
    parser.add_argument("--performance_only", action="store_true")
    parser.add_argument("--quality_only", action="store_true")
    parser.add_argument("--preamble", type=str, default="/home/azureuser")

    args = parser.parse_args()

    # Setup
    pipeline = SimpleQuickPipeline(
        base_dir=args.preamble,
        subset_queries=args.subset_queries,
        docs_per_cluster=args.docs_per_cluster,
    )
    config_file = pipeline.setup()

    # Create results directory
    results_dir = pipeline.quick_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Run experiments
    if not args.quality_only:
        print("\n" + "=" * 50)
        run_quick_performance_test(args.preamble, results_dir=str(results_dir))

    if not args.performance_only:
        print("\n" + "=" * 50)
        run_quick_quality_test(config_file, results_dir=str(results_dir))

    print("\n🎉 Quick experiments complete!")
    print(f"📁 Results in: {results_dir}")

    # Show what was created
    if results_dir.exists():
        print("\nFiles created:")
        for file in results_dir.glob("*"):
            size = file.stat().st_size
            print(f"  📄 {file.name} ({size} bytes)")

    print("\nTo iterate:")
    print("  python3 subset_test.py --subset_queries 5 --performance_only")
    print("  python3 subset_test.py --subset_queries 5 --quality_only")

    # Quick analysis
    print("\nQuick analysis:")
    print(f"  cd analysis && python3 analyse_results.py --results_dir {results_dir}")

    show_quick_results(results_dir)


if __name__ == "__main__":
    main()
