"""
Enhanced analysis with better pattern matching and debugging for quick iteration
"""

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Dict

# Import the MRR computation module
try:
    from compute_mrr import enhance_quality_metrics

    MRR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  MRR computation not available: {e}")
    MRR_AVAILABLE = False


class TiptoeAnalyser:
    """Better analysis"""

    def __init__(self, results_dir: str = "/home/azureuser/experiment_results"):
        self.results_dir = results_dir

    def parse_latency_log(self, log_file: str) -> Dict:
        """Parse latency log to extract communication costs and timing"""
        metrics = {
            "avg_latency": 0.0,
            "upload_mb": 0.0,
            "download_mb": 0.0,
            "total_comm_mb": 0.0,
            "queries_processed": 0,
        }

        if not os.path.exists(log_file):
            print(f"❌ Latency log not found: {log_file}")
            return metrics

        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        print(f"🔍 Parsing latency log: {os.path.basename(log_file)}")

        # Extract communication costs
        upload_matches = re.findall(r"Upload:\s+([\d.]+)\s+MB", content, re.IGNORECASE)
        download_matches = re.findall(
            r"Download:\s+([\d.]+)\s+MB", content, re.IGNORECASE
        )

        if upload_matches and download_matches:
            metrics["upload_mb"] = float(upload_matches[0])
            metrics["download_mb"] = float(download_matches[0])
            metrics["total_comm_mb"] = metrics["upload_mb"] + metrics["download_mb"]
            print(f"  📡 Communication: {metrics['total_comm_mb']:.2f} MB total")

        # Look for timing information
        timing_patterns = [
            r"Query latency:\s+([\d.]+)s",
            r"Total time:\s+([\d.]+)s",
            r"(\d+\.\d+)s",  # Generic seconds pattern
        ]

        for pattern in timing_patterns:
            latency_matches = re.findall(pattern, content, re.IGNORECASE)
            if latency_matches:
                try:
                    latencies = [float(x) for x in latency_matches]
                    metrics["avg_latency"] = sum(latencies) / len(latencies)
                    metrics["queries_processed"] = len(latencies)
                    print(f"  ⏱️  Average latency: {metrics['avg_latency']:.3f}s")
                    break
                except ValueError:
                    continue

        return metrics

    def parse_quality_log(self, log_file: str) -> Dict:
        """Parse quality log and compute actual MRR using qrels"""

        # Use the enhanced MRR computation
        if MRR_AVAILABLE:
            try:
                enhanced_metrics = enhance_quality_metrics(log_file)
                if enhanced_metrics and enhanced_metrics.get("mrr_100", 0) > 0:
                    print(
                        f"  ✅ Computed actual MRR@100: {enhanced_metrics.get('mrr_100', 0):.4f}"
                    )
                    return enhanced_metrics
                else:
                    print("  ⚠️  MRR computation returned no results")
            except (FileNotFoundError, KeyError, ValueError, TypeError) as e:
                print(f"  ⚠️  MRR computation failed: {e}")

        # Fallback to basic parsing
        metrics = {
            "mrr_100": 0.0,
            "mrr_10": 0.0,
            "queries_evaluated": 0,
            "total_results_returned": 0,
            "avg_results_per_query": 0.0,
            "queries_with_results": 0,
        }

        if not os.path.exists(log_file):
            print(f"❌ Quality log not found: {log_file}")
            return metrics

        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        print(f"🔍 Parsing quality log: {os.path.basename(log_file)}")

        lines = content.split("\n")

        # Count queries and results using the search.py output format
        query_count = len(
            [line for line in lines if re.match(r"Query:\s*\d+", line, re.IGNORECASE)]
        )
        result_count = len(
            [line for line in lines if re.match(r"^[\d.-]+\s+\S+", line.strip())]
        )

        metrics["queries_evaluated"] = query_count
        metrics["total_results_returned"] = result_count

        if query_count > 0:
            metrics["avg_results_per_query"] = result_count / query_count
            metrics["queries_with_results"] = query_count

        print(f"  📊 Basic parsing: {query_count} queries, {result_count} results")
        print("  ⚠️  No MRR computed (need qrels file and proper setup)")

        return metrics

    def analyze_multi_cluster_experiments(self) -> Dict:
        """Analyze multi-cluster experiment results"""

        # Determine the correct multi-cluster directory based on current context
        if "multi_cluster_results" in self.results_dir:
            # We're already in a multi-cluster directory
            multi_cluster_dirs = [Path(self.results_dir)]
        else:
            # Look for multi-cluster results directory
            multi_cluster_dirs = [
                Path(self.results_dir) / "multi_cluster_results",
                Path("/home/azureuser/quick_data/multi_cluster_results"),
                Path("/home/azureuser/full_multi_cluster_results"),
            ]

        multi_results = {}

        for multi_dir in multi_cluster_dirs:
            if not multi_dir.exists():
                continue

            print(f"🔍 Found multi-cluster results in: {multi_dir}")

            # Find experiment subdirectories (avoid infinite recursion)
            exp_dirs = []
            try:
                for d in multi_dir.iterdir():
                    if (
                        d.is_dir()
                        and "c_" in d.name
                        and not d.name.startswith(
                            "multi_cluster"
                        )  # Avoid recursive directories
                        and len(d.name.split("_")) >= 2
                    ):  # Ensure proper format like "1c_basic"
                        exp_dirs.append(d)
            except (OSError, PermissionError) as e:
                print(f"   ⚠️  Error reading directory {multi_dir}: {e}")
                continue

            if not exp_dirs:
                print(f"   No valid experiment directories found in {multi_dir}")
                continue

            print(f"   Found {len(exp_dirs)} experiments: {[d.name for d in exp_dirs]}")

            for exp_dir in exp_dirs:
                exp_name = exp_dir.name

                # Parse experiment configuration from name
                parts = exp_name.split("_")
                if len(parts) >= 2:
                    try:
                        num_clusters = int(parts[0].replace("c", ""))
                        optimization = "_".join(parts[1:])
                    except ValueError:
                        print(f"   ⚠️  Could not parse experiment name: {exp_name}")
                        continue
                else:
                    print(f"   ⚠️  Invalid experiment name format: {exp_name}")
                    continue

                # Prevent recursive analysis by checking if we're already in an experiment directory
                if exp_dir == Path(self.results_dir):
                    print(f"   ⚠️  Skipping recursive analysis of {exp_name}")
                    continue

                # Analyze this specific experiment (but prevent infinite recursion)
                try:
                    exp_analyzer = TiptoeAnalyser(str(exp_dir))
                    # Only analyze single cluster results, not multi-cluster to prevent recursion
                    exp_analysis = exp_analyzer.analyze_single_experiment()

                    # Store in multi_results
                    multi_results[exp_name] = {
                        "num_clusters": num_clusters,
                        "optimization": optimization,
                        "latency": exp_analysis.get("latency", {}),
                        "quality": exp_analysis.get("quality", {}),
                    }

                    print(
                        f"   📊 {exp_name}: "
                        f"MRR={multi_results[exp_name]['quality'].get('mrr_100', 0):.4f}, "
                        f"Comm={multi_results[exp_name]['latency'].get('total_comm_mb', 0):.2f}MB"
                    )
                except Exception as e:
                    print(f"   ❌ Error analyzing {exp_name}: {e}")
                    continue

            # Only process the first valid directory to avoid duplicates
            if multi_results:
                break

        return multi_results

    def analyze_single_experiment(self) -> Dict:
        """Analyze a single experiment directory without multi-cluster recursion"""
        results = {"latency": {}, "quality": {}}

        # Look for files directly in this directory
        latency_files = list(Path(self.results_dir).glob("*latency*.log"))
        quality_files = list(Path(self.results_dir).glob("*quality*.log"))

        # Parse latency results
        if latency_files:
            results["latency"] = self.parse_latency_log(str(latency_files[0]))

        # Parse quality results
        if quality_files:
            results["quality"] = self.parse_quality_log(str(quality_files[0]))

        return results

    def analyze_all_results(self) -> Dict:
        """Analyze all available experiment results"""
        results = {"single_cluster": {"latency": {}, "quality": {}, "throughput": {}}}

        print(f"🔍 Analyzing results in: {self.results_dir}")

        # Check if we're already in a specific experiment directory
        current_dir = Path(self.results_dir)
        if (
            current_dir.name
            and "c_" in current_dir.name
            and len(current_dir.name.split("_")) >= 2
        ):
            # We're in a specific experiment directory
            print(f"   Analyzing single experiment: {current_dir.name}")
            single_results = self.analyze_single_experiment()
            results["single_cluster"]["latency"] = single_results.get("latency", {})
            results["single_cluster"]["quality"] = single_results.get("quality", {})
            return results

        # Look in multiple locations for files
        search_dirs = [
            self.results_dir,
            "/home/azureuser/quick_data/results",
            "/home/azureuser/arc-exps",
            ".",
        ]

        latency_files = []
        quality_files = []

        for search_dir in search_dirs:
            search_path = Path(search_dir)
            if search_path.exists():
                try:
                    print(f"   🔍 Searching in: {search_dir}")

                    # Look for files directly in directory
                    dir_latency = list(search_path.glob("*latency*.log"))
                    dir_quality = list(search_path.glob("*quality*.log"))

                    # Also look in subdirectories for multi-cluster results
                    subdir_latency = list(search_path.glob("*/*latency*.log"))
                    subdir_quality = list(search_path.glob("*/*quality*.log"))

                    latency_files.extend(dir_latency + subdir_latency)
                    quality_files.extend(dir_quality + subdir_quality)

                    found_files = len(
                        dir_latency + subdir_latency + dir_quality + subdir_quality
                    )
                    if found_files > 0:
                        print(
                            f"     📄 Found {len(dir_latency + subdir_latency)} latency, {len(dir_quality + subdir_quality)} quality files"
                        )

                except (OSError, PermissionError) as e:
                    print(f"   ⚠️  Error reading {search_dir}: {e}")
                    continue

        # Remove duplicates and convert to strings
        latency_files = list(set(str(f) for f in latency_files))
        quality_files = list(set(str(f) for f in quality_files))

        print(
            f"Found {len(latency_files)} latency files, {len(quality_files)} quality files"
        )

        if latency_files:
            print(f"  📊 Latency files: {[Path(f).name for f in latency_files[:5]]}")
        if quality_files:
            print(f"  🔍 Quality files: {[Path(f).name for f in quality_files[:5]]}")

        # Find the most relevant files for multi-cluster analysis
        # Prefer files from multi-cluster experiments
        best_latency_file = None
        best_quality_file = None

        # Look for multi-cluster files first
        for lf in latency_files:
            if "multi_cluster_results" in lf and any(x in lf for x in ["2c_", "4c_"]):
                best_latency_file = lf
                break
        if not best_latency_file and latency_files:
            best_latency_file = latency_files[0]

        for qf in quality_files:
            if "multi_cluster_results" in qf and any(x in qf for x in ["2c_", "4c_"]):
                # Check if file has content
                if Path(qf).stat().st_size > 0:
                    best_quality_file = qf
                    break
        if not best_quality_file:
            # Find any non-empty quality file
            for qf in quality_files:
                if Path(qf).stat().st_size > 0:
                    best_quality_file = qf
                    break

        # Parse latency results
        if best_latency_file:
            print(f"  📊 Using latency file: {Path(best_latency_file).name}")
            results["single_cluster"]["latency"] = self.parse_latency_log(
                best_latency_file
            )

        # Parse quality results
        if best_quality_file:
            print(f"  🔍 Using quality file: {Path(best_quality_file).name}")
            results["single_cluster"]["quality"] = self.parse_quality_log(
                best_quality_file
            )
        else:
            print("  ⚠️  No non-empty quality files found")

        # Check for multi-cluster experiments (only if not already in multi-cluster context)
        if "multi_cluster_results" not in self.results_dir:
            try:
                multi_cluster_results = self.analyze_multi_cluster_experiments()
                if multi_cluster_results:
                    results["multi_cluster"] = multi_cluster_results
            except RecursionError:
                print("   ⚠️  Prevented infinite recursion in multi-cluster analysis")
            except Exception as e:
                print(f"   ⚠️  Multi-cluster analysis failed: {e}")

        return results

    def generate_summary_report(self, results: Dict):
        """Generate a comprehensive summary report"""
        print("\n" + "=" * 60)
        print("🎯 EXPERIMENT RESULTS SUMMARY")
        print("=" * 60)

        single_cluster = results.get("single_cluster", {})

        # Performance metrics
        latency = single_cluster.get("latency", {})
        if latency:
            print("\n📊 Performance Metrics:")
            print(f"   Average Latency: {latency.get('avg_latency', 0):.3f}s")
            print(f"   Upload Cost: {latency.get('upload_mb', 0):.2f} MB")
            print(f"   Download Cost: {latency.get('download_mb', 0):.2f} MB")
            print(f"   Total Communication: {latency.get('total_comm_mb', 0):.2f} MB")
            print(f"   Queries Processed: {latency.get('queries_processed', 0)}")

        # Quality metrics
        quality = single_cluster.get("quality", {})
        if quality:
            print("\n🔍 Search Quality Metrics:")
            mrr_100 = quality.get("mrr_100", 0)
            if mrr_100 > 0:
                print(f"   ✅ MRR@100: {mrr_100:.4f}")
                print(f"   ✅ MRR@10: {quality.get('mrr_10', 0):.4f}")
            else:
                print("   ⚠️  MRR@100: Not computed (need qrels)")
            print(f"   📝 Queries Evaluated: {quality.get('queries_evaluated', 0)}")
            print(f"   📄 Total Results: {quality.get('total_results_returned', 0)}")

        # Multi-cluster analysis
        multi_cluster = results.get("multi_cluster", {})
        if multi_cluster:
            print("\n🔬 MULTI-CLUSTER EXPERIMENT RESULTS")
            print("=" * 60)

            # Create comparison table
            print(
                f"{'Experiment':<15} {'Clusters':<8} {'Opt':<10} {'MRR@100':<8} {'Comm(MB)':<10} {'Efficiency':<10}"
            )
            print("-" * 75)

            # Sort by communication cost
            sorted_experiments = sorted(
                multi_cluster.items(),
                key=lambda x: x[1]["latency"].get("total_comm_mb", 0),
            )

            for exp_name, exp_data in sorted_experiments:
                clusters = exp_data["num_clusters"]
                opt = exp_data["optimization"]
                mrr = exp_data["quality"].get("mrr_100", 0)
                comm = exp_data["latency"].get("total_comm_mb", 0)
                efficiency = mrr / comm if comm > 0 else 0

                print(
                    f"{exp_name:<15} {clusters:<8} {opt:<10} {mrr:<8.4f} {comm:<10.2f} {efficiency:<10.4f}"
                )

            # Find best configurations
            print("\n💡 Key Insights:")

            if sorted_experiments:
                # Best efficiency
                best_efficiency = max(
                    sorted_experiments,
                    key=lambda x: x[1]["quality"].get("mrr_100", 0)
                    / max(x[1]["latency"].get("total_comm_mb", 1), 1),
                )
                efficiency_score = best_efficiency[1]["quality"].get(
                    "mrr_100", 0
                ) / max(best_efficiency[1]["latency"].get("total_comm_mb", 1), 1)
                print(
                    f"   🏆 Best efficiency: {best_efficiency[0]} ({efficiency_score:.4f} MRR/MB)"
                )

                # Lowest communication with decent quality
                good_quality_experiments = [
                    (name, data)
                    for name, data in sorted_experiments
                    if data["quality"].get("mrr_100", 0) > 0.10
                ]
                if good_quality_experiments:
                    lowest_comm = min(
                        good_quality_experiments,
                        key=lambda x: x[1]["latency"].get("total_comm_mb", 0),
                    )
                    print(
                        f"   📡 Lowest communication (MRR>0.10): {lowest_comm[0]} "
                        f"({lowest_comm[1]['latency'].get('total_comm_mb', 0):.2f} MB)"
                    )

        else:
            print("\n⚠️  No multi-cluster experiments found")
            print("   ⚠️  Need to run both performance and quality experiments")

        print("\n" + "=" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Enhanced results analysis")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/home/azureuser/experiment_results",
        help="Results directory to analyze",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Analyze quick experiment results"
    )
    parser.add_argument(
        "--multi_cluster",
        action="store_true",
        help="Analyze multi-cluster experiment results",
    )

    args = parser.parse_args()

    # Handle conflicting flags - multi_cluster takes precedence
    if args.multi_cluster:
        results_dir = "/home/azureuser/quick_data/multi_cluster_results"
        print("🔬 Multi-cluster analysis mode")
    elif args.quick:
        results_dir = "/home/azureuser/quick_data/results"
        print("⚡ Quick analysis mode")
    else:
        results_dir = args.results_dir
        print(f"📁 Custom directory analysis: {results_dir}")

    # Ensure the directory exists
    if not Path(results_dir).exists():
        print(f"❌ Results directory not found: {results_dir}")
        print("Available directories:")
        base_dir = Path("/home/azureuser/quick_data")
        if base_dir.exists():
            for d in base_dir.iterdir():
                if d.is_dir():
                    print(f"  📁 {d}")
        return

    analyzer = TiptoeAnalyser(results_dir)

    # Analyze all results
    print(f"🚀 Starting analysis of: {results_dir}")
    try:
        results = analyzer.analyze_all_results()

        # Generate comprehensive report
        analyzer.generate_summary_report(results)

        # If multi-cluster analysis, also create plot
        multi_cluster = results.get("multi_cluster", {})
        # if multi_cluster:
        #     try:
        #         create_multi_cluster_plot(multi_cluster, results_dir)
        #     except ImportError:
        #         print("\n⚠️  matplotlib not available - skipping plot generation")
        #     except Exception as e:
        #         print(f"\n⚠️  Plot generation failed: {e}")

    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
