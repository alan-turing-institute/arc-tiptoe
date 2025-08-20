"""
Enhanced analysis with better pattern matching and debugging for quick iteration
"""

import argparse
import glob
import os
import re
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

    def analyze_all_results(self) -> Dict:
        """Analyze all available experiment results"""
        results = {"single_cluster": {"latency": {}, "quality": {}, "throughput": {}}}

        print(f"🔍 Analyzing results in: {self.results_dir}")

        # Find result files
        latency_files = glob.glob(
            f"{self.results_dir}/**/*latency*.log", recursive=True
        )
        quality_files = glob.glob(
            f"{self.results_dir}/**/*quality*.log", recursive=True
        )

        print(
            f"Found {len(latency_files)} latency files, {len(quality_files)} quality files"
        )

        # Parse latency results
        if latency_files:
            results["single_cluster"]["latency"] = self.parse_latency_log(
                latency_files[0]
            )

        # Parse quality results
        if quality_files:
            results["single_cluster"]["quality"] = self.parse_quality_log(
                quality_files[0]
            )

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
            print(f"   🎯 Queries with Qrels: {quality.get('queries_with_qrels', 0)}")

        # Key insights
        print("\n🔑 Key Insights:")
        if latency and quality:
            comm_cost = latency.get("total_comm_mb", 0)
            mrr_score = quality.get("mrr_100", 0)
            if comm_cost > 0 and mrr_score > 0:
                print(
                    f"   📈 Quality vs Communication: {mrr_score:.4f} MRR @ {comm_cost:.2f} MB"
                )
            elif comm_cost > 0:
                print(
                    f"   📡 Communication cost measured: {comm_cost:.2f} MB per query"
                )
                print("   ⚠️  Quality needs qrels file for MRR computation")
            else:
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

    args = parser.parse_args()

    if args.quick:
        results_dir = "/home/azureuser/quick_data/results"
    else:
        results_dir = args.results_dir

    analyzer = TiptoeAnalyser(results_dir)

    # Analyze all results
    results = analyzer.analyze_all_results()

    # Generate comprehensive report
    analyzer.generate_summary_report(results)


if __name__ == "__main__":
    main()
