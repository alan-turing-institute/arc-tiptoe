"""
Enhanced analysis with better pattern matching and debugging for quick iteration
"""

import argparse
import glob
import json
import os
import re
from typing import Dict

# import matplotlib.pyplot as plt


class TiptoeAnalyser:
    """Better analysis"""

    def __init__(self, results_dir: str = "/home/azureuser/experiment_results"):
        self.results_dir = results_dir

    def parse_latency_log(self, log_file: str) -> Dict:
        """Parse latency log to extract communication costs and timing"""
        metrics = {
            "avg_latency": 0.0,
            "total_queries": 0,
            "total_upload_mb": 0.0,
            "total_download_mb": 0.0,
            "total_communication_mb": 0.0,
            "avg_communication_per_query_mb": 0.0,
        }

        if not os.path.exists(log_file):
            print(f"❌ Latency log not found: {log_file}")
            return metrics

        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        print(f"📊 Parsing latency log: {os.path.basename(log_file)}")

        # Parse Go client-latency output patterns
        patterns = {
            # Look for summary statistics
            "total_queries": [
                r"Processed\s+(\d+)\s+queries",
                r"Total queries:\s*(\d+)",
                r"(\d+)\s+queries\s+processed",
            ],
            "avg_latency": [
                r"Average latency:\s*([\d.]+)\s*s",
                r"Mean latency:\s*([\d.]+)\s*s",
                r"Avg:\s*([\d.]+)\s*s",
            ],
            "total_upload_mb": [
                r"Total upload:\s*([\d.]+)\s*MB",
                r"Upload total:\s*([\d.]+)\s*MB",
                r"Total sent:\s*([\d.]+)\s*MB",
            ],
            "total_download_mb": [
                r"Total download:\s*([\d.]+)\s*MB",
                r"Download total:\s*([\d.]+)\s*MB",
                r"Total received:\s*([\d.]+)\s*MB",
            ],
        }

        # Try each pattern
        for metric, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    try:
                        if "queries" in metric:
                            metrics[metric] = int(match.group(1))
                        else:
                            metrics[metric] = float(match.group(1))
                        print(f"  ✅ Found {metric}: {metrics[metric]}")
                        break
                    except (ValueError, IndexError):
                        continue

        # Calculate derived metrics
        metrics["total_communication_mb"] = (
            metrics["total_upload_mb"] + metrics["total_download_mb"]
        )

        if metrics["total_queries"] > 0:
            metrics["avg_communication_per_query_mb"] = (
                metrics["total_communication_mb"] / metrics["total_queries"]
            )

        # If we didn't find summary stats, try to parse individual query results
        if metrics["total_queries"] == 0:
            print("  📝 No summary found, parsing individual queries...")

            # Look for individual query latencies
            query_latencies = re.findall(
                r"Query\s+\d+.*?(\d+\.?\d*)\s*ms", content, re.IGNORECASE
            )
            if not query_latencies:
                query_latencies = re.findall(r"(\d+\.?\d*)\s*ms", content)

            if query_latencies:
                latencies = [
                    float(x) for x in query_latencies if x.replace(".", "").isdigit()
                ]
                if latencies:
                    metrics["avg_latency"] = (
                        sum(latencies) / len(latencies) / 1000
                    )  # Convert ms to s
                    metrics["total_queries"] = len(latencies)
                    print(f"  ✅ Parsed {len(latencies)} individual query latencies")

            # Look for individual communication costs
            upload_values = re.findall(r"upload[:\s]+([\d.]+)", content, re.IGNORECASE)
            download_values = re.findall(
                r"download[:\s]+([\d.]+)", content, re.IGNORECASE
            )

            if upload_values:
                metrics["total_upload_mb"] = sum(
                    float(x) for x in upload_values if x.replace(".", "").isdigit()
                )
            if download_values:
                metrics["total_download_mb"] = sum(
                    float(x) for x in download_values if x.replace(".", "").isdigit()
                )

            metrics["total_communication_mb"] = (
                metrics["total_upload_mb"] + metrics["total_download_mb"]
            )

            if metrics["total_queries"] > 0:
                metrics["avg_communication_per_query_mb"] = (
                    metrics["total_communication_mb"] / metrics["total_queries"]
                )

        return metrics

    def parse_quality_log(self, log_file: str) -> Dict:
        """Parse quality log to extract MRR and search metrics"""
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

        # Look for MRR scores in various formats
        mrr_patterns = [
            r"MRR@100[:\s]+([\d.]+)",
            r"MRR@10[:\s]+([\d.]+)",
            r"Mean Reciprocal Rank.*?100.*?[:\s]+([\d.]+)",
            r"Overall MRR[:\s]+([\d.]+)",
            r"Final MRR[:\s]+([\d.]+)",
        ]

        for pattern in mrr_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    mrr_value = float(match.group(1))
                    if "@100" in pattern or "Overall" in pattern or "Final" in pattern:
                        metrics["mrr_100"] = mrr_value
                        print(f"  ✅ Found MRR@100: {mrr_value:.4f}")
                    elif "@10" in pattern:
                        metrics["mrr_10"] = mrr_value
                        print(f"  ✅ Found MRR@10: {mrr_value:.4f}")
                    break
                except (ValueError, IndexError):
                    continue

        # Count queries and results
        lines = content.split("\n")

        # Count queries processed
        query_lines = [
            line for line in lines if re.match(r"Query\s*\d+", line, re.IGNORECASE)
        ]
        metrics["queries_evaluated"] = len(query_lines)

        # Count result lines (lines that look like search results)
        result_patterns = [
            r"^\s*\d+[\)\.]\s+",  # "1. result" or "1) result"
            r"^\s*[\d.]+\s+\S+",  # "0.85 document_id"
            r"^\s*Score:\s*[\d.]+",  # "Score: 0.85"
        ]

        result_count = 0
        queries_with_results = 0
        current_query_has_results = False

        for line in lines:
            line = line.strip()

            # Check if this is a new query
            if re.match(r"Query\s*\d+", line, re.IGNORECASE):
                if current_query_has_results:
                    queries_with_results += 1
                current_query_has_results = False

            # Check if this is a result line
            for pattern in result_patterns:
                if re.match(pattern, line):
                    result_count += 1
                    current_query_has_results = True
                    break

        # Check last query
        if current_query_has_results:
            queries_with_results += 1

        metrics["total_results_returned"] = result_count
        metrics["queries_with_results"] = queries_with_results

        if metrics["queries_evaluated"] > 0:
            metrics["avg_results_per_query"] = (
                result_count / metrics["queries_evaluated"]
            )

        # If no MRR found but we have results, calculate basic metrics
        if metrics["mrr_100"] == 0.0 and result_count > 0:
            print(
                f"  📊 No MRR found, but processed {metrics['queries_evaluated']} queries with {result_count} results"
            )

        return metrics

    def analyze_experiment_structure(self, results_dir: str):
        """Analyze the structure of experiment results"""
        print("=== Analyzing Results Structure ===")
        print(f"Results directory: {results_dir}")

        if not os.path.exists(results_dir):
            print(f"❌ Results directory does not exist: {results_dir}")
            return None

        # List all files
        all_files = []
        for root, _, files in os.walk(results_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, results_dir)
                size = os.path.getsize(full_path)
                all_files.append((rel_path, size))

        print(f"\nFound {len(all_files)} files:")
        for file_path, size in sorted(all_files):
            print(f"  {file_path:40} ({size:8} bytes)")

        # Identify experiment types
        latency_files = glob.glob(f"{results_dir}/**/*latency*.log", recursive=True)
        quality_files = glob.glob(f"{results_dir}/**/*quality*.log", recursive=True)
        tput_files = glob.glob(f"{results_dir}/**/*tput*.log", recursive=True)

        print("\nExperiment files found:")
        print(f"  Latency files: {len(latency_files)}")
        for f in latency_files:
            print(f"    📄 {os.path.basename(f)}")
        print(f"  Quality files: {len(quality_files)}")
        for f in quality_files:
            print(f"    📄 {os.path.basename(f)}")
        print(f"  Throughput files: {len(tput_files)}")

        return {
            "latency_files": latency_files,
            "quality_files": quality_files,
            "throughput_files": tput_files,
        }

    def analyze_all_results(self) -> Dict:
        """Analyze all results and extract key metrics"""
        print("\n=== Analyzing All Results ===")

        file_structure = self.analyze_experiment_structure(self.results_dir)
        if not file_structure:
            return {}

        results = {}

        # Parse latency results
        if file_structure["latency_files"]:
            print("\n📊 Processing latency files...")
            for latency_file in file_structure["latency_files"]:
                exp_name = os.path.basename(latency_file).replace(".log", "")
                results[exp_name] = {"latency": self.parse_latency_log(latency_file)}

        # Parse quality results
        if file_structure["quality_files"]:
            print("\n🔍 Processing quality files...")
            for quality_file in file_structure["quality_files"]:
                exp_name = os.path.basename(quality_file).replace(".log", "")
                if exp_name not in results:
                    results[exp_name] = {}
                results[exp_name]["quality"] = self.parse_quality_log(quality_file)

        return results

    def generate_summary_report(self, results: Dict):
        """Generate a comprehensive summary focusing on MRR and communication"""
        print("\n" + "=" * 60)
        print("📋 EXPERIMENT SUMMARY REPORT")
        print("=" * 60)

        for exp_name, exp_data in results.items():
            print(f"\n🔬 Experiment: {exp_name}")
            print("-" * 40)

            # Performance/Communication metrics
            if "latency" in exp_data:
                latency = exp_data["latency"]
                print("📡 COMMUNICATION METRICS:")
                print(f"   Total Queries: {latency.get('total_queries', 0)}")
                print(f"   Total Upload: {latency.get('total_upload_mb', 0):.2f} MB")
                print(
                    f"   Total Download: {latency.get('total_download_mb', 0):.2f} MB"
                )
                print(
                    f"   TOTAL COMMUNICATION: {latency.get('total_communication_mb', 0):.2f} MB"
                )
                print(
                    f"   Avg per Query: {latency.get('avg_communication_per_query_mb', 0):.3f} MB"
                )
                print(
                    f"   Average Latency: {latency.get('avg_latency', 0):.3f} seconds"
                )

            # Quality metrics
            if "quality" in exp_data:
                quality = exp_data["quality"]
                print("🎯 SEARCH QUALITY METRICS:")
                print(f"   OVERALL MRR@100: {quality.get('mrr_100', 0):.4f}")
                print(f"   MRR@10: {quality.get('mrr_10', 0):.4f}")
                print(f"   Queries Evaluated: {quality.get('queries_evaluated', 0)}")
                print(f"   Total Results: {quality.get('total_results_returned', 0)}")
                print(
                    f"   Avg Results/Query: {quality.get('avg_results_per_query', 0):.1f}"
                )
                print(
                    f"   Queries with Results: {quality.get('queries_with_results', 0)}"
                )

        # Overall summary
        print("\n" + "=" * 60)
        print("🎯 KEY METRICS SUMMARY")
        print("=" * 60)

        for exp_name, exp_data in results.items():
            latency = exp_data.get("latency", {})
            quality = exp_data.get("quality", {})

            mrr = quality.get("mrr_100", 0)
            comm = latency.get("total_communication_mb", 0)

            print(f"{exp_name:20} | MRR: {mrr:6.4f} | Communication: {comm:8.2f} MB")

        # Save detailed results
        report_file = os.path.join(self.results_dir, "detailed_analysis.json")
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"\n📄 Detailed results saved to: {report_file}")

        return results


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
