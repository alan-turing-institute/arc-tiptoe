"""
Refactored experiment code, to run both the throughput and quality experiments
locally on the azure client
"""

import argparse
import json

# import json
import os

# import signal
import subprocess
import time
from pathlib import Path


class LocalTiptoeCluster:
    """
    Local cluster for running Tiptoe experiments
    """

    def __init__(
        self, num_embed_servers, num_url_servers, preamble, image_search=False
    ):
        self.num_embed_servers = num_embed_servers
        self.num_url_servers = num_url_servers
        self.preamble = preamble
        self.image_search = image_search
        self.processes = []
        self.embed_ports = list(range(8001, 8001 + num_embed_servers))
        self.url_ports = list(range(9001, 9001 + num_url_servers))
        self.coordinator_port = 8000

        # Set default search directory
        self.search_dir = "../search"

        # Try to find the correct search directory
        current_dir = Path.cwd()
        if (current_dir / "search").exists():
            self.search_dir = "search"
        elif (current_dir.parent / "search").exists():
            self.search_dir = "../search"
        else:
            # Look for search directory in the workspace
            search_paths = [
                current_dir / "search",
                current_dir.parent / "search",
                Path("/home/azureuser/search"),
                Path.cwd().parent / "search",
            ]
            for path in search_paths:
                if path.exists():
                    self.search_dir = str(path)
                    break

    def start_embedding_servers(self):
        """Start embedding servers locally"""
        print(f"Starting {self.num_embed_servers} embedding servers...")

        for i in range(self.num_embed_servers):
            cmd = ["go", "run", ".", "emb-server", str(i), "-preamble", self.preamble]
            if self.image_search:
                cmd.extend(["-image_search", "true"])

            print(f"Starting embedding server {i}...")
            print(f"Working directory: {self.search_dir}")  # Debug info
            proc = subprocess.Popen(
                cmd, cwd=self.search_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            self.processes.append(("embed", i, proc))
            time.sleep(2)  # Stagger startup

        # Wait for servers to be ready
        time.sleep(30)
        print("All embedding servers started")

    # Update other methods similarly to use self.search_dir instead of "../search"
    def start_url_servers(self):
        """Start URL servers locally"""
        print(f"Starting {self.num_url_servers} URL servers...")

        for i in range(self.num_url_servers):
            cmd = ["go", "run", ".", "url-server", str(i), "-preamble", self.preamble]
            if self.image_search:
                cmd.extend(["-image_search", "true"])

            print(f"Starting URL server {i}...")
            proc = subprocess.Popen(
                cmd, cwd=self.search_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            self.processes.append(("url", i, proc))
            time.sleep(2)

        time.sleep(15)
        print("All URL servers started")

    def start_coordinator(self):
        """Start coordinator locally"""
        print("Starting coordinator...")

        # Build IP string for localhost
        embed_ips = " ".join(["127.0.0.1"] * self.num_embed_servers)
        url_ips = " ".join(["127.0.0.1"] * self.num_url_servers)
        ip_string = f"{embed_ips} {url_ips}"

        cmd = [
            "go",
            "run",
            ".",
            "coordinator",
            str(self.num_embed_servers),
            str(self.num_url_servers),
            *ip_string.split(),
            "-preamble",
            self.preamble,
        ]
        if self.image_search:
            cmd.extend(["-image_search", "true"])

        proc = subprocess.Popen(
            cmd, cwd=self.search_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self.processes.append(("coordinator", 0, proc))

        # Wait longer for coordinator to be ready
        print("Waiting for coordinator to initialize...")
        time.sleep(60)  # Increased from 10 to 60 seconds

        # Check if coordinator is still running
        if proc.poll() is not None:
            print(f"❌ Coordinator failed to start (exit code: {proc.poll()})")
            _, stderr = proc.communicate()
            print(f"Error: {stderr.decode()}")
            return False

        print("Coordinator started")
        return True

    def run_latency_experiment(self):
        """Run latency experiment"""
        print("Running latency experiment...")

        cmd = [
            "go",
            "run",
            ".",
            "client-latency",
            "127.0.0.1",
            "-preamble",
            self.preamble,
        ]
        if self.image_search:
            cmd.extend(["-image_search", "true"])

        # Add more detailed error handling
        try:
            result = subprocess.run(
                cmd, cwd=self.search_dir, capture_output=True, text=True, check=False
            )

            if result.returncode != 0:
                print(
                    f"❌ Client latency experiment failed with exit code {result.returncode}"
                )
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return None

            # Save results only if successful
            prefix = "local-img/" if self.image_search else "local-text/"
            os.makedirs(prefix, exist_ok=True)

            filename = (
                f"{prefix}{self.num_embed_servers}-{self.num_url_servers}-1-latency.log"
            )
            with open(filename, "w", encoding="utf-8") as f:
                f.write(result.stdout)

            print(f"✅ Latency experiment completed. Results saved to {filename}")
            return result.stdout

        except (subprocess.SubprocessError, OSError, IOError) as e:
            print(f"❌ Exception during latency experiment: {e}")
            return None

    def run_throughput_experiments(self):
        """Run throughput experiments"""
        results = {}

        prefix = "local-img/" if self.image_search else "local-text/"

        # Embedding throughput
        print("Running embedding throughput experiment...")
        cmd = [
            "go",
            "run",
            ".",
            "client-tput-embed",
            "127.0.0.1",
            "-preamble",
            self.preamble,
        ]
        if self.image_search:
            cmd.extend(["-image_search", "true"])

        result = subprocess.run(
            cmd, cwd="../search", capture_output=True, text=True, check=True
        )

        filename = (
            f"{prefix}{self.num_embed_servers}-{self.num_url_servers}-1-tput-embed.log"
        )
        with open(filename, "w", encoding="utf-8") as f:
            f.write(result.stdout)
        results["embed_tput"] = result.stdout

        # URL throughput
        print("Running URL throughput experiment...")
        cmd = [
            "go",
            "run",
            ".",
            "client-tput-url",
            "127.0.0.1",
            "-preamble",
            self.preamble,
        ]
        if self.image_search:
            cmd.extend(["-image_search", "true"])

        result = subprocess.run(
            cmd, cwd="../search", capture_output=True, text=True, check=True
        )

        filename = (
            f"{prefix}{self.num_embed_servers}-{self.num_url_servers}-1-tput-url.log"
        )
        with open(filename, "w", encoding="utf-8") as f:
            f.write(result.stdout)
        results["url_tput"] = result.stdout

        # Offline throughput
        print("Running offline throughput experiment...")
        cmd = [
            "go",
            "run",
            ".",
            "client-tput-offline",
            "127.0.0.1",
            "-preamble",
            self.preamble,
        ]
        if self.image_search:
            cmd.extend(["-image_search", "true"])

        result = subprocess.run(
            cmd, cwd="../search", capture_output=True, text=True, check=True
        )

        filename = f"{prefix}{self.num_embed_servers}-{self.num_url_servers}-1-tput-offline.log"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(result.stdout)
        results["offline_tput"] = result.stdout

        print("All throughput experiments completed")
        return results

    def check_servers_running(self):
        """Check if servers are still running"""
        running_servers = []
        for proc_type, idx, proc in self.processes:
            if proc.poll() is None:  # Still running
                running_servers.append(f"{proc_type}-{idx}")
            else:
                print(
                    f"⚠️  Server {proc_type}-{idx} has stopped (exit code: {proc.poll()})"
                )
                # Print any error output
                _, stderr = proc.communicate()
                if stderr:
                    print(f"   Error: {stderr.decode()}")

        print(f"   🟢 Running servers: {running_servers}")
        return len(running_servers) > 0

    def run_throughput_experiment_for_clusters(self, num_clusters=1):
        """Run throughput experiments and save with cluster-specific names"""
        print(f"Running throughput experiments for {num_clusters} clusters...")

        # First check if servers are still running
        if not self.check_servers_running():
            print("❌ No servers running - cannot run throughput experiments")
            return {}

        # Also test coordinator connectivity
        try:
            test_cmd = [
                "go",
                "run",
                ".",
                "client-latency",
                "127.0.0.1",
                "-preamble",
                self.preamble,
            ]
            # Quick test with timeout
            test_result = subprocess.run(
                test_cmd,
                cwd=self.search_dir,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

            if test_result.returncode != 0:
                print(f"❌ Coordinator connectivity test failed: {test_result.stderr}")
                return {}
            else:
                print("✅ Coordinator connectivity verified")

        except subprocess.TimeoutExpired:
            print("❌ Coordinator connectivity test timed out")
            return {}
        except Exception as e:
            print(f"❌ Coordinator connectivity test error: {e}")
            return {}

        results = {}
        prefix = "local-img/" if self.image_search else "local-text/"
        os.makedirs(prefix, exist_ok=True)

        # Embedding throughput
        print("  📊 Running embedding throughput...")
        cmd = [
            "go",
            "run",
            ".",
            "client-tput-embed",
            "127.0.0.1",
            "-preamble",
            self.preamble,
        ]
        if self.image_search:
            cmd.extend(["-image_search", "true"])

        try:
            result = subprocess.run(
                cmd,
                cwd=self.search_dir,
                capture_output=True,
                text=True,
                check=False,
                timeout=600,
            )

            if result.returncode == 0:
                filename = f"{prefix}{num_clusters}c_tput_embed.log"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(result.stdout)
                results["embed_tput"] = result.stdout
                print(f"    ✅ Embed throughput saved to: {filename}")
            else:
                print(f"    ❌ Embed throughput failed: {result.stderr}")

        except Exception as e:
            print(f"    ❌ Embed throughput error: {e}")

        # URL throughput
        print("  📊 Running URL throughput...")
        cmd = [
            "go",
            "run",
            ".",
            "client-tput-url",
            "127.0.0.1",
            "-preamble",
            self.preamble,
        ]
        if self.image_search:
            cmd.extend(["-image_search", "true"])

        try:
            result = subprocess.run(
                cmd,
                cwd=self.search_dir,
                capture_output=True,
                text=True,
                check=False,
                timeout=600,
            )

            if result.returncode == 0:
                filename = f"{prefix}{num_clusters}c_tput_url.log"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(result.stdout)
                results["url_tput"] = result.stdout
                print(f"    ✅ URL throughput saved to: {filename}")
            else:
                print(f"    ❌ URL throughput failed: {result.stderr}")

        except Exception as e:
            print(f"    ❌ URL throughput error: {e}")

        # Offline throughput
        print("  📊 Running offline throughput...")
        cmd = [
            "go",
            "run",
            ".",
            "client-tput-offline",
            "127.0.0.1",
            "-preamble",
            self.preamble,
        ]
        if self.image_search:
            cmd.extend(["-image_search", "true"])

        try:
            result = subprocess.run(
                cmd,
                cwd=self.search_dir,
                capture_output=True,
                text=True,
                check=False,
                timeout=600,
            )

            if result.returncode == 0:
                filename = f"{prefix}{num_clusters}c_tput_offline.log"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(result.stdout)
                results["offline_tput"] = result.stdout
                print(f"    ✅ Offline throughput saved to: {filename}")
            else:
                print(f"    ❌ Offline throughput failed: {result.stderr}")

        except Exception as e:
            print(f"    ❌ Offline throughput error: {e}")

        print(f"  ✅ Throughput experiments completed for {num_clusters} clusters")
        return results

    def run_quality_experiment(
        self, num_clusters=1, optimization="basic", query_file=None
    ):
        """Run quality experiment using clustering/search.py with parallelization"""
        print(
            f"Running quality experiment: {num_clusters} clusters, {optimization} optimization..."
        )

        # Use the full query file if not specified
        if query_file is None:
            # Try to find the FULL query file first (not the subset)
            possible_query_files = [
                f"{self.preamble}/msmarco_data/msmarco-docdev-queries.tsv",  # Full dataset first!
                f"{self.preamble}/data/msmarco-docdev-queries.tsv",
                f"{self.preamble}/quick_data/full_queries.tsv",  # If you created this
                f"{self.preamble}/quick_data/quick_queries.tsv",  # Fallback to subset
            ]

            for qf in possible_query_files:
                if os.path.exists(qf):
                    query_file = qf
                    # Check how many queries are in this file
                    with open(qf, "r") as f:
                        query_count = len(f.readlines())
                    print(f"   📁 Using query file: {qf} ({query_count} queries)")

                    # Only use files with substantial number of queries
                    if (
                        query_count >= 1000
                    ):  # Require at least 1000 queries for full experiment
                        break
                    else:
                        print(
                            f"   ⚠️  File {qf} only has {query_count} queries, looking for larger dataset..."
                        )
                        query_file = None

            if query_file is None:
                print(
                    "   ❌ No substantial query file found! Need file with 1000+ queries"
                )
                print("   📋 Available files:")
                for qf in possible_query_files:
                    if os.path.exists(qf):
                        with open(qf, "r") as f:
                            count = len(f.readlines())
                        print(f"     {qf}: {count} queries")
                return None

        # Create config for clustering/search.py
        config = {
            "pca_components_file": f"{self.preamble}/data/embeddings/pca_components_192.txt",
            "query_file": query_file,
            "cluster_file_location": f"{self.preamble}/data/clusters/",
            "url_bundle_base_dir": f"{self.preamble}/data/clusters/",
            "index_file": f"{self.preamble}/data/artifact/dim192/index.faiss",
            "is_text": True,
            "run_msmarco_dev_queries": True,
            "filter_badwords": False,
            "short_exp": False,  # CRITICAL: Must be False for full dataset
            "num_clusters": num_clusters,
            "centroids_file": f"{self.preamble}/data/embeddings/centroids.txt",
            "badwords_file": None,
            "img_results_dir": None,
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
        elif optimization == "pca":
            config.update(
                {
                    "run_pca": True,
                    "run_url_filter": False,
                    "url_filter_by_cluster": False,
                }
            )

        # Save config in clustering directory
        config_filename = f"multi_cluster_config_{num_clusters}c_{optimization}.json"
        config_file = f"clustering/{config_filename}"

        print(f"   💾 Saving config to: {config_file}")
        print(
            f"   🔧 Config: short_exp={config['short_exp']}, query_file={config['query_file']}"
        )

        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        # Run clustering/search.py
        cmd = ["python3", "search.py", config_filename]

        try:
            result = subprocess.run(
                cmd,
                cwd="clustering",
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hours for full dataset
                check=False,
            )

            if result.returncode == 0:
                # Save results
                prefix = "local-img/" if self.image_search else "local-text/"
                os.makedirs(prefix, exist_ok=True)

                filename = f"{prefix}{num_clusters}c_{optimization}_quality.log"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(result.stdout)

                print(f"✅ Quality experiment completed. Results saved to {filename}")

                # Quick analysis
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
                    f"   📊 Processed {query_count} queries, {result_count} total results"
                )

                return result.stdout
            else:
                print(f"❌ Quality experiment failed: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            print(f"❌ Quality experiment timed out after 2 hours")
            return None
        except Exception as e:
            print(f"❌ Quality experiment error: {e}")
            return None

    def run_multi_cluster_experiments_parallel(
        self, cluster_configs=None, optimizations=None
    ):
        """Run multi-cluster experiments in parallel"""
        import concurrent.futures

        if cluster_configs is None:
            cluster_configs = [1, 2, 4]
        if optimizations is None:
            optimizations = ["basic", "pca"]

        print("🚀 Running multi-cluster experiments in parallel...")
        print(f"   Cluster configs: {cluster_configs}")
        print(f"   Optimizations: {optimizations}")

        results = []

        # Run quality experiments in parallel
        quality_experiments = []
        for num_clusters in cluster_configs:
            for optimization in optimizations:
                quality_experiments.append((num_clusters, optimization))

        print("\n📊 Running quality experiments...")
        # DON'T use servers for quality experiments yet - run them separately

        # First, run quality experiments WITHOUT starting servers (use existing running servers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_to_experiment = {
                executor.submit(
                    self.run_quality_experiment, num_clusters, optimization
                ): (num_clusters, optimization)
                for num_clusters, optimization in quality_experiments
            }

            for future in concurrent.futures.as_completed(future_to_experiment):
                num_clusters, optimization = future_to_experiment[future]
                experiment_name = f"{num_clusters}c_{optimization}"

                try:
                    quality_results = future.result()
                    success = quality_results is not None

                    result = {
                        "experiment": experiment_name,
                        "num_clusters": num_clusters,
                        "optimization": optimization,
                        "quality_success": success,
                    }

                    results.append(result)

                    if success:
                        print(f"✅ {experiment_name}: Quality Success")
                    else:
                        print(f"❌ {experiment_name}: Quality Failed")

                except Exception as e:
                    print(f"❌ {experiment_name}: Quality Error - {e}")
                    results.append(
                        {
                            "experiment": experiment_name,
                            "num_clusters": num_clusters,
                            "optimization": optimization,
                            "error": str(e),
                            "quality_success": False,
                        }
                    )

        # Run throughput experiments WHILE servers are still running
        print("\n⚡ Running throughput experiments...")
        for num_clusters in cluster_configs:
            experiment_name = f"{num_clusters}c_throughput"
            print(f"🔬 {experiment_name}")

            try:
                # Servers should still be running from the main startup
                throughput_results = self.run_throughput_experiment_for_clusters(
                    num_clusters
                )
                throughput_success = bool(throughput_results)

                # Update corresponding quality results
                for result in results:
                    if result["num_clusters"] == num_clusters:
                        result["throughput_success"] = throughput_success

                if throughput_success:
                    print(f"✅ {experiment_name}: Success")
                else:
                    print(f"❌ {experiment_name}: Failed")

            except Exception as e:
                print(f"❌ {experiment_name}: Error - {e}")
                # Update results
                for result in results:
                    if result["num_clusters"] == num_clusters:
                        result["throughput_success"] = False
                        result["throughput_error"] = str(e)

        # Summary
        print("\n📋 Multi-Cluster Experiments Summary:")
        print(f"{'Experiment':<12} {'Quality':<8} {'Throughput':<10}")
        print("-" * 35)
        for result in sorted(
            results, key=lambda x: (x["num_clusters"], x["optimization"])
        ):
            qual = "✅" if result.get("quality_success", False) else "❌"
            tput = "✅" if result.get("throughput_success", False) else "❌"
            print(f"{result['experiment']:<12} {qual:<8} {tput:<10}")

        return results

    def cleanup(self):
        """Terminate all processes"""
        print("Cleaning up processes...")
        for proc_type, idx, proc in self.processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            print(f"Terminated {proc_type} server {idx}")

        # Kill any remaining Go processes
        subprocess.run(["pkill", "-f", "go run"], check=False)
        print("Cleanup completed")


def main(args):
    """Main function for running refactored experiment"""
    cluster = LocalTiptoeCluster(
        num_embed_servers=args.num_embed_servers,
        num_url_servers=args.num_url_servers,
        preamble=args.preamble,
        image_search=args.image_search,
    )

    try:
        # Start all servers
        cluster.start_embedding_servers()
        cluster.start_url_servers()
        cluster.start_coordinator()

        # Check if multi-cluster mode
        if args.multi_cluster:
            # Run multi-cluster experiments
            cluster.run_multi_cluster_experiments_parallel(
                cluster_configs=args.clusters, optimizations=args.optimizations
            )
        elif args.quality_only:
            # Run only quality experiment
            cluster.run_quality_experiment(
                num_clusters=args.num_clusters, optimization=args.optimization
            )
        elif args.performance_only:
            # Run only performance experiments
            cluster.run_latency_experiment()
            cluster.run_throughput_experiments()
        else:
            # Run single experiments (original functionality)
            cluster.run_latency_experiment()
            cluster.run_throughput_experiments()

        print("\n=== Experiment Summary ===")
        print(f"Embedding servers: {args.num_embed_servers}")
        print(f"URL servers: {args.num_url_servers}")
        print("Results saved to local-text/ directory")
        print("Experiments completed successfully!")

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"Error during experiments: {e}")
    finally:
        cluster.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local performance experiments")
    parser.add_argument("-is", "--image_search", action="store_true", default=False)
    parser.add_argument(
        "--num_embed_servers", type=int, default=8, help="Number of embedding servers"
    )
    parser.add_argument(
        "--num_url_servers", type=int, default=2, help="Number of URL servers"
    )
    parser.add_argument(
        "--preamble", type=str, default="/home/azureuser", help="Data path"
    )

    # Multi-cluster options
    parser.add_argument(
        "--multi_cluster", action="store_true", help="Run multi-cluster experiments"
    )
    parser.add_argument(
        "--clusters",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Cluster configurations to test",
    )
    parser.add_argument(
        "--optimizations",
        type=str,
        nargs="+",
        default=["basic", "pca"],
        help="Optimizations to test",
    )

    # Single experiment options
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=1,
        help="Number of clusters for single experiment",
    )
    parser.add_argument(
        "--optimization",
        type=str,
        default="basic",
        choices=["basic", "pca"],
        help="Optimization for single experiment",
    )
    parser.add_argument("--performance_only", action="store_true")
    parser.add_argument("--quality_only", action="store_true")

    arguments = parser.parse_args()
    main(arguments)
