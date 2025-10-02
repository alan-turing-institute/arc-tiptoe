"""
Multi-cluster search experiment orchestrator using Go processes
"""

import asyncio
import json
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import psutil


@dataclass
class SearchResult:
    """Single search result with URL and score"""

    url: str
    score: float
    cluster_id: int


@dataclass
class QueryExperimentResult:
    """Results for a single query across different cluster counts"""

    query_id: str
    query_text: str
    cluster_search_results: Dict[int, List[SearchResult]]  # num_clusters -> all results
    latency_results: Dict[int, float]  # num_clusters -> latency in ms
    throughput_results: Dict[int, float]  # num_clusters -> QPS


def aggressive_cleanup():
    """Aggressively clean up any remaining processes"""

    print("Performing aggressive cleanup...")

    # Kill all Go processes
    subprocess.run(["pkill", "-f", "go run"], check=False)
    time.sleep(2)

    # Kill processes by name
    for name in ["emb-server", "url-server", "coordinator", "all-servers"]:
        subprocess.run(["pkill", "-f", name], check=False)
    time.sleep(2)

    # Kill processes using Tiptoe ports
    for port in range(1237, 1250):
        try:
            result = subprocess.run(
                f"lsof -ti:{port}", shell=True, capture_output=True, text=True
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split("\n")
                for pid in pids:
                    if pid:
                        subprocess.run(["kill", "-9", pid], check=False)
                        print(f"   Killed PID {pid} using port {port}")
        except:
            pass

    time.sleep(5)
    print("   Cleanup complete")


class TiptoeServerManager:
    """Manages Tiptoe server processes"""

    def __init__(self, config_path: str, search_dir: str = "search"):
        self.config_path = config_path
        self.search_dir = Path(search_dir).resolve()
        self.server_process = None

    def start_servers(self) -> bool:
        """Start all Tiptoe servers"""
        print("Starting Tiptoe servers...")

        try:
            aggressive_cleanup()

            # Kill any existing processes
            self.cleanup_existing_processes()

            # Start all servers
            cmd = [
                "go",
                "run",
                ".",
                "--search_config",
                str(Path(self.config_path).resolve()),
                "all-servers",
            ]

            print(f"Running: {' '.join(cmd)}")
            print(f"Working directory: {self.search_dir}")

            self.server_process = subprocess.Popen(
                cmd,
                cwd=self.search_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            # Wait for servers to be ready
            return self.wait_for_servers_ready()

        except Exception as e:
            print(f"Failed to start servers: {e}")
            return False

    def wait_for_servers_ready(self, timeout: int = 300) -> bool:
        """Wait for servers to be ready by monitoring output"""
        print("Waiting for servers to be ready...")

        start_time = time.time()
        ready_indicators = {
            "embedding_servers": False,
            "url_servers": False,
            "coordinator": False,
            "ready": False,
        }

        try:
            for line in iter(self.server_process.stdout.readline, ""):
                print(f"[SERVER] {line.strip()}")

                # Check for readiness indicators
                if "Set up all embedding servers" in line:
                    ready_indicators["embedding_servers"] = True
                elif "Set up all url servers" in line:
                    ready_indicators["url_servers"] = True
                elif "Ready to start answering queries" in line:
                    ready_indicators["ready"] = True
                elif "TLS server listening" in line:
                    ready_indicators["coordinator"] = True

                # Check if all components are ready
                if all(ready_indicators.values()):
                    print("All servers are ready!")
                    return True

                # Check timeout
                if time.time() - start_time > timeout:
                    print(f"Timeout waiting for servers (>{timeout}s)")
                    return False

                # Check if process died
                if self.server_process.poll() is not None:
                    print("Server process died unexpectedly")
                    return False

        except Exception as e:
            print(f"Error waiting for servers: {e}")
            return False

        return False

    def cleanup_existing_processes(self):
        """Kill any existing Tiptoe processes"""
        print("Cleaning up existing processes...")

        # Kill Go processes that match our patterns
        patterns = [
            "go run.*server",
            "go run.*coordinator",
            "go run.*all-servers",
            "emb-server",
            "url-server",
        ]

        killed_count = 0
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = " ".join(proc.info["cmdline"] or [])
                for pattern in patterns:
                    if pattern.replace(".*", "") in cmdline.replace(" ", ""):
                        print(f"   Killing process {proc.info['pid']}: {cmdline[:100]}")
                        proc.kill()
                        killed_count += 1
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        time.sleep(2)  # Give processes time to die

        # Also kill processes using the specific ports

        ports_to_check = [
            1237,
            1238,
            1239,
            1240,
            1241,
            1242,
            1243,
            1244,
            1245,
        ]  # Tiptoe ports

        for port in ports_to_check:
            try:
                # Use lsof to find processes using the port
                result = subprocess.run(
                    f"lsof -ti:{port}",
                    shell=True,
                    capture_output=True,
                    text=True,
                    check=False,
                )

                if result.stdout.strip():
                    pids = result.stdout.strip().split("\n")
                    for pid in pids:
                        if pid:
                            try:
                                subprocess.run(["kill", "-9", pid], check=False)
                                print(f"   Killed PID {pid} using port {port}")
                                killed_count += 1
                            except:
                                pass
            except Exception as e:
                # If lsof isn't available, try pkill
                try:
                    subprocess.run(f"pkill -f ':{port}'", shell=True, check=False)
                except:
                    pass

        if killed_count > 0:
            print(f"   Killed {killed_count} processes")
            time.sleep(5)  # Give processes time to die and release ports
        else:
            print("   No processes to clean up")

    def check_ports_free(self):
        """Check that required ports are free"""

        ports_to_check = [1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245]
        occupied_ports = []

        for port in ports_to_check:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                result = sock.connect_ex(("127.0.0.1", port))
                if result == 0:  # Port is occupied
                    occupied_ports.append(port)
            except:
                pass
            finally:
                sock.close()

        if occupied_ports:
            print(f"  Ports still occupied: {occupied_ports}")
            print("   Waiting for ports to be released...")
            time.sleep(10)

            # Force kill anything on those ports
            for port in occupied_ports:
                try:
                    subprocess.run(
                        f"lsof -ti:{port} | xargs kill -9", shell=True, check=False
                    )
                except:
                    pass

            time.sleep(5)
        else:
            print("All required ports are free")

    def is_healthy(self) -> bool:
        """Check if servers are still running"""
        if self.server_process is None:
            return False
        return self.server_process.poll() is None

    def shutdown(self):
        """Shutdown all servers"""
        print("Shutting down servers...")

        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()

        self.cleanup_existing_processes()


class MultiClusterSearchExperiment:
    """Multi-cluster search experiment using Go client processes"""

    def __init__(
        self,
        config_path: str,
        max_clusters: int = 5,
        top_n_results: int = 100,
        search_dir: str = "search",
        results_dir: str = "multi_cluster_results",
    ):

        self.config_path = Path(config_path).resolve()
        self.max_clusters = max_clusters
        self.top_n_results = top_n_results
        self.search_dir = Path(search_dir).resolve()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        self.server_manager = TiptoeServerManager(
            str(self.config_path), str(self.search_dir)
        )

        print("   Multi-cluster experiment setup:")
        print(f"   Config: {self.config_path}")
        print(f"   Max clusters: {max_clusters}")
        print(f"   Top results per query: {top_n_results}")
        print(f"   Search directory: {self.search_dir}")
        print(f"   Results directory: {self.results_dir}")

    async def run_single_query_experiment(
        self, query_id: str, query_text: str
    ) -> QueryExperimentResult:
        """Run a single query experiment using Go multi-cluster command"""

        print(f"Processing query {query_id}: '{query_text[:50]}...'")

        result = QueryExperimentResult(
            query_id=query_id,
            query_text=query_text,
            cluster_search_results={},
            latency_results={},
            throughput_results={},  # Will store communication costs
        )

        try:
            # Run the Go multi-cluster command
            cmd = [
                "go",
                "run",
                ".",
                "--search_config",
                str(Path(self.config_path).resolve()),
                "multi-cluster",
                "127.0.0.1",
                query_text,
                str(self.max_clusters),
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.search_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                print(f"Go multi-cluster failed: {stderr.decode()}")
                return result

            # Parse the output
            output = stdout.decode()
            output_lines = output.split("\n")

            current_cluster_count = None
            current_latency = None
            current_comm_mb = None  # Communication cost instead of throughput
            in_results_section = False
            current_results = []

            for line in output_lines:
                line = line.strip()

                if line.startswith("CLUSTER_COUNT:"):
                    current_cluster_count = int(line.split(":")[1])
                    continue

                if line.startswith("LATENCY_MS:"):
                    current_latency = float(line.split(":")[1])
                    continue

                if line.startswith("COMMUNICATION_MB:"):
                    current_comm_mb = float(line.split(":")[1])
                    continue

                if line == "RESULTS_START":
                    in_results_section = True
                    current_results = []
                    continue

                if line == "RESULTS_END":
                    in_results_section = False

                    # Store results for this cluster count
                    if current_cluster_count is not None:
                        url_score_pairs = [(r.url, r.score) for r in current_results]

                        result.cluster_search_results[current_cluster_count] = (
                            url_score_pairs
                        )
                        if current_latency is not None:
                            result.latency_results[current_cluster_count] = (
                                current_latency
                            )
                        if current_comm_mb is not None:
                            result.throughput_results[current_cluster_count] = (
                                current_comm_mb  # Store as "throughput" for compatibility
                            )

                        print(
                            f"   {current_cluster_count} clusters: "
                            f"{current_latency:.1f}ms, {current_comm_mb:.6f} MB comm, "
                            f"{len(current_results)} results"
                        )
                    continue

                if in_results_section and line.startswith("RESULT:"):
                    try:
                        parts = line.split(":", 3)
                        if len(parts) >= 4:
                            url = parts[1]
                            score = float(parts[2])
                            cluster_id = int(parts[3])

                            current_results.append(
                                SearchResult(
                                    url=url, score=score, cluster_id=cluster_id
                                )
                            )
                    except (ValueError, IndexError):
                        continue

        except Exception as e:
            print(f"❌ Error processing query: {e}")

        return result

    async def get_top_clusters(self, query_text: str, top_k: int) -> list[int]:
        """Get top-k clusters using the embedding process"""
        try:
            cmd = [
                "python3",
                "embeddings/embed_text.py",
                str(self.config_path),
                "1280",  # Total clusters (will be read from config)
                str(top_k),
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.search_dir,
            )

            stdout, stderr = await process.communicate(input=f"{query_text}\n".encode())

            if process.returncode != 0:
                print(f"Embedding process failed: {stderr.decode()}")
                return [0]  # Fallback

            response = json.loads(stdout.decode().strip())
            top_k_clusters = response.get(
                "Top_k_clusters", [response.get("Cluster_index", 0)]
            )

            return top_k_clusters[:top_k]  # Ensure we don't exceed requested amount

        except Exception as e:
            print(f"Error getting top clusters: {e}")
            return [0]  # Fallback

    async def run_go_search(
        self, query_text: str, cluster_list: List[int]
    ) -> List[SearchResult]:
        """Run search using the existing working client command"""

        search_results = []

        try:
            # Use the existing client command that works with your servers
            cmd = [
                "go",
                "run",
                ".",
                "--search_config",
                str(Path(self.config_path).resolve()),
                "multi-cluster",
                "127.0.0.1",
                query_text,
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.search_dir,
            )

            # Send the query and quit command
            input_text = f"{query_text}\nquit\n"
            stdout, stderr = await process.communicate(input=input_text.encode())

            if process.returncode != 0:
                print(f" Go client failed: {stderr.decode()}")
                return []

            # Parse the output to extract search results
            output = stdout.decode()
            search_results = self.parse_go_client_output(output, cluster_list)

        except Exception as e:
            print(f" Error running Go search: {e}")
            search_results = []

        return search_results[: self.top_n_results]

    def parse_go_client_output(
        self, output: str, target_clusters: List[int]
    ) -> List[SearchResult]:
        """Parse Go multi-cluster output to extract URLs and scores"""
        results = []

        lines = output.split("\n")
        current_cluster_count = None
        in_results_section = False

        for line in lines:
            line = line.strip()

            # Parse cluster count
            if line.startswith("CLUSTER_COUNT:"):
                current_cluster_count = int(line.split(":")[1])
                continue

            # Parse results section
            if line == "RESULTS_START":
                in_results_section = True
                continue

            if line == "RESULTS_END":
                in_results_section = False
                continue

            # Parse individual results
            if in_results_section and line.startswith("RESULT:"):
                try:
                    # Format: RESULT:url:score:cluster_id
                    parts = line.split(":", 3)  # Split into max 4 parts
                    if len(parts) >= 4:
                        url = parts[1]
                        score = float(parts[2])
                        cluster_id = int(parts[3])

                        results.append(
                            SearchResult(url=url, score=score, cluster_id=cluster_id)
                        )

                except (ValueError, IndexError) as e:
                    continue

        return results

    async def run_experiment(self, queries: list[tuple[str, str]]) -> pd.DataFrame:
        """Run the complete multi-cluster experiment"""

        print(f"Starting multi-cluster experiment with {len(queries)} queries")

        # Start servers
        if not self.server_manager.start_servers():
            error = "Failed to start Tiptoe servers"
            raise RuntimeError(error)

        try:
            # Run all queries with limited concurrency
            semaphore = asyncio.Semaphore(1)  # Sequential for now to avoid overloading

            async def run_single_query(query_id: str, query_text: str):
                async with semaphore:
                    return await self.run_single_query_experiment(query_id, query_text)

            # Execute queries
            tasks = [run_single_query(qid, qtext) for qid, qtext in queries]
            query_results = []

            for i, task in enumerate(asyncio.as_completed(tasks)):
                try:
                    result = await task
                    query_results.append(result)
                    print(f"Completed query {i+1}/{len(queries)}")

                    # Check server health periodically
                    if not self.server_manager.is_healthy():
                        print("Servers died, stopping experiment")
                        break

                except Exception as e:
                    print(f"Query failed: {e}")
                    continue

            print(f"Completed {len(query_results)}/{len(queries)} queries successfully")

            # Convert to DataFrame
            return self.results_to_dataframe(query_results)

        finally:
            # Always cleanup
            self.server_manager.shutdown()

    def results_to_dataframe(
        self, results: list[QueryExperimentResult]
    ) -> pd.DataFrame:
        """Convert experiment results to DataFrame"""
        df_data = []

        for result in results:
            row = {"query_id": result.query_id}

            # Add results for each cluster count
            for cluster_count in range(1, self.max_clusters + 1):
                # Search results
                search_results = result.cluster_search_results.get(cluster_count, [])
                row[f"{cluster_count}_cluster_search_results"] = search_results

                # Performance metrics
                row[f"{cluster_count}_cluster_latency_result"] = (
                    result.latency_results.get(cluster_count, 0)
                )
                row[f"{cluster_count}_cluster_communication_mb"] = (
                    result.throughput_results.get(cluster_count, 0)
                )  # Communication cost

            df_data.append(row)

        df = pd.DataFrame(df_data)

        # Save results
        csv_file = self.results_dir / "multi_cluster_results.csv"
        json_file = self.results_dir / "multi_cluster_results.json"

        df.to_csv(csv_file, index=False)

        # Also save detailed JSON
        detailed_results = [asdict(r) for r in results]
        with open(json_file, "w") as f:
            json.dump(detailed_results, f, indent=2)

        print(f"\n📁 Results saved to:")
        print(f"   CSV: {csv_file}")
        print(f"   JSON: {json_file}")

        return df


def load_queries_from_file(
    query_file: str, max_queries: Optional[int] | None = None
) -> list[tuple[str, str]]:
    """Load queries from a TSV file"""
    queries = []

    with open(query_file, "r") as f:
        for i, line in enumerate(f):
            if max_queries and i >= max_queries:
                break

            parts = line.strip().split("\t")
            if len(parts) >= 2:
                query_id, query_text = parts[0], parts[1]
                queries.append((query_id, query_text))

    return queries


async def main():
    """Main function to run the experiment"""

    if len(sys.argv) < 2:
        print(
            "Usage: python3 multi_cluster_search_experiment.py <config_path> [max_queries] [max_clusters]"
        )
        sys.exit(1)

    config_path = sys.argv[1]
    max_queries = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    max_clusters = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    # Find query file
    query_files = ["test_data/test_queries.tsv"]

    query_file = None
    for qf in query_files:
        if Path(qf).exists():
            query_file = qf
            break

    if not query_file:
        print(
            "Could not find query file. Please ensure msmarco-docdev-queries.tsv exists."
        )
        sys.exit(1)

    # Load queries
    queries = load_queries_from_file(query_file, max_queries)
    print(f"Loaded {len(queries)} queries from {query_file}")

    # Run experiment
    experiment = MultiClusterSearchExperiment(
        config_path=config_path, max_clusters=max_clusters, top_n_results=100
    )

    try:
        results_df = await experiment.run_experiment(queries)

        # Quick analysis
        print("\nQuick Results Summary:")
        print("Average latency by cluster count:")
        for i in range(1, max_clusters + 1):
            if f"{i}_cluster_latency_result" in results_df.columns:
                avg_latency = results_df[f"{i}_cluster_latency_result"].mean()
                avg_throughput = results_df[f"{i}_cluster_throughput_result"].mean()
                print(
                    f"  {i} clusters: {avg_latency:.1f}ms latency, {avg_throughput:.2f} QPS"
                )

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
    except Exception as e:
        print(f"\nExperiment failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
