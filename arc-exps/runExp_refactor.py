"""
Refactored experiment code, to run both the throughput and quality experiments
locally on the azure client
"""

import argparse

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
        time.sleep(10)
        print("Coordinator started")

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

        # Run experiments
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
    arguments = parser.parse_args()
    main(arguments)
