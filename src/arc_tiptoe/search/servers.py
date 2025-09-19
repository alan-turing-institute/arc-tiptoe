"""
Class for running search servers and methods for cleaning up processes
"""

import contextlib
import socket
import subprocess
import time
from pathlib import Path

import psutil


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
                        subprocess.run(["kill", "-9", pid], check=False)
                        print(f"   Killed PID {pid} using port {port}")
        except (subprocess.SubprocessError, OSError, ValueError):
            pass

    time.sleep(5)
    print("   Cleanup complete")
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

        except (subprocess.SubprocessError, OSError, FileNotFoundError) as e:
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

        except (subprocess.SubprocessError, OSError, ValueError) as e:
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
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
            except (subprocess.SubprocessError, OSError, FileNotFoundError):
                # If lsof isn't available, try pkill
                with contextlib.suppress(
                    subprocess.SubprocessError, OSError, FileNotFoundError
                ):
                    subprocess.run(f"pkill -f ':{port}'", shell=True, check=False)

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
            except OSError:
                pass
            finally:
                sock.close()

        if occupied_ports:
            print(f"  Ports still occupied: {occupied_ports}")
            print("   Waiting for ports to be released...")
            time.sleep(10)

            # Force kill anything on those ports
            for port in occupied_ports:
                with contextlib.suppress(
                    subprocess.SubprocessError, OSError, FileNotFoundError
                ):
                    subprocess.run(
                        f"lsof -ti:{port} | xargs kill -9", shell=True, check=False
                    )

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
        self.cleanup_existing_processes()
