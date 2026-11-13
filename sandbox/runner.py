import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

from config.config_loader import load_sandbox_limits, load_time_budget_sec  
from sandbox.broker import Broker
import bittensor as bt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_WORK_ROOT = Path("/data/miner_runs")
DATA_OUTPUT_ROOT = Path("/data/miner_output")

SANDBOX_IMAGE_TAG = "urdof7/miner-sandbox:latest"
SANDBOX_UID = 10001  # mineruser in sandbox/Dockerfile
SANDBOX_DB_DIR = "/usr/local/lib/python3.12/site-packages/nova_miner/combinatorial_db"


class SandboxCleanupError(RuntimeError):
    pass


def ensure_docker_image() -> None:
    try:
        bt.logging.info(f"Pulling Docker image {SANDBOX_IMAGE_TAG}…")
        subprocess.run(["docker", "pull", SANDBOX_IMAGE_TAG], check=True)
    except subprocess.CalledProcessError as e:
        bt.logging.warning(f"Pull failed for {SANDBOX_IMAGE_TAG}: {e}. Using local image if available.")
        subprocess.run(["docker", "image", "inspect", SANDBOX_IMAGE_TAG], check=True)

def prepare_workdir(challenge_params: dict, dest_dir: Path) -> Tuple[Path, Path]:
    if not DATA_OUTPUT_ROOT.is_mount():
        raise RuntimeError("Miner output volume is not mounted")
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix=dest_dir.name + "_", dir=dest_dir.parent))
    outdir = None
    try:
        workdir.chmod(0o755)
        outdir = Path(tempfile.mkdtemp(prefix=workdir.name + "_", dir=DATA_OUTPUT_ROOT))
        os.chown(outdir, SANDBOX_UID, -1)
        outdir.chmod(0o700)
        with (workdir / "input.json").open("w", encoding="utf-8") as f:
            json.dump(challenge_params, f)
        (workdir / "input.json").chmod(0o444)
        return workdir, outdir
    except OSError:
        for path in (outdir, workdir):
            if path is not None:
                shutil.rmtree(path)
        raise


def run_container(workdir: Path, outdir: Path, period: int, entry_id: str) -> Tuple[int, str]:
    timeout_seconds = load_time_budget_sec() 
    limits = load_sandbox_limits()
    output_volume = os.environ["MINER_OUTPUT_VOLUME"]
    # Translate container /data path to host path for docker -v
    host_runs_root = Path(os.environ.get("HOST_MINER_RUNS", str(DATA_WORK_ROOT)))
    try:
        rel = workdir.relative_to(DATA_WORK_ROOT)
        host_workdir = (host_runs_root / rel).resolve()
    except Exception:
        host_workdir = workdir

    # docker -v is resolved by the host daemon, so pass the host path.
    sock_path = workdir / "oracle.sock"
    host_sock = host_workdir / "oracle.sock"

    # Docker container names allow only [a-zA-Z0-9][a-zA-Z0-9_.-]; entry_id workdirs
    # contain '@' (hotkey@epoch), so sanitize anything outside that set.
    container_name = re.sub(r"[^a-z0-9_.-]", "-", f"miner-sbx-{workdir.name}".lower())
    # Ensure any previous container with the same name is removed
    try:
        subprocess.run(
            ["docker", "rm", "-f", "-v", container_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        bt.logging.warning(f"pre-run cleanup failed for {container_name}: {e}")

    cmd = [
        "docker", "run", "--rm",
        "--log-driver=json-file",
        "--log-opt=max-size=64m",
        "--log-opt=max-file=4",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt", "no-new-privileges:true",
        f"--memory={limits['memory']}",
        f"--memory-swap={limits['memory']}",
        f"--cpus={limits['cpus']}",
        f"--pids-limit={limits['pids_limit']}",
        "--tmpfs", f"/tmp:rw,noexec,nosuid,nodev,size={limits['tmp_size']}",
        "--tmpfs", f"{SANDBOX_DB_DIR}:rw,noexec,nosuid,nodev,mode=1777,size={limits['db_size']}",
        "--network=none",
        "--user", str(SANDBOX_UID),
        "-e", "HOME=/tmp",
        "-e", "XDG_CACHE_HOME=/tmp",
        "-e", "MPLCONFIGDIR=/tmp",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", "SQLITE_TMPDIR=/tmp",
        "-e", "WORKDIR=/workspace",
        "-e", "OUTPUT_DIR=/output",
        "-e", "ORACLE_SOCKET=/run/oracle.sock",
        "-v", f"{host_workdir / 'submission.tar.gz'}:/run/submission.tar.gz:ro",
        "-v", f"{host_workdir / 'input.json'}:/run/input.json:ro",
        "--mount", f"type=volume,src={output_volume},dst=/output,volume-subpath={outdir.name},volume-nocopy",
        "-v", f"{host_sock}:/run/oracle.sock",
        "--label", "app=nova_blueprint",
        "--label", "role=miner",
        "--label", f"period={period}",
        "--label", f"entry_id={entry_id}",
        "--name", container_name,
        SANDBOX_IMAGE_TAG,
    ]
    with Broker(sock_path, miner=entry_id, epoch=str(period)):
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            rc = proc.wait(timeout=timeout_seconds)
            return rc, ""
        except subprocess.TimeoutExpired:
            bt.logging.info(f"miner time budget reached entry={entry_id}")
            return 124, "timeout"
        finally:
            try:
                try:
                    if proc.poll() is None:
                        proc.kill()
                    proc.wait(timeout=10)
                finally:
                    subprocess.run(
                        ["docker", "rm", "-f", "-v", container_name],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=30,
                    )
                    # --rm may already have removed it. A successful list must confirm absence.
                    remaining = subprocess.run(
                        ["docker", "ps", "-aq", "--filter", f"name=^/{re.escape(container_name)}$"],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if remaining.stdout.strip():
                        raise RuntimeError(f"Container still exists: {container_name}")
            except Exception as e:
                raise SandboxCleanupError(f"Sandbox cleanup failed: {container_name}") from e
