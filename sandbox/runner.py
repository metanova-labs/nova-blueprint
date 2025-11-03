import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Optional
import threading

from config.config_loader import load_time_budget_sec  
import bittensor as bt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_WORK_ROOT = Path("/data/miner_runs")
DATA_RESULTS_ROOT = Path("/data/results")

SANDBOX_IMAGE_TAG = "urdof7/miner-sandbox:latest"


def ensure_docker_image() -> None:
    try:
        bt.logging.info(f"Pulling Docker image {SANDBOX_IMAGE_TAG}…")
        subprocess.run(["docker", "pull", SANDBOX_IMAGE_TAG], check=True)
    except subprocess.CalledProcessError as e:
        bt.logging.warning(f"Pull failed for {SANDBOX_IMAGE_TAG}: {e}. Using local image if available.")
        subprocess.run(["docker", "image", "inspect", SANDBOX_IMAGE_TAG], check=True)


def prepare_workdir(source_dir: Path, challenge_params: dict, dest_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    work_root = DATA_WORK_ROOT
    work_root.mkdir(parents=True, exist_ok=True)
    if dest_dir is None:
        workdir = Path(tempfile.mkdtemp(prefix="run_", dir=str(work_root)))
    else:
        workdir = Path(dest_dir)
        workdir.mkdir(parents=True, exist_ok=True)
    outdir = workdir / "out"
    outdir.mkdir(parents=True, exist_ok=True)

    if not (source_dir / "miner.py").is_file():
        raise FileNotFoundError(f"miner.py not found in {source_dir}")
    for entry in source_dir.iterdir():
        if entry.name in {".git", ".hg", ".svn", "__pycache__"}:
            continue
        dest = workdir / entry.name
        if entry.is_dir():
            shutil.copytree(entry, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, dest)

    with open(workdir / "input.json", "w", encoding="utf-8") as f:
        json.dump(challenge_params, f)

    # Ensure outdir is writable by the sandbox user
    try:
        host_uid = os.stat(PROJECT_ROOT).st_uid
        host_gid = os.stat(PROJECT_ROOT).st_gid
        os.chown(outdir, host_uid, host_gid)
    except Exception:
        pass


    return workdir, outdir


def run_container(workdir: Path, outdir: Path) -> Tuple[int, str]:
    timeout_seconds = load_time_budget_sec() 
    def _to_str(x) -> str:
        if x is None:
            return ""
        if isinstance(x, (bytes, bytearray)):
            try:
                return x.decode("utf-8", errors="replace")
            except Exception:
                return str(x)
        return str(x)
    # Translate container /data path to host path for docker -v
    host_runs_root = Path(os.environ.get("HOST_MINER_RUNS", str(DATA_WORK_ROOT)))
    try:
        rel = workdir.relative_to(DATA_WORK_ROOT)
        host_workdir = (host_runs_root / rel).resolve()
    except Exception:
        host_workdir = workdir
    try:
        rel_out = outdir.relative_to(DATA_WORK_ROOT)
        host_outdir = (host_runs_root / rel_out).resolve()
    except Exception:
        host_outdir = outdir
    host_uid = os.stat(PROJECT_ROOT).st_uid
    host_gid = os.stat(PROJECT_ROOT).st_gid
    cmd = [
        "docker", "run", "--rm",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt", "no-new-privileges:true",
        "--tmpfs", "/tmp:rw,noexec,nosuid,nodev",
        "--gpus", "device=0",
        "--network=none",
        "--user", f"{host_uid}:{host_gid}",
        "-e", "HOME=/tmp",
        "-e", "XDG_CACHE_HOME=/tmp",
        "-e", "HF_HOME=/tmp",
        "-e", "TORCH_HOME=/opt/torch_cache",
        "-e", "TRANSFORMERS_CACHE=/tmp",
        "-e", "MPLCONFIGDIR=/tmp",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", "SQLITE_TMPDIR=/tmp",
        "-e", "WORKDIR=/workspace",
        "-e", "OUTPUT_DIR=/output",
        "-v", f"{host_workdir}:/workspace:ro",
        "-v", f"{host_outdir}:/output:rw",
        SANDBOX_IMAGE_TAG,
    ]
    with open(workdir / "log.txt", "w", encoding="utf-8") as logf:
        logf.write("starting docker\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def _pump():
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    logf.write(_to_str(line))
                    logf.flush()
            except Exception:
                pass

        t = threading.Thread(target=_pump, daemon=True)
        t.start()
        try:
            rc = proc.wait(timeout=timeout_seconds)
            t.join(timeout=2)
            return rc, ""
        except subprocess.TimeoutExpired:
            try:
                logf.write("timeout\n")
                logf.flush()
            except Exception:
                pass
            try:
                proc.kill()
            except Exception:
                pass
            t.join(timeout=2)
            return 124, "timeout"


 

 
 


