import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Optional
import threading

from config.config_loader import load_time_budget_sec  


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SANDBOX_IMAGE_TAG = "miner-sandbox:dev"


def ensure_docker_image() -> None:
    try:
        subprocess.run(["docker", "image", "inspect", SANDBOX_IMAGE_TAG], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print(f"Building Docker image {SANDBOX_IMAGE_TAG}...")
        subprocess.run(["docker", "build", "-t", SANDBOX_IMAGE_TAG, "-f", str(PROJECT_ROOT / "sandbox" / "Dockerfile"), str(PROJECT_ROOT.parent)], check=True)


def prepare_workdir(source_dir: Path, challenge_params: dict, dest_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    work_root = PROJECT_ROOT / ".miner_runs"
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

    try:
        db_container_path = "/usr/local/lib/python3.12/site-packages/nova_ph2/combinatorial_db/molecules.sqlite"
        db_workspace_dir = workdir / "nova_ph2" / "combinatorial_db"
        db_workspace_dir.mkdir(parents=True, exist_ok=True)
        db_workspace_link = db_workspace_dir / "molecules.sqlite"
        if not db_workspace_link.exists():
            os.symlink(db_container_path, db_workspace_link)
    except Exception:
        pass

    try:
        os.chmod(workdir, 0o755)
        os.chmod(outdir, 0o750)
        for root, dirs, files in os.walk(workdir):
            try:
                os.chmod(root, 0o755)
            except Exception as e:
                print(f"[runner] chmod dir failed {root}: {e}")
            for d in dirs:
                p = os.path.join(root, d)
                try:
                    os.chmod(p, 0o755)
                except Exception as e:
                    print(f"[runner] chmod dir failed {p}: {e}")
            for f in files:
                p = os.path.join(root, f)
                try:
                    os.chmod(p, 0o644)
                except Exception as e:
                    print(f"[runner] chmod file failed {p}: {e}")
        try:
            st = os.stat(outdir)
            print(f"[runner] outdir perms set to {oct(st.st_mode & 0o777)} owner={st.st_uid} group={st.st_gid} path={outdir}")
        except Exception as e:
            print(f"[runner] stat outdir failed: {e}")
    except Exception as e:
        print(f"[runner] chmod outdir/workdir failed: {e}")

    try:
        res = subprocess.run(["setfacl", "-m", "u:10001:rwx", str(outdir)], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"[runner] setfacl applied on {outdir} for uid 10001")
        else:
            stderr = (res.stderr or "").strip()
            print(f"[runner] setfacl failed rc={res.returncode} on {outdir}: {stderr}")
    except FileNotFoundError:
        print("[runner] setfacl not found; ACL step skipped")
    except Exception as e:
        print(f"[runner] setfacl error: {e}")

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
    db_dir = "/usr/local/lib/python3.12/site-packages/nova_ph2/combinatorial_db"
    cmd = [
        "docker", "run", "--rm",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt", "no-new-privileges:true",
        "--tmpfs", "/tmp:rw,noexec,nosuid,nodev",
        "--gpus", "device=0",
        "--network=none",
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
        "-v", f"{workdir}:/workspace:ro",
        "-v", f"{outdir}:/output:rw",
        "-v", db_dir,
        SANDBOX_IMAGE_TAG,
    ]
    with open(workdir / "log.txt", "w", encoding="utf-8") as logf:
        logf.write("starting docker\n")
        logf.flush()
        ensure_docker_image()
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


 

 
 


