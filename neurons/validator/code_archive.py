import os
import tarfile
import tempfile
from pathlib import Path
from typing import Optional, Dict

import requests
from minio import Minio
import shutil


SNAPSHOT_BUCKET = "blueprint-code-archive"
MAX_REPO_MB = 100
BENCHMARK_OBJECT_KEY = "benchmarks/brute_force.tar.gz"


def _create_minio_client() -> Minio:
    """
    Create a MinIO client using MINIO_ACCESS_KEY, MINIO_SECRET_KEY and SNAPSHOT_S3_ENDPOINT from env.
    """
    access_key = os.environ.get("MINIO_ACCESS_KEY")
    secret_key = os.environ.get("MINIO_SECRET_KEY")
    if not access_key or not secret_key:
        raise ValueError("Missing MinIO credentials in environment: MINIO_ACCESS_KEY / MINIO_SECRET_KEY")

    ep = os.environ.get("SNAPSHOT_S3_ENDPOINT")
    if not ep:
        raise ValueError("Missing snapshot endpoint in env: SNAPSHOT_S3_ENDPOINT")
    return Minio(ep, access_key=access_key, secret_key=secret_key, secure=True)


def _resolve_snapshot_key(epoch: int, uid: int) -> Optional[str]:
    """
    Resolve the object key in MinIO for a given (epoch, uid) using the configured bucket.
    Returns the key string, or None if no matching object exists.
    """
    client = _create_minio_client()
    bucket_name = SNAPSHOT_BUCKET

    prefix = f"{epoch}/"
    for obj in client.list_objects(bucket_name, prefix=prefix, recursive=False):
        name = obj.object_name
        try:
            base = name.split("/", 1)[1]
        except Exception:
            base = name
        if base.startswith(f"{uid}_"):
            return name
    return None


def _upload_snapshot_archive(
    epoch: int,
    uid: int,
    archive_path: Path,
) -> str:
    """
    Upload a snapshot archive to MinIO for (epoch, uid) into the configured bucket.
    Returns the object key.
    """
    client = _create_minio_client()
    bucket_name = SNAPSHOT_BUCKET

    archive_path = archive_path.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Snapshot archive not found: {archive_path}")

    filename = archive_path.name
    object_name = f"{epoch}/{filename}"

    size = archive_path.stat().st_size
    with archive_path.open("rb") as f:
        client.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=f,
            length=size,
            content_type="application/gzip",
        )

    return object_name


def download_and_extract_snapshot(
    epoch: int,
    uid: int,
    work_root: Path,
    dest_dir: Path,
) -> Optional[Path]:
    """
    Download the archived snapshot for (epoch, uid) from MinIO and extract it into dest_dir.

    Returns the extracted repository directory Path on success, or None if no snapshot exists.
    Raises on hard MinIO errors or filesystem issues.
    """
    client = _create_minio_client()
    bucket_name = SNAPSHOT_BUCKET

    key = _resolve_snapshot_key(epoch=epoch, uid=uid)
    if not key:
        return None

    work_root = work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    archive_path = work_root / f"{epoch}_{uid}_snapshot.tar.gz"

    client.fget_object(bucket_name, key, str(archive_path))

    dest_dir = dest_dir.resolve()
    if dest_dir.exists():
        import shutil

        shutil.rmtree(dest_dir, ignore_errors=True)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_dir = dest_dir

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=target_dir)
    try:
        archive_path.unlink(missing_ok=True)  # type: ignore[call-arg]
    except TypeError:
        # Python < 3.8 compatibility (missing_ok not supported)
        try:
            if archive_path.exists():
                archive_path.unlink()
        except Exception:
            pass

    return target_dir


def clone_repo(owner: str, repo: str, branch: str, work_root: Path, target_dir: Path) -> Path:
    """
    Shallow-clone a GitHub repo branch into target_dir, enforcing a simple max-size check.
    """
    repo_url = f"https://github.com/{owner}/{repo}.git"
    target_dir = target_dir.resolve()
    if target_dir.exists():
        import shutil

        shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        api_url = f"https://api.github.com/repos/{owner}/{repo}"
        headers: Dict[str, str] = {}
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        resp = requests.get(api_url, headers=headers, timeout=5)
        if resp.ok:
            size_kb = int(resp.json().get("size", 0))
            if (size_kb / 1024.0) > MAX_REPO_MB:
                raise RuntimeError(
                    f"Repo {owner}/{repo} reported size {size_kb/1024:.1f} MiB exceeds limit {MAX_REPO_MB} MiB"
                )
    except Exception:
        pass

    import subprocess

    subprocess.run([
        "git",
        "-c", "filter.lfs.smudge=",
        "-c", "filter.lfs.required=false",
        "clone", "--depth", "1", "--single-branch", "--branch", branch,
        repo_url, str(target_dir)
    ], check=True)

    subprocess.run(["git", "-C", str(target_dir), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    return target_dir


def build_snapshot_archive(repo_dir: Path, archive_path: Path) -> Path:
    """
    Create a tar.gz snapshot archive for the given repo_dir at archive_path.
    """
    archive_path = archive_path.resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        for root, dirs, files in os.walk(repo_dir):
            rel_root = os.path.relpath(root, start=repo_dir)
            if rel_root == ".":
                rel_root = ""
            for skip in [".git", ".hg", ".svn", "__pycache__", ".github"]:
                if skip in dirs:
                    dirs.remove(skip)
            for name in sorted(files):
                if name.endswith(".pyc") or name.endswith(".pyo"):
                    continue
                full_path = Path(root) / name
                rel_path = os.path.normpath(os.path.join(rel_root, name))
                tar.add(full_path, arcname=rel_path, recursive=False)

    return archive_path


def download_benchmark_snapshot(work_root: Path, dest_dir: Path) -> Path:
    """
    Download and extract the benchmark snapshot into dest_dir.
    """
    client = _create_minio_client()
    bucket = SNAPSHOT_BUCKET

    work_root = work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    archive_path = work_root / "benchmark_brute_force.tar.gz"

    client.fget_object(bucket, BENCHMARK_OBJECT_KEY, str(archive_path))

    dest_dir = dest_dir.resolve()
    if dest_dir.exists():
        shutil.rmtree(dest_dir, ignore_errors=True)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=dest_dir)
    try:
        archive_path.unlink(missing_ok=True) 
    except TypeError:
        if archive_path.exists():
            archive_path.unlink()

    return dest_dir


def upload_miner_snapshot(
    owner: str,
    repo: str,
    branch: str,
    uid: int,
    epoch: int,
) -> str:
    """
    Clone, archive, upload a miner repo for a given epoch, then clean up locally.
    Returns the uploaded object key.
    """
    tmp_root = Path(tempfile.mkdtemp(prefix=f"snapshot-{epoch}-{uid}-"))
    repo_dir = tmp_root / "repo"
    safe_repo = f"{owner}_{repo}@{branch}".replace("/", "_")
    archive_path = tmp_root / f"{uid}_{safe_repo}.tar.gz"
    try:
        clone_repo(owner=owner, repo=repo, branch=branch, work_root=tmp_root, target_dir=repo_dir)
        build_snapshot_archive(repo_dir=repo_dir, archive_path=archive_path)
        key = _upload_snapshot_archive(epoch=epoch, uid=uid, archive_path=archive_path)
    finally:
        import shutil

        shutil.rmtree(tmp_root, ignore_errors=True)
    return key



