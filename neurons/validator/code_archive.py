import os
import tarfile
from pathlib import Path
from typing import Optional, Dict

from minio import Minio
from minio.error import S3Error
import shutil


SNAPSHOT_BUCKET = "blueprint-code-archive"
BENCHMARK_OBJECT_KEYS: Dict[str, str] = {
    "brute_force": "benchmarks/brute_force.tar.gz",
    "thompson_sampling": "benchmarks/thompson_sampling.tar.gz",
}


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

    key = f"{int(epoch)}/{int(uid)}.tar.gz"
    try:
        client.stat_object(bucket_name, key)
        return key
    except S3Error as e:
        if e.code in {"NoSuchKey", "NoSuchObject"}:
            return None
        raise


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

def download_benchmark_snapshot(
    work_root: Path,
    dest_dir: Path,
    name: str,
) -> Path:
    """
    Download and extract a benchmark snapshot into dest_dir.
    """
    client = _create_minio_client()
    bucket = SNAPSHOT_BUCKET

    object_key = BENCHMARK_OBJECT_KEYS.get(str(name))
    if not object_key:
        raise ValueError(
            f"Unknown benchmark snapshot name '{name}'. "
            f"Known: {sorted(BENCHMARK_OBJECT_KEYS.keys())}"
        )

    work_root = work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    archive_path = work_root / f"benchmark_{name}.tar.gz"

    client.fget_object(bucket, object_key, str(archive_path))

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



