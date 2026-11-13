import os
from pathlib import Path
from typing import Optional, Dict

from minio import Minio
from minio.error import S3Error


SNAPSHOT_BUCKET = "blueprint-boltz-code-archive"
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


def _resolve_snapshot_key(epoch: int, hotkey: str) -> Optional[str]:
    """
    Resolve the object key in MinIO for a given (epoch, hotkey) using the configured bucket.
    Returns the key string, or None if no matching object exists.
    """
    client = _create_minio_client()
    bucket_name = SNAPSHOT_BUCKET

    key = f"{int(epoch)}/{hotkey}.tar.gz"
    try:
        client.stat_object(bucket_name, key)
        return key
    except S3Error as e:
        if e.code in {"NoSuchKey", "NoSuchObject"}:
            return None
        raise


def download_snapshot(epoch: int, hotkey: str, archive_path: Path) -> Optional[Path]:
    key = _resolve_snapshot_key(epoch=epoch, hotkey=hotkey)
    if key is None:
        return None
    _create_minio_client().fget_object(SNAPSHOT_BUCKET, key, str(archive_path))
    archive_path.chmod(0o444)
    return archive_path


def download_benchmark_snapshot(archive_path: Path, name: str) -> Path:
    object_key = BENCHMARK_OBJECT_KEYS.get(name)
    if object_key is None:
        raise ValueError(f"Unknown benchmark snapshot name: {name}")
    _create_minio_client().fget_object(SNAPSHOT_BUCKET, object_key, str(archive_path))
    archive_path.chmod(0o444)
    return archive_path
