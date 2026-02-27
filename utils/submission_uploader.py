from __future__ import annotations

import base64
import hashlib
import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests

REQUEST_TIMEOUT_SEC = 10
DEFAULT_API_URL = "https://submission-api.metanova-labs.ai"
DEFAULT_WALLET_PATH = "~/.bittensor/wallets"


@dataclass(frozen=True)
class UploadResult:
    status_code: int
    ok: bool
    request_id: str | None
    body: dict | None
    text: str


def _canonical_message(hotkey: str, code_hash: str) -> bytes:
    payload = {"hotkey": hotkey, "code_hash": code_hash}
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _code_hash_sha256(code_bytes: bytes) -> str:
    return "sha256:" + hashlib.sha256(code_bytes).hexdigest()


def _tar_gz_from_directory(root: Path) -> bytes:
    root = root.resolve()
    miner_py = root / "miner.py"
    if not miner_py.is_file():
        raise ValueError(f"Directory must contain miner.py at root: {root}")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(root, arcname=root.name)
    return buf.getvalue()

def _parse_github_repo(github_url: str) -> str:
    raw = github_url.strip()
    if not raw:
        raise ValueError("github_url is required")
    if "://" in raw:
        parsed = urlparse(raw)
        parts = [p for p in parsed.path.strip("/").split("/") if p]
    else:
        parts = [p for p in raw.strip("/").split("/") if p]
    if len(parts) < 2:
        raise ValueError("GitHub URL must include owner/repo")

    owner = parts[0]
    repo = parts[1].replace(".git", "")
    return f"{owner}/{repo}"


def _download_github_main_tarball(github_url: str) -> bytes:
    repo = _parse_github_repo(github_url)
    tar_url = f"https://codeload.github.com/{repo}/tar.gz/main"
    resp = requests.get(tar_url, timeout=90)
    resp.raise_for_status()
    return resp.content


def _load_wallet_keypair(wallet_name: str, hotkey_name: str, wallet_path: str):
    import bittensor as bt

    wallet = bt.Wallet(
        name=wallet_name.strip(),
        hotkey=hotkey_name.strip(),
        path=str(Path(wallet_path).expanduser()),
    )
    if wallet.hotkey is None:
        raise RuntimeError("Failed to load wallet hotkey keypair")
    return wallet.hotkey


def _sign_message_hex(keypair, message: bytes) -> str:
    sig_raw = keypair.sign(message)
    if isinstance(sig_raw, str):
        return "0x" + sig_raw.replace("0x", "")
    return "0x" + bytes(sig_raw).hex()


def _post_submission(
    *,
    api_url: str,
    code_bytes: bytes,
    wallet_name: str,
    hotkey_name: str,
    wallet_path: str,
    submission_name: str = "",
) -> UploadResult:
    keypair = _load_wallet_keypair(
        wallet_name=wallet_name,
        hotkey_name=hotkey_name,
        wallet_path=wallet_path,
    )
    hotkey_ss58 = keypair.ss58_address
    code_hash = _code_hash_sha256(code_bytes)
    signature = _sign_message_hex(keypair, _canonical_message(hotkey_ss58, code_hash))

    payload = {
        "hotkey": hotkey_ss58,
        "code": base64.b64encode(code_bytes).decode("utf-8"),
        "code_hash": code_hash,
        "signature": signature,
    }
    chosen_name = submission_name.strip()
    if chosen_name:
        payload["submission_name"] = chosen_name

    resp = requests.post(
        api_url.rstrip("/") + "/submit",
        json=payload,
        timeout=REQUEST_TIMEOUT_SEC,
    )

    try:
        body = resp.json()
    except Exception:
        body = None

    return UploadResult(
        status_code=resp.status_code,
        ok=resp.status_code < 400,
        request_id=resp.headers.get("x-request-id"),
        body=body,
        text=resp.text,
    )


def submit_from_local_path(
    *,
    local_path: str,
    wallet_name: str,
    hotkey_name: str,
    submission_name: str = "",
    api_url: str = DEFAULT_API_URL,
    wallet_path: str = DEFAULT_WALLET_PATH,
) -> UploadResult:
    path = Path(local_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if path.is_file():
        if not path.name.endswith(".tar.gz"):
            raise ValueError("File input must be a .tar.gz archive")
        code_bytes = path.read_bytes()
    elif path.is_dir():
        code_bytes = _tar_gz_from_directory(path)
    else:
        raise ValueError(f"Unsupported path type: {path}")

    return _post_submission(
        api_url=api_url,
        code_bytes=code_bytes,
        wallet_name=wallet_name,
        hotkey_name=hotkey_name,
        wallet_path=wallet_path,
        submission_name=submission_name,
    )


def submit_from_github_url(
    *,
    github_url: str,
    wallet_name: str,
    hotkey_name: str,
    submission_name: str = "",
    api_url: str = DEFAULT_API_URL,
    wallet_path: str = DEFAULT_WALLET_PATH,
) -> UploadResult:
    code_bytes = _download_github_main_tarball(github_url)
    return _post_submission(
        api_url=api_url,
        code_bytes=code_bytes,
        wallet_name=wallet_name,
        hotkey_name=hotkey_name,
        wallet_path=wallet_path,
        submission_name=submission_name,
    )
