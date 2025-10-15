import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sandbox import runner
from utils.challenge_params import build_challenge_params
from neurons.validator import scoring as scoring_module
from config.config_loader import load_period_duration

from neurons.validator.commitments import get_commitments
import bittensor as bt

MAX_REPO_MB = 100

"""Orchestrator: fetch commitments, run miners in sandbox, persist results."""

COMMITMENT_REGEX = re.compile(
    r"^(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)@(?P<branch>[\w./-]+)$"
)

@dataclass
class Miner:
    uid: int
    block_number: int
    raw: str
    owner: str
    repo: str
    branch: str


def parse_commitment(raw: str, uid: int, block_number: int) -> Optional[Miner]:
    match = COMMITMENT_REGEX.match(raw.strip())
    if not match:
        return None
    owner = match.group("owner")
    repo = match.group("repo")
    branch = match.group("branch")
    if len(owner) == 0 or len(repo) == 0 or len(branch) == 0:
        return None
    return Miner(uid=uid, block_number=block_number, raw=raw, owner=owner, repo=repo, branch=branch)


async def fetch_commitments_from_chain(network: Optional[str], netuid: int, min_block: int, max_block: int) -> List[Tuple[int, int, str]]:
    """Fetch plaintext commitments within a block window (one per UID)."""
    subtensor = bt.async_subtensor(network=network)
    await subtensor.initialize()
    metagraph = await subtensor.metagraph(netuid)
    block_hash = await subtensor.determine_block_hash(max_block)
    commits = await get_commitments(
        subtensor=subtensor,
        metagraph=metagraph,
        block_hash=block_hash,
        netuid=netuid,
        min_block=min_block,
        max_block=max_block,
    )
    out: List[Tuple[int, int, str]] = []
    for c in commits.values():
        out.append((int(c.uid), int(c.block), str(c.data)))
    return out


def to_miners(commitments: Iterable[Miner]) -> List[Miner]:
    return list(commitments)


def clone_repo(owner: str, repo: str, branch: str, work_root: Path) -> Tuple[Path, Optional[str]]:
    repo_url = f"https://github.com/{owner}/{repo}.git"
    target_dir = Path(tempfile.mkdtemp(prefix=f"{owner}-{repo}-", dir=str(work_root)))

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

    subprocess.run([
        "git",
        "-c", "filter.lfs.smudge=",
        "-c", "filter.lfs.required=false",
        "clone", "--depth", "1", "--single-branch", "--branch", branch,
        repo_url, str(target_dir)
    ], check=True)

    completed = subprocess.run(["git", "-C", str(target_dir), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    sha = completed.stdout.strip()
    return target_dir, sha


def ensure_miner_exists(repo_dir: Path) -> Path:
    miner_path = repo_dir / "miner.py"
    if not miner_path.is_file():
        raise FileNotFoundError("miner.py not found at repository root")
    return repo_dir


def write_run_artifacts(runs_root: Path, period: str, miner: Miner, result_obj: Optional[Dict]) -> Optional[Path]:
    if result_obj is None:
        return None
    results_dir = runs_root
    results_dir.mkdir(parents=True, exist_ok=True)

    combined = {
        "uid": miner.uid,
        "block_number": miner.block_number,
        "owner": miner.owner,
        "repo": miner.repo,
        "branch": miner.branch,
        "raw": miner.raw,
        "result": result_obj,
    }
    try:
        out_file = results_dir / f"period_{period}_results.jsonl"
        with out_file.open("a", encoding="utf-8") as agg:
            agg.write(json.dumps(combined, separators=(",", ":")) + "\n")
    except Exception as e:
        print(f"[orchestrator] aggregate write failed for period {period}: {e}")
        raise
    return None


def run_job(miner: Miner, runs_root: Path, work_root: Path, challenge_params: dict, period: str) -> None:
    started = time.time()
    repo_dir: Optional[Path] = None
    commit_sha: Optional[str] = None
    result_obj: Optional[Dict] = None
    exit_code: Optional[int] = None
    reason_on_fail: Optional[str] = None

    try:
        repo_dir, commit_sha = clone_repo(miner.owner, miner.repo, miner.branch, work_root)
        miner_dir = ensure_miner_exists(repo_dir)

        runner.ensure_docker_image()

        safe_repo = f"{miner.owner}_{miner.repo}".replace("/", "_")
        dest = work_root / f"{period}_{safe_repo}_{miner.uid}"
        workdir, outdir = runner.prepare_workdir(miner_dir, challenge_params, dest_dir=dest)
        print(f"[orchestrator] cloning/running {miner.owner}/{miner.repo}@{miner.branch} uid={miner.uid} workdir={workdir}")
        code, output = runner.run_container(workdir, outdir)
        print(f"[orchestrator] run finished uid={miner.uid} exit={code} log={outdir / 'log.txt'} result={outdir / 'result.json'}")
        exit_code = code
        try:
            with open(outdir / "result.json", "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict) and "result" in raw and isinstance(raw["result"], dict):
                result_obj = raw["result"]
            elif isinstance(raw, dict):
                result_obj = raw
        except Exception:
            result_obj = None

    except Exception as e:
        reason_on_fail = f"exception: {type(e).__name__}: {e}"
        print(f"[orchestrator] run failed uid={miner.uid}: {type(e).__name__}: {e}")
    finally:
        if repo_dir is not None:
            try:
                shutil.rmtree(repo_dir, ignore_errors=True)
            except Exception:
                pass
        try:
            print(f"[orchestrator] finished uid={miner.uid} workdir={workdir if 'workdir' in locals() else 'n/a'}")
        except Exception:
            pass

    write_run_artifacts(runs_root, period, miner, result_obj)


def gather_parse_and_schedule(commit_triplets: Iterable[Tuple[int, int, str]]) -> List[Miner]:
    parsed: List[Miner] = []
    for uid, block_number, raw in commit_triplets:
        c = parse_commitment(raw, uid, block_number)
        if c is not None:
            parsed.append(c)
    miners = to_miners(parsed)
    miners.sort(key=lambda m: (m.block_number, m.uid))
    return miners


async def main() -> int:
    runs_root = (PROJECT_ROOT / "results").resolve()
    work_root = (PROJECT_ROOT / ".miner_runs").resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    load_dotenv(PROJECT_ROOT / ".env")

    network = os.environ.get("SUBTENSOR_NETWORK")
    netuid = int(os.environ.get("NETUID", "68"))

    subtensor = bt.async_subtensor(network=network)
    await subtensor.initialize()
    current_block = await subtensor.get_current_block()
    period_blocks = load_period_duration()
    period = str(current_block // period_blocks)
    min_block = max(0, current_block - period_blocks)
    max_block = current_block

    submissions = await fetch_commitments_from_chain(network=network, netuid=netuid, min_block=min_block, max_block=max_block)
    miners = gather_parse_and_schedule(submissions)
    print(f"[orchestrator] current_block={current_block} submissions={len(submissions)} miners={len(miners)}")

    block_hash = await subtensor.determine_block_hash(current_block)
    challenge_params = build_challenge_params(str(block_hash))
    for miner in miners:
        run_job(miner, runs_root=runs_root, work_root=work_root, challenge_params=challenge_params, period=period)

    try:
        jsonl_path = (PROJECT_ROOT / "results" / f"period_{period}_results.jsonl")
        uid_to_data: Dict[int, Dict] = {}
        if jsonl_path.exists():
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    uid = int(rec["uid"]) if "uid" in rec else None
                    if uid is None:
                        continue
                    molecules = rec.get("result", {}).get("molecules", [])
                    uid_to_data[uid] = {
                        "molecules": molecules,
                        "github_data": rec.get("raw"),
                    }
        cfg = dict(challenge_params.get("config", {}))
        cfg.update(challenge_params.get("challenge", {}))

        await scoring_module.process_epoch(cfg, period, uid_to_data)
    except Exception as e:
        print(f"[validator] scoring step failed: {e}")

    return 0


if __name__ == "__main__":
    try:
        import asyncio
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)


