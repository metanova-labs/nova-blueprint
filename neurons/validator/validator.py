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

from sandbox import runner
from utils.challenge_params import build_challenge_params
from neurons.validator import scoring as scoring_module
from config.config_loader import load_config

from neurons.validator.commitments import get_commitments
import bittensor as bt

MAX_REPO_MB = 100

"""fetch commitments, run miners in sandbox, persist results."""

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
    hotkey: str
    coldkey: Optional[str] = None


def parse_commitment(raw: str, uid: int, block_number: int, hotkey: str) -> Optional[Miner]:
    match = COMMITMENT_REGEX.match(raw.strip())
    if not match:
        return None
    owner = match.group("owner")
    repo = match.group("repo")
    branch = match.group("branch")
    if len(owner) == 0 or len(repo) == 0 or len(branch) == 0:
        return None
    return Miner(uid=uid, block_number=block_number, raw=raw, owner=owner, repo=repo, branch=branch, hotkey=hotkey)


async def fetch_commitments_from_chain(network: Optional[str], netuid: int, min_block: int, max_block: int) -> List[Tuple[int, int, str, str]]:
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
    out: List[Tuple[int, int, str, str]] = []
    for c in commits.values():
        out.append((int(c.uid), int(c.block), str(c.data), str(c.hotkey)))
    return out


def to_miners(commitments: Iterable[Miner]) -> List[Miner]:
    return list(commitments)


def clone_repo(owner: str, repo: str, branch: str, work_root: Path) -> Path:
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

    subprocess.run(["git", "-C", str(target_dir), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    return target_dir


def ensure_miner_exists(repo_dir: Path) -> Path:
    miner_path = repo_dir / "miner.py"
    if not miner_path.is_file():
        raise FileNotFoundError("miner.py not found at repository root")
    return repo_dir


def write_run_artifacts(runs_root: Path, period: int, miner: Miner, result_obj: Optional[Dict]) -> None:
    if result_obj is None:
        return None
    results_dir = runs_root
    results_dir.mkdir(parents=True, exist_ok=True)

    combined = {
        "uid": miner.uid,
        "coldkey": miner.coldkey,
        "hotkey": miner.hotkey,
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
        bt.logging.error(f"aggregate write failed for period {period}: {e}")
        raise
    return None


def run_job(miner: Miner, runs_root: Path, work_root: Path, challenge_params: dict, period: int) -> None:
    started = time.time()
    repo_dir: Optional[Path] = None
    result_obj: Optional[Dict] = None
    exit_code: Optional[int] = None
    reason_on_fail: Optional[str] = None

    try:
        repo_dir = clone_repo(miner.owner, miner.repo, miner.branch, work_root)
        miner_dir = ensure_miner_exists(repo_dir)

        runner.ensure_docker_image()

        safe_repo = f"{miner.owner}_{miner.repo}".replace("/", "_")
        dest = work_root / f"{period}_{safe_repo}_{miner.uid}"
        workdir, outdir = runner.prepare_workdir(miner_dir, challenge_params, dest_dir=dest)
        bt.logging.info(f"cloning/running {miner.owner}/{miner.repo}@{miner.branch} uid={miner.uid} workdir={workdir}")
        code, output = runner.run_container(workdir, outdir)
        bt.logging.info(f"run finished uid={miner.uid} exit={code} log={outdir / 'log.txt'} result={outdir / 'result.json'}")
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
        bt.logging.error(f"run failed uid={miner.uid}: {type(e).__name__}: {e}")
    finally:
        if repo_dir is not None:
            try:
                shutil.rmtree(repo_dir, ignore_errors=True)
            except Exception:
                pass
        try:
            bt.logging.info(f"finished uid={miner.uid} workdir={workdir if 'workdir' in locals() else 'n/a'}")
        except Exception:
            pass

    write_run_artifacts(runs_root, period, miner, result_obj)


def gather_parse_and_schedule(commit_quads: Iterable[Tuple[int, int, str, str]]) -> List[Miner]:
    parsed: List[Miner] = []
    for uid, block_number, raw, hotkey in commit_quads:
        c = parse_commitment(raw, uid, block_number, hotkey)
        if c is not None:
            parsed.append(c)
    miners = to_miners(parsed)
    miners.sort(key=lambda m: (m.block_number, m.uid))
    return miners


async def main() -> int:
    runs_root = Path("/data/results").resolve()
    work_root = Path("/data/miner_runs").resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    load_dotenv(PROJECT_ROOT / ".env")

    network = os.environ.get("SUBTENSOR_NETWORK")
    netuid = int(os.environ.get("NETUID", "68"))

    subtensor = bt.async_subtensor(network=network)
    await subtensor.initialize()
    current_block = await subtensor.get_current_block()

    cfg_all = load_config()
    interval_seconds = int(cfg_all["competition_interval_seconds"]) 
    now_ts = int(time.time())
    period_index = now_ts // interval_seconds
    period_start_ts = period_index * interval_seconds
    period = period_index

    approx_block_time_s = 12
    blocks_window = max(1, interval_seconds // approx_block_time_s)
    min_block = max(0, current_block - blocks_window)
    max_block = current_block
    bt.logging.info(
        f"period_index={period} start_utc={dt.datetime.fromtimestamp(period_start_ts, dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}Z "
        f"interval_seconds={interval_seconds} window_blocks≈{blocks_window} "
        f"min_block={min_block} max_block={max_block}"
    )

    submissions = await fetch_commitments_from_chain(network=network, netuid=netuid, min_block=min_block, max_block=max_block)
    miners = gather_parse_and_schedule(submissions)
    bt.logging.info(f"current_block={current_block} submissions={len(submissions)} miners={len(miners)}")

    block_hash = await subtensor.determine_block_hash(current_block)
    challenge_params = build_challenge_params(str(block_hash))

    try:
        benchmark = Miner(
            uid=-1,
            block_number=current_block,
            raw="nova68miner/random_miner@main",
            owner="nova68miner",
            repo="random_miner",
            branch="main",
            hotkey="benchmark",
        )
        bt.logging.info("benchmark: running nova68miner/random_miner@main (uid=0)")
        run_job(benchmark, runs_root=runs_root, work_root=work_root, challenge_params=challenge_params, period=period)
    except Exception as e:
        bt.logging.error(f"benchmark run failed: {type(e).__name__}: {e}")

    try:
        metagraph = await subtensor.metagraph(netuid)
        coldkeys = getattr(metagraph, 'coldkeys', None)
        if coldkeys is not None:
            for miner in miners:
                if isinstance(miner.uid, int) and 0 <= miner.uid < len(coldkeys):
                    miner.coldkey = coldkeys[miner.uid]
    except Exception as e:
        bt.logging.error(f"failed to populate coldkeys: {type(e).__name__}: {e}")
    for miner in miners:
        run_job(miner, runs_root=runs_root, work_root=work_root, challenge_params=challenge_params, period=period)

    try:
        jsonl_path = (Path("/data/results") / f"period_{period}_results.jsonl")
        uid_to_data: Dict[int, Dict] = {}
        if jsonl_path.exists():
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    uid = int(rec["uid"]) if "uid" in rec else None
                    if uid is None or uid < 0:
                        continue
                    molecules = rec.get("result", {}).get("molecules", [])
                    uid_to_data[uid] = {
                        "molecules": molecules,
                        "github_data": rec.get("raw"),
                        "hotkey": rec.get("hotkey"),
                        "coldkey": rec.get("coldkey"),
                    }
        cfg = dict(challenge_params.get("config", {}))
        cfg.update(challenge_params.get("challenge", {}))

        winner_uid, winner_score = await scoring_module.process_epoch(cfg, period, uid_to_data)
        # Persist winner: overwrite each run
        try:
            if isinstance(winner_uid, int):
                win = uid_to_data.get(winner_uid, {})
                winner_obj = {
                    "uid": winner_uid,
                    "hotkey": win.get("hotkey"),
                    "coldkey": win.get("coldkey"),
                    "score": winner_score,
                    "updated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                }
                out_dir = Path("/data/results").resolve()
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / "winner.json"
                tmp_path = out_dir / "winner.json.tmp"
                with tmp_path.open("w", encoding="utf-8") as f:
                    json.dump(winner_obj, f, separators=(",", ":"))
                os.replace(tmp_path, out_path)
                bt.logging.info(f"winner persisted uid={winner_uid} at {out_path}")
        except Exception as e:
            bt.logging.error(f"failed to persist winner: {type(e).__name__}: {e}")
    except Exception as e:
        bt.logging.error(f"scoring step failed: {e}")

    return 0


if __name__ == "__main__":
    try:
        import asyncio
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)


