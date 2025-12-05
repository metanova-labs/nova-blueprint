import datetime as dt
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from dotenv import load_dotenv
import asyncio

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from sandbox import runner
from utils.challenge_params import build_challenge_params
from neurons.validator import scoring as scoring_module
from neurons.validator.code_archive import (
    download_and_extract_snapshot,
    download_benchmark_snapshot,
    upload_miner_snapshot,
)
from config.config_loader import load_config

from neurons.validator.commitments import get_commitments
import bittensor as bt

"""fetch commitments, run miners in sandbox, persist results."""

COMMITMENT_REGEX = re.compile(
    r"^(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)@(?P<branch>[\w./-]+)$"
)

BENCHMARK_GITHUB = os.environ.get("BENCHMARK_GITHUB", "nova68miner/random_miner@main")

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


async def call_st(subtensor, network: Optional[str], rpc_fn, timeout_s: int = 10):
    """
    Reuse the provided async_subtensor for an RPC under timeout.
    On timeout/exception, close and recreate the client and retry once.
    """
    try:
        res = await asyncio.wait_for(rpc_fn(subtensor), timeout=timeout_s)
        return res, subtensor
    except (asyncio.TimeoutError, Exception) as e:
        bt.logging.warning(f"Subtensor RPC reconnect triggered due to {type(e).__name__}: {e}")
        await subtensor.close()
        st = bt.async_subtensor(network=network)
        await st.initialize()
        res = await asyncio.wait_for(rpc_fn(st), timeout=timeout_s)
        return res, st


async def fetch_commitments_from_chain(network: Optional[str], netuid: int, min_block: int, max_block: int) -> List[Tuple[int, int, str, str]]:
    """Fetch plaintext commitments within a block window (one per UID)."""
    subtensor = bt.async_subtensor(network=network)
    await subtensor.initialize()
    metagraph, subtensor = await call_st(subtensor, network, lambda st: st.metagraph(netuid), timeout_s=10)
    block_hash, subtensor = await call_st(subtensor, network, lambda st: st.determine_block_hash(max_block), timeout_s=10)
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


def commitment_to_clone(raw: Optional[str]):
    """
    Convert commitment string 'owner/repo@branch' to (clone_url, branch).
    Returns (None, None) if parsing fails.
    """
    if not isinstance(raw, str) or "/" not in raw or "@" not in raw:
        return None, None
    try:
        owner, rest = raw.split("/", 1)
        repo, branch = rest.split("@", 1)
        return f"https://github.com/{owner}/{repo}.git", branch
    except Exception:
        return None, None


def get_previous_winner(current_block: int) -> Optional[tuple[Miner, int]]:
    """
    Read previous winner once and build a Miner with the original UID.
    Returns (Miner, snapshot_epoch) or None if no previous winner is recorded.
    """
    try:
        winner_json_path = Path("/data/results/winner.json")
        if not winner_json_path.exists():
            bt.logging.info("previous winner: no previous winner found; skipping")
            return None
        with winner_json_path.open("r", encoding="utf-8") as f:
            last_win = json.load(f)
        uid_val = int(last_win["uid"])
        raw = str(last_win["raw"])
        hotkey_val = last_win["hotkey"]
        snapshot_epoch_val = last_win["snapshot_epoch"]
        snapshot_epoch_int = int(snapshot_epoch_val)
        prev_winner = parse_commitment(raw, uid=uid_val, block_number=current_block, hotkey=hotkey_val)
        if prev_winner is None:
            bt.logging.info("previous winner: malformed winner.json (commitment parse failed); skipping")
            return None
        return prev_winner, snapshot_epoch_int
    except Exception as e:
        bt.logging.error(f"previous winner: failed: {type(e).__name__}: {e}")
        return None


def inject_previous_winner(miners: List[Miner], prev: Optional[Miner]) -> tuple[List[Miner], Optional[int]]:
    """
    Replace any submission for the previous winner's UID with the saved winner Miner.
    Returns updated miners and the previous winner UID (if present).
    """
    if prev is None:
        return miners, None
    updated = [m for m in miners if int(m.uid) != int(prev.uid)]
    updated.append(prev)
    bt.logging.info(f"previous winner: included {prev.owner}/{prev.repo}@{prev.branch} as UID={prev.uid}")
    return updated, int(prev.uid)


def upload_snapshots_for_epoch(miners: List[Miner], epoch: int) -> List[Miner]:
    """
    Upload code snapshots for all miners for a single epoch.
    Returns the miners whose snapshot upload succeeded.
    """
    successful_miners: List[Miner] = []
    for miner in miners:
        try:
            upload_miner_snapshot(
                owner=miner.owner,
                repo=miner.repo,
                branch=miner.branch,
                uid=int(miner.uid),
                epoch=epoch,
            )
            bt.logging.info(
                f"snapshot: uploaded {miner.owner}/{miner.repo}@{miner.branch} uid={miner.uid} epoch={epoch}"
            )
            successful_miners.append(miner)
        except Exception as e:
            bt.logging.warning(
                f"snapshot: upload failed for uid={miner.uid} {miner.owner}/{miner.repo}@{miner.branch}: "
                f"{type(e).__name__}: {e}"
            )
    return successful_miners
    
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


def run_job(
    miner: Miner,
    runs_root: Path,
    work_root: Path,
    challenge_params: dict,
    period: int,
    snapshot_epoch: Optional[int] = None,
) -> None:
    repo_dir: Optional[Path] = None
    result_obj: Optional[Dict] = None

    try:
        safe_repo = f"{miner.owner}_{miner.repo}".replace("/", "_")
        dest = work_root / f"{period}_{safe_repo}_{miner.uid}"

        if int(miner.uid) == -1:
            try:
                repo_dir = download_benchmark_snapshot(work_root=work_root, dest_dir=dest)
                bt.logging.info("using benchmark snapshot")
            except Exception as e:
                bt.logging.error(f"benchmark snapshot download failed: {type(e).__name__}: {e}")
                return
        else:
            effective_epoch = snapshot_epoch if snapshot_epoch is not None else period
            try:
                repo_dir = download_and_extract_snapshot(
                    epoch=effective_epoch,
                    uid=int(miner.uid),
                    work_root=work_root,
                    dest_dir=dest,
                )
                if repo_dir is not None:
                    bt.logging.info(f"using archived snapshot for uid={miner.uid} epoch={effective_epoch}")
            except Exception as e:
                bt.logging.error(
                    f"snapshot download failed for uid={miner.uid} epoch={effective_epoch}: {type(e).__name__}: {e}"
                )
                repo_dir = None

            if repo_dir is None:
                bt.logging.info(
                    f"no archived snapshot available for uid={miner.uid} epoch={effective_epoch}; skipping run"
                )
                return

        miner_dir = ensure_miner_exists(repo_dir)

        runner.ensure_docker_image()

        workdir, outdir = runner.prepare_workdir(miner_dir, challenge_params, dest_dir=dest)
        is_current_winner = snapshot_epoch is not None
        start_prefix = "run started for current winner" if is_current_winner else "run started for"
        start_msg = f"{start_prefix} uid={miner.uid} repo={miner.owner}/{miner.repo}@{miner.branch}"
        bt.logging.info(start_msg)
        code, output = runner.run_container(workdir, outdir, period=period, uid=int(miner.uid))
        bt.logging.info(f"run finished uid={miner.uid} exit={code}")
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
        bt.logging.error(f"run failed uid={miner.uid}: {type(e).__name__}: {e}")
    finally:
        if repo_dir is not None:
            try:
                shutil.rmtree(repo_dir, ignore_errors=True)
            except Exception as cleanup_err:
                bt.logging.warning(
                    f"cleanup: failed to remove {repo_dir}: {type(cleanup_err).__name__}: {cleanup_err}"
                )

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
    current_block, subtensor = await call_st(subtensor, network, lambda st: st.get_current_block(), timeout_s=10)

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

    miners = upload_snapshots_for_epoch(miners, period)

    block_hash, subtensor = await call_st(subtensor, network, lambda st: st.determine_block_hash(current_block), timeout_s=10)
    challenge_params = build_challenge_params(str(block_hash))

    try:
        m = COMMITMENT_REGEX.match(BENCHMARK_GITHUB)
        if not m:
            raise ValueError(f"Invalid BENCHMARK_GITHUB: {BENCHMARK_GITHUB}")
        benchmark = Miner(
            uid=-1,
            block_number=current_block,
            raw=BENCHMARK_GITHUB,
            owner=m.group("owner"),
            repo=m.group("repo"),
            branch=m.group("branch"),
            hotkey="benchmark",
        )
        bt.logging.info(f"benchmark: running {BENCHMARK_GITHUB} (uid=-1)")
        run_job(benchmark, runs_root=runs_root, work_root=work_root, challenge_params=challenge_params, period=period)
    except Exception as e:
        bt.logging.error(f"benchmark run failed: {type(e).__name__}: {e}")

    prev_winner_data = get_previous_winner(current_block)
    if prev_winner_data is not None:
        prev_winner, prev_snapshot_epoch = prev_winner_data
        miners, prev_winner_uid = inject_previous_winner(miners, prev_winner)
    else:
        prev_winner_uid = None
        prev_snapshot_epoch = None

    bench_owner = m.group("owner")
    bench_repo = m.group("repo")
    bench_scores_path = Path("/data/miner_runs") / f"{period}_{bench_owner}_{bench_repo}_-1" / "out" / "all_scores_0.json"

    try:
        metagraph, subtensor = await call_st(subtensor, network, lambda st: st.metagraph(netuid), timeout_s=10)
        coldkeys = getattr(metagraph, 'coldkeys', None)
        if coldkeys is not None:
            for miner in miners:
                if isinstance(miner.uid, int) and 0 <= miner.uid < len(coldkeys):
                    miner.coldkey = coldkeys[miner.uid]
    except Exception as e:
        bt.logging.error(f"failed to populate coldkeys: {type(e).__name__}: {e}")
    total_miners = len(miners)
    for idx, miner in enumerate(miners, start=1):
        bt.logging.info(f"running miner {idx}/{total_miners} uid={miner.uid}")
        snapshot_epoch: Optional[int]
        if prev_winner_uid is not None and miner.uid == prev_winner_uid and prev_snapshot_epoch is not None:
            snapshot_epoch = prev_snapshot_epoch
        else:
            snapshot_epoch = None
        run_job(
            miner,
            runs_root=runs_root,
            work_root=work_root,
            challenge_params=challenge_params,
            period=period,
            snapshot_epoch=snapshot_epoch,
        )

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
                    if uid is None or uid == -1:
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
        if prev_winner_uid is not None:
            cfg["prev_winner_uid"] = prev_winner_uid
        cfg["min_improvement_margin"] = cfg_all["min_improvement_margin"]

        winner_uid, winner_score = await scoring_module.process_epoch(cfg, period, uid_to_data, str(bench_scores_path))
        # Persist winner: overwrite each run
        try:
            if isinstance(winner_uid, int):
                win = uid_to_data.get(winner_uid, {})

                raw_commitment = win.get("github_data")
                github_url, github_branch = commitment_to_clone(raw_commitment)

                # Determine snapshot_epoch: keep existing for same uid, else set to current period
                snapshot_epoch: Optional[int] = None
                winner_json_path = Path("/data/results/winner.json")
                if winner_json_path.exists():
                    try:
                        with winner_json_path.open("r", encoding="utf-8") as f:
                            prev = json.load(f)
                        prev_uid = int(prev.get("uid", -1))
                        prev_snapshot_epoch_val = prev.get("snapshot_epoch")
                        if prev_uid == winner_uid and prev_snapshot_epoch_val is not None:
                            snapshot_epoch = int(prev_snapshot_epoch_val)
                    except Exception:
                        snapshot_epoch = None
                if snapshot_epoch is None:
                    snapshot_epoch = period

                winner_obj = {
                    "uid": winner_uid,
                    "hotkey": win.get("hotkey"),
                    "coldkey": win.get("coldkey"),
                    "raw": raw_commitment,
                    "github": github_url,
                    "branch": github_branch,
                    "score": winner_score,
                    "snapshot_epoch": snapshot_epoch,
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


