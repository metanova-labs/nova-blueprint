import datetime as dt
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
import asyncio

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from sandbox import runner
from utils.challenge_params import build_challenge_params
from neurons.validator import scoring as scoring_module
from neurons.validator.code_archive import (
    download_and_extract_snapshot,
    download_benchmark_snapshot,
)
from config.config_loader import load_config

import bittensor as bt
import requests

"""fetch submission records, run miners in sandbox, persist results."""

BENCHMARK_UID_RANDOM = -1
BENCHMARK_UID_THOMPSON = -2
BLUEPRINT_BOUNTY_URL = "https://emission-transfer.metanova-labs.ai/payouts/blueprint-bounty"


def _benchmark_snapshot_name(uid: int) -> str:
    if int(uid) == BENCHMARK_UID_RANDOM:
        return "brute_force"
    if int(uid) == BENCHMARK_UID_THOMPSON:
        return "thompson_sampling"
    raise ValueError(f"Unknown benchmark uid={uid} (expected {BENCHMARK_UID_RANDOM} or {BENCHMARK_UID_THOMPSON})")



@dataclass
class Miner:
    uid: int
    submitted_at_utc: int
    hotkey: str
    coldkey: Optional[str] = None
    submission_name: Optional[str] = None


def _is_emission_override_active(cfg: dict) -> bool:
    return cfg.get("emission_target_override_uid") is not None


def payout_blueprint_bounty(epoch: int, destination_coldkey: str) -> dict | None:
    api_key = os.environ.get("BLUEPRINT_BOUNTY_API_KEY", "").strip()
    if not api_key:
        bt.logging.warning("blueprint bounty: BLUEPRINT_BOUNTY_API_KEY is not set; skipping payout")
        return None

    resp = requests.post(
        BLUEPRINT_BOUNTY_URL,
        json={
            "epoch": int(epoch),
            "destination_coldkey": destination_coldkey,
        },
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=60,
    )
    if resp.status_code >= 400:
        raise RuntimeError(
            f"blueprint bounty payout failed: status={resp.status_code} body={resp.text[:300]}"
        )

    body = resp.json()
    if not isinstance(body, dict):
        raise RuntimeError("blueprint bounty payout response was not a JSON object")
    return body


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


def fetch_submission_miners(period: int) -> List[Miner]:
    base_url = os.environ.get("SUBMISSION_API_URL", "").strip().rstrip("/")
    api_key = os.environ.get("SUBMISSION_API_KEY", "").strip()
    if not base_url:
        raise RuntimeError("SUBMISSION_API_URL must be set")
    if not api_key:
        raise RuntimeError("SUBMISSION_API_KEY must be set")

    url = f"{base_url}/submissions/by-epoch"
    resp = requests.get(
        url,
        params={"epoch": int(period), "active_only": "true"},
        headers={"X-API-Key": api_key},
        timeout=20,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"submission_api epoch fetch failed: status={resp.status_code} body={resp.text[:300]}")

    body = resp.json()
    items = body.get("items") if isinstance(body, dict) else None
    if not isinstance(items, list):
        raise RuntimeError("submission_api response missing items list")

    miners: List[Miner] = []
    for item in items:
        uid = int(item["uid"])
        hotkey = str(item["hotkey"])
        coldkey = item["coldkey"]
        submitted_at_utc = int(item["submitted_at_utc"])
        submission_name = item["submission_name"]
        miners.append(
            Miner(
                uid=uid,
                submitted_at_utc=submitted_at_utc,
                hotkey=hotkey,
                coldkey=str(coldkey) if coldkey is not None else None,
                submission_name=str(submission_name) if submission_name is not None else None,
            )
        )
    miners.sort(key=lambda m: m.uid)
    return miners


def get_previous_winner() -> Optional[tuple[Miner, int]]:
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
        winner_snapshot = last_win["winner_snapshot"]
        uid_val = int(winner_snapshot["uid"])
        hotkey_val = str(winner_snapshot.get("hotkey") or "")
        coldkey_val = winner_snapshot.get("coldkey")
        submitted_at_utc_val = int(winner_snapshot.get("submitted_at_utc", 0) or 0)
        snapshot_epoch_val = winner_snapshot["snapshot_epoch"]
        snapshot_epoch_int = int(snapshot_epoch_val)
        prev_winner = Miner(
            uid=uid_val,
            submitted_at_utc=submitted_at_utc_val,
            hotkey=hotkey_val,
            coldkey=str(coldkey_val) if coldkey_val else None,
            submission_name=winner_snapshot.get("submission_name"),
        )
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
    bt.logging.info(f"previous winner: included uid={prev.uid}")
    return updated, int(prev.uid)
    
def ensure_miner_exists(repo_dir: Path) -> Path:
    miner_path = repo_dir / "miner.py"
    if not miner_path.is_file():
        raise FileNotFoundError("miner.py not found at repository root")
    return repo_dir


def write_run_artifacts(runs_root: Path, period: int, miner: Miner, result_obj: Optional[Dict]) -> None:
    if result_obj is None:
        bt.logging.info(
            f"run artifacts: uid={miner.uid} produced no result object; "
            f"skipping write to period_{period}_results.jsonl"
        )
        return None
    results_dir = runs_root
    results_dir.mkdir(parents=True, exist_ok=True)

    combined = {
        "uid": miner.uid,
        "submitted_at_utc": miner.submitted_at_utc,
        "coldkey": miner.coldkey,
        "hotkey": miner.hotkey,
        "submission_name": miner.submission_name,
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
        dest = work_root / f"{period}_uid_{miner.uid}"

        if int(miner.uid) < 0:
            try:
                snapshot_name = _benchmark_snapshot_name(int(miner.uid))
                repo_dir = download_benchmark_snapshot(
                    work_root=work_root, dest_dir=dest, name=snapshot_name
                )
                bt.logging.info(
                    f"using benchmark snapshot name={snapshot_name} uid={miner.uid}"
                )
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
        start_msg = f"{start_prefix} uid={miner.uid}"
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
            # Persist benchmark scores before cleanup 
            if int(miner.uid) == -1:
                try:
                    src_scores = outdir / "all_scores_0.json"
                    if src_scores.exists():
                        results_dir = Path("/data/results").resolve()
                        results_dir.mkdir(parents=True, exist_ok=True)
                        dst_scores = results_dir / f"period_{period}_benchmark_all_scores_0.json"
                        shutil.copy2(src_scores, dst_scores)
                        bt.logging.info(f"benchmark: saved scores to {dst_scores}")
                except Exception as copy_err:
                    bt.logging.warning(
                        f"benchmark: failed to persist scores: {type(copy_err).__name__}: {copy_err}"
                    )
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


async def main() -> int:
    runs_root = Path("/data/results").resolve()
    work_root = Path("/data/miner_runs").resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    load_dotenv(PROJECT_ROOT / ".env")

    network = os.environ.get("SUBTENSOR_NETWORK")

    subtensor = bt.async_subtensor(network=network)
    await subtensor.initialize()
    current_block, subtensor = await call_st(subtensor, network, lambda st: st.get_current_block(), timeout_s=10)

    cfg_all = load_config()
    interval_seconds = int(cfg_all["competition_interval_seconds"]) 
    now_ts = int(time.time())
    period_index = now_ts // interval_seconds
    period_start_ts = period_index * interval_seconds
    period = period_index

    bt.logging.info(
        f"period_index={period} start_utc={dt.datetime.fromtimestamp(period_start_ts, dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}Z "
        f"interval_seconds={interval_seconds}"
    )

    miners = fetch_submission_miners(period=period)
    bt.logging.info(f"current_block={current_block} submission_api_miners={len(miners)} period={period}")

    block_hash, subtensor = await call_st(subtensor, network, lambda st: st.determine_block_hash(current_block), timeout_s=10)
    challenge_params = build_challenge_params(str(block_hash))
    # Persist the exact input used for this period
    try:
        results_dir = Path("/data/results").resolve()
        results_dir.mkdir(parents=True, exist_ok=True)
        input_path = results_dir / f"period_{period}_input.json"
        with input_path.open("w", encoding="utf-8") as f:
            json.dump(challenge_params, f, separators=(",", ":"))
        bt.logging.info(f"saved period input to {input_path}")
    except Exception as e:
        bt.logging.warning(f"failed to persist period input: {type(e).__name__}: {e}")

    try:
        benchmark = Miner(
            uid=BENCHMARK_UID_RANDOM,
            submitted_at_utc=now_ts,
            hotkey="benchmark",
        )
        bt.logging.info(f"benchmark: running brute_force snapshot (uid={BENCHMARK_UID_RANDOM})")
        run_job(benchmark, runs_root=runs_root, work_root=work_root, challenge_params=challenge_params, period=period)
    except Exception as e:
        bt.logging.error(f"benchmark run failed: {type(e).__name__}: {e}")

    try:
        ts_benchmark = Miner(
            uid=BENCHMARK_UID_THOMPSON,
            submitted_at_utc=now_ts,
            hotkey="benchmark",
        )
        bt.logging.info(f"thompson_sampling: running snapshot (uid={BENCHMARK_UID_THOMPSON})")
        run_job(ts_benchmark, runs_root=runs_root, work_root=work_root, challenge_params=challenge_params, period=period)
    except Exception as e:
        bt.logging.error(f"thompson_sampling run failed: {type(e).__name__}: {e}")

    prev_winner_data = get_previous_winner()
    if prev_winner_data is not None:
        prev_winner, prev_snapshot_epoch = prev_winner_data
        miners, prev_winner_uid = inject_previous_winner(miners, prev_winner)
    else:
        prev_winner_uid = None
        prev_snapshot_epoch = None

    bench_scores_path = Path("/data/results") / f"period_{period}_benchmark_all_scores_0.json"

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
                    if uid is None or uid < 0:
                        continue
                    molecules = rec.get("result", {}).get("molecules", [])
                    uid_to_data[uid] = {
                        "molecules": molecules,
                        "github_data": None,
                        "hotkey": rec.get("hotkey"),
                        "coldkey": rec.get("coldkey"),
                        "submission_name": rec.get("submission_name"),
                        "submitted_at_utc": int(rec.get("submitted_at_utc", 0) or 0),
                    }
        cfg = dict(challenge_params.get("config", {}))
        cfg.update(challenge_params.get("challenge", {}))
        if prev_winner_uid is not None:
            cfg["prev_winner_uid"] = prev_winner_uid
        cfg["min_improvement_margin"] = cfg_all["min_improvement_margin"]
        cfg["min_improvement_decay_rate"] = cfg_all["min_improvement_decay_rate"]
        cfg["winner_snapshot_epoch"] = prev_snapshot_epoch
        cfg["time_budget_sec"] = cfg_all.get("time_budget_sec", 0)

        winner_uid, winner_score = await scoring_module.process_epoch(
            cfg,
            period,
            uid_to_data,
            str(bench_scores_path),
        )
        # Persist winner only when winner UID changes.
        try:
            if isinstance(winner_uid, int):
                win = uid_to_data.get(winner_uid, {})
                winner_json_path = Path("/data/results/winner.json")
                prev_winner_uid: Optional[int] = None
                if winner_json_path.exists():
                    try:
                        with winner_json_path.open("r", encoding="utf-8") as f:
                            prev = json.load(f)
                        prev_winner_uid = int(prev["winner_snapshot"]["uid"])
                    except Exception:
                        prev_winner_uid = None

                if prev_winner_uid == winner_uid:
                    bt.logging.info(
                        f"winner unchanged uid={winner_uid}; keeping existing winner.json metadata unchanged"
                    )
                else:
                    snapshot_epoch = period
                    winner_code_link = f"{int(snapshot_epoch)}/{int(winner_uid)}"
                    winner_obj = {
                        "winner_snapshot": {
                            "uid": winner_uid,
                            "hotkey": win.get("hotkey"),
                            "coldkey": win.get("coldkey"),
                            "submission_name": win.get("submission_name"),
                            "submitted_at_utc": int(win.get("submitted_at_utc", 0) or 0),
                            "code_link": winner_code_link,
                            "score": winner_score,
                            "snapshot_epoch": snapshot_epoch,
                            "updated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                        },
                        "emission_target_uid": winner_uid,
                    }
                    out_dir = Path("/data/results").resolve()
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / "winner.json"
                    tmp_path = out_dir / "winner.json.tmp"
                    with tmp_path.open("w", encoding="utf-8") as f:
                        json.dump(winner_obj, f, separators=(",", ":"))
                    os.replace(tmp_path, out_path)
                    bt.logging.info(f"winner persisted uid={winner_uid} at {out_path}")

                    if _is_emission_override_active(cfg_all):
                        if prev_winner_uid is None:
                            bt.logging.info(
                                "blueprint bounty: no previous winner recorded; skipping payout"
                            )
                        elif not win.get("coldkey"):
                            bt.logging.error(
                                f"blueprint bounty: winner uid={winner_uid} is missing a coldkey; skipping payout"
                            )
                        else:
                            try:
                                payout = payout_blueprint_bounty(
                                    epoch=period,
                                    destination_coldkey=str(win["coldkey"]),
                                )
                                if payout is None:
                                    bt.logging.warning(
                                        "blueprint bounty: payout request was skipped due to missing authentication"
                                    )
                                else:
                                    status = str(payout.get("status", "unknown"))
                                    amount_alpha = payout.get("amount_alpha")
                                    destination = payout.get("destination_coldkey", win["coldkey"])
                                    extrinsic_id = payout.get("extrinsic_id")
                                    detail = payout.get("detail")
                                    if status == "success":
                                        bt.logging.info(
                                            f"rewarded {amount_alpha} alpha bounty to {destination}, extrinsic id: {extrinsic_id}"
                                        )
                                    else:
                                        bt.logging.warning(
                                            f"blueprint bounty payout status={status} destination={destination} "
                                            f"amount_alpha={amount_alpha} extrinsic_id={extrinsic_id} detail={detail}"
                                        )
                            except Exception as payout_err:
                                bt.logging.error(
                                    f"blueprint bounty payout failed for uid={winner_uid}: "
                                    f"{type(payout_err).__name__}: {payout_err}"
                                )

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


