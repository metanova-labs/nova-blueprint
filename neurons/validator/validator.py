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
from neurons.validator.contest import (
    entry_id,
    load_contest_state,
    save_contest_state,
    WINNER_JSON_PATH,
)
from config.config_loader import load_config

import bittensor as bt
import requests

"""fetch submission records, run miners in sandbox, persist results."""

BENCHMARK_UID_RANDOM = -1
BENCHMARK_UID_THOMPSON = -2
BLUEPRINT_BOUNTY_URL = "https://emission-transfer-api.metanova-labs.ai/payouts/blueprint-bounty"


def _benchmark_snapshot_name(uid: int) -> str:
    if int(uid) == BENCHMARK_UID_RANDOM:
        return "brute_force"
    if int(uid) == BENCHMARK_UID_THOMPSON:
        return "thompson_sampling"
    raise ValueError(f"Unknown benchmark uid={uid} (expected {BENCHMARK_UID_RANDOM} or {BENCHMARK_UID_THOMPSON})")



@dataclass
class Miner:
    uid: Optional[int]
    submitted_at_utc: int
    hotkey: str
    coldkey: Optional[str] = None
    submission_name: Optional[str] = None
    entry_id: Optional[str] = None
    kind: str = "entrant"  # entrant | champion | challenger | benchmark
    snapshot_epoch: Optional[int] = None


def _safe_path_token(value: str) -> str:
    """Filesystem-safe token for workdir naming."""
    return "".join(c if (c.isalnum() or c in "._-@") else "_" for c in str(value))


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
        timeout=(5, 70),
    )
    if resp.status_code >= 400:
        raise RuntimeError(
            f"blueprint bounty payout failed: status={resp.status_code} body={resp.text[:300]}"
        )

    body = resp.json()
    if not isinstance(body, dict):
        raise RuntimeError("blueprint bounty payout response was not a JSON object")
    return body


def _fire_payout(epoch: int, coldkey: str, hotkey: str) -> None:
    try:
        payout = payout_blueprint_bounty(epoch=epoch, destination_coldkey=coldkey)
        if payout is None:
            bt.logging.warning("blueprint bounty: payout request was skipped due to missing authentication")
            return
        status = str(payout.get("status", "unknown"))
        amount_alpha = payout.get("amount_alpha")
        destination = payout.get("destination_coldkey", coldkey)
        extrinsic_id = payout.get("extrinsic_id")
        detail = payout.get("detail")
        if status == "success":
            bt.logging.info(f"rewarded {amount_alpha} alpha bounty to {destination}, extrinsic id: {extrinsic_id}")
        else:
            bt.logging.warning(
                f"blueprint bounty payout status={status} destination={destination} "
                f"amount_alpha={amount_alpha} extrinsic_id={extrinsic_id} detail={detail}"
            )
    except Exception as payout_err:
        bt.logging.error(
            f"blueprint bounty payout failed for hotkey={hotkey}: {type(payout_err).__name__}: {payout_err}"
        )


def _persist_champion_and_payout(state: dict, champion_score, epoch: int, cfg_all: dict) -> None:
    """Mirror the champion to winner.json and pay the bounty when the champion hotkey changes."""
    champion = state.get("champion")
    if not champion:
        return
    try:
        prev_hotkey = None
        if WINNER_JSON_PATH.exists():
            try:
                with WINNER_JSON_PATH.open("r", encoding="utf-8") as f:
                    prev_hotkey = json.load(f)["winner_snapshot"].get("hotkey")
            except Exception:
                prev_hotkey = None

        if prev_hotkey == champion["hotkey"]:
            bt.logging.info(f"champion unchanged hotkey={champion['hotkey']}; winner.json kept")
            return

        snapshot_epoch = int(champion["snapshot_epoch"])
        winner_obj = {
            "winner_snapshot": {
                "uid": champion.get("uid"),
                "hotkey": champion["hotkey"],
                "coldkey": champion.get("coldkey"),
                "submission_name": champion.get("submission_name"),
                "code_link": f"{snapshot_epoch}/{champion['hotkey']}",
                "score": champion_score,
                "snapshot_epoch": snapshot_epoch,
                "updated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            },
            "emission_target_uid": champion.get("uid"),
        }
        out_dir = WINNER_JSON_PATH.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = out_dir / "winner.json.tmp"
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(winner_obj, f, separators=(",", ":"))
        os.replace(tmp_path, WINNER_JSON_PATH)
        bt.logging.info(f"champion persisted hotkey={champion['hotkey']} at {WINNER_JSON_PATH}")

        if _is_emission_override_active(cfg_all):
            if prev_hotkey is None:
                bt.logging.info("blueprint bounty: no previous champion recorded; skipping payout")
            elif not champion.get("coldkey"):
                bt.logging.error(
                    f"blueprint bounty: champion hotkey={champion['hotkey']} missing coldkey; skipping payout"
                )
            else:
                _fire_payout(epoch, str(champion["coldkey"]), str(champion["hotkey"]))
    except Exception as e:
        bt.logging.error(f"failed to persist champion: {type(e).__name__}: {e}")


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
        st = bt.AsyncSubtensor(network=network)
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
        # Credit-mode entrants pay per epoch instead of holding a registration, so
        # the api returns uid null for them. This is the only place uid is coerced;
        # every later hop reads it back from json as int-or-null and just carries it.
        raw_uid = item.get("uid")
        uid = int(raw_uid) if raw_uid is not None else None
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
                entry_id=entry_id(hotkey, period),
                kind="entrant",
                snapshot_epoch=int(period),
            )
        )
    # No sort: the api returns entrants oldest-submission-first, which is both
    # deterministic and meaningful. Champion/challenger ordering is set by
    # build_run_list, not here.
    return miners


def _frozen_miner(member: dict, kind: str) -> Miner:
    """Build a frozen Miner (champion/challenger) from contest state."""
    snapshot_epoch = int(member["snapshot_epoch"])
    hotkey = str(member["hotkey"])
    return Miner(
        uid=member.get("uid"),
        submitted_at_utc=0,
        hotkey=hotkey,
        coldkey=member.get("coldkey"),
        submission_name=member.get("submission_name"),
        entry_id=entry_id(hotkey, snapshot_epoch),
        kind=kind,
        snapshot_epoch=snapshot_epoch,
    )


def build_run_list(entrants: List[Miner], state: dict) -> List[Miner]:
    """
    Run list = champion + challengers (frozen, from state) + current entrants.

    Keyed by entry_id (hotkey@snapshot_epoch), so a hotkey already frozen in the
    pool can still run a fresh current-epoch submission as an entrant
    """
    champion = state.get("champion")
    challengers = state.get("challengers", [])

    run_list: List[Miner] = []
    seen_ids: set = set()

    def _add(miner: Miner) -> None:
        if miner.entry_id in seen_ids:
            return
        seen_ids.add(miner.entry_id)
        run_list.append(miner)

    if champion:
        _add(_frozen_miner(champion, "champion"))
    for challenger in challengers:
        _add(_frozen_miner(challenger, "challenger"))
    for miner in entrants:
        _add(miner)

    bt.logging.info(
        f"run list: champion={'yes' if champion else 'no'} "
        f"challengers={len(challengers)} total={len(run_list)}"
    )
    return run_list


def ensure_miner_exists(repo_dir: Path) -> Path:
    miner_path = repo_dir / "miner.py"
    if not miner_path.is_file():
        raise FileNotFoundError("miner.py not found at repository root")
    return repo_dir


def write_run_artifacts(runs_root: Path, period: int, miner: Miner, result_obj: Optional[Dict]) -> None:
    if result_obj is None:
        bt.logging.info(
            f"run artifacts: entry={miner.entry_id or miner.uid} produced no result object; "
            f"skipping write to period_{period}_results.jsonl"
        )
        return None
    results_dir = runs_root
    results_dir.mkdir(parents=True, exist_ok=True)

    combined = {
        "entry_id": miner.entry_id,
        "kind": miner.kind,
        "uid": miner.uid,
        "snapshot_epoch": miner.snapshot_epoch,
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
) -> None:
    repo_dir: Optional[Path] = None
    result_obj: Optional[Dict] = None

    try:
        # One token names the workdir, the container and the log stream, so a single
        # run is greppable across disk, docker and loki.
        entry_token = miner.entry_id or f"benchmark_{miner.uid}"
        dest = work_root / f"{period}_{_safe_path_token(entry_token)}"

        if miner.kind == "benchmark":
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
            effective_epoch = miner.snapshot_epoch if miner.snapshot_epoch is not None else period
            try:
                repo_dir = download_and_extract_snapshot(
                    epoch=effective_epoch,
                    hotkey=miner.hotkey,
                    work_root=work_root,
                    dest_dir=dest,
                )
                if repo_dir is not None:
                    bt.logging.info(
                        f"using archived snapshot for entry={miner.entry_id} epoch={effective_epoch}"
                    )
            except Exception as e:
                bt.logging.error(
                    f"snapshot download failed for entry={miner.entry_id} epoch={effective_epoch}: "
                    f"{type(e).__name__}: {e}"
                )
                repo_dir = None

            if repo_dir is None:
                bt.logging.info(
                    f"no archived snapshot available for entry={miner.entry_id} epoch={effective_epoch}; skipping run"
                )
                return

        miner_dir = ensure_miner_exists(repo_dir)

        runner.ensure_docker_image()

        workdir, outdir = runner.prepare_workdir(miner_dir, challenge_params, dest_dir=dest)
        bt.logging.info(f"run started for {miner.kind} entry={entry_token}")
        code, output = runner.run_container(workdir, outdir, period=period, entry_id=entry_token)
        bt.logging.info(f"run finished entry={entry_token} exit={code}")
        try:
            with open(outdir / "result.json", "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict) and "result" in raw and isinstance(raw["result"], dict):
                result_obj = raw["result"]
            elif isinstance(raw, dict):
                result_obj = raw
            # Persist benchmark scores before cleanup
            if miner.kind == "benchmark" and int(miner.uid) == -1:
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
        bt.logging.error(f"run failed entry={entry_token}: {type(e).__name__}: {e}")
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
    bt.logging.enable_info()
    runs_root = Path("/data/results").resolve()
    work_root = Path("/data/miner_runs").resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    load_dotenv(PROJECT_ROOT / ".env")

    network = os.environ.get("SUBTENSOR_NETWORK")

    subtensor = bt.AsyncSubtensor(network=network)
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
            kind="benchmark",
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
            kind="benchmark",
        )
        bt.logging.info(f"thompson_sampling: running snapshot (uid={BENCHMARK_UID_THOMPSON})")
        run_job(ts_benchmark, runs_root=runs_root, work_root=work_root, challenge_params=challenge_params, period=period)
    except Exception as e:
        bt.logging.error(f"thompson_sampling run failed: {type(e).__name__}: {e}")

    state = load_contest_state()
    run_list = build_run_list(miners, state)

    bench_scores_path = Path("/data/results") / f"period_{period}_benchmark_all_scores_0.json"

    total = len(run_list)
    for idx, miner in enumerate(run_list, start=1):
        bt.logging.info(f"running {idx}/{total} {miner.kind} entry={miner.entry_id}")
        run_job(
            miner,
            runs_root=runs_root,
            work_root=work_root,
            challenge_params=challenge_params,
            period=period,
        )

    try:
        jsonl_path = (Path("/data/results") / f"period_{period}_results.jsonl")
        entries: Dict[str, Dict] = {}
        if jsonl_path.exists():
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    # Exclude benchmarks by kind, not by absence of a uid: a
                    # credit-mode entrant has no uid and is a real competitor.
                    if rec.get("kind") == "benchmark":
                        continue
                    eid = rec.get("entry_id")
                    if not eid:
                        continue
                    snapshot_epoch_val = rec.get("snapshot_epoch")
                    entries[eid] = {
                        "molecules": rec.get("result", {}).get("molecules", []),
                        "github_data": None,
                        "hotkey": rec.get("hotkey"),
                        "uid": rec.get("uid"),
                        "coldkey": rec.get("coldkey"),
                        "submission_name": rec.get("submission_name"),
                        "snapshot_epoch": int(snapshot_epoch_val) if snapshot_epoch_val is not None else period,
                        "kind": rec.get("kind", "entrant"),
                    }

        cfg = dict(challenge_params.get("config", {}))
        cfg.update(challenge_params.get("challenge", {}))
        cfg["wins_required"] = int(cfg_all["wins_required"])
        cfg["improvement_margin"] = float(cfg_all["improvement_margin"])
        cfg["time_budget_sec"] = cfg_all.get("time_budget_sec", 0)

        new_state, champion_entry_id, champion_score = await scoring_module.process_epoch(
            cfg,
            period,
            entries,
            state,
            str(bench_scores_path),
        )

        try:
            save_contest_state(new_state)
            bt.logging.info("contest state persisted")
        except Exception as e:
            bt.logging.error(f"failed to persist contest state: {type(e).__name__}: {e}")

        _persist_champion_and_payout(new_state, champion_score, period, cfg_all)

    except Exception as e:
        bt.logging.error(f"scoring step failed: {e}")

    return 0


if __name__ == "__main__":
    try:
        import asyncio
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)


