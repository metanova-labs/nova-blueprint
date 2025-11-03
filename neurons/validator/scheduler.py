import time
import subprocess
from datetime import datetime, timezone
import asyncio
import signal
import json
import os
import threading
from pathlib import Path
import argparse
import bittensor as bt
from config.config_loader import load_config
from neurons.validator.setup import get_config, setup_logging, check_registration
from neurons.validator.weights import apply_weights


def _get_interval_seconds() -> int:
    cfg = load_config()
    val = int(cfg["competition_interval_seconds"]) 
    return val


def _next_aligned_ts(now_ts: float, interval: int) -> float:
    k = int(now_ts // interval)
    return (k + 1) * interval


def _format_duration_hms(total_seconds: int) -> str:
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes or hours:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


def _format_utc(ts: float) -> str:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _read_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _weights_loop(stop_event: threading.Event, cfg) -> None:
    try:
        interval_s = 1800  # 30 minutes

        winner_path = Path("/data/results/winner.json")

        bt.logging.info(f"weights: thread started (interval={interval_s}s)")
        next_ts = time.time() + interval_s
        while not stop_event.is_set():
            now = time.time()
            if now >= next_ts:
                winner = _read_json(winner_path)
                target_uid = 0
                if winner and isinstance(winner.get("uid"), int):
                    target_uid = int(winner["uid"]) 
                bt.logging.info(f"weights: applying target_uid={target_uid}")
                try:
                    apply_weights(target_uid)
                except Exception as e:
                    bt.logging.error(f"weights: failed to set: {type(e).__name__}: {e}")
                # schedule next run
                next_ts = now + interval_s

            # Responsive sleep with stop check
            remaining = max(0.0, next_ts - now)
            time.sleep(min(1.0, remaining))
    except Exception as e:
        bt.logging.error(f"weights: thread crashed: {type(e).__name__}: {e}")

def run_competition(immediate_exit_requested: dict, current_proc: dict, extra_args: list[str] | None = None) -> int:
    start = time.perf_counter()
    cmd = [
        "python",
        "neurons/validator/validator.py",
    ]
    if extra_args:
        cmd.extend(extra_args)
    proc = subprocess.Popen(cmd)
    current_proc["proc"] = proc
    rc: int
    while True:
        try:
            rc = proc.wait(timeout=1)
            break
        except subprocess.TimeoutExpired:
            if immediate_exit_requested.get("flag"):
                bt.logging.info("immediate shutdown requested: terminating validator…")
                try:
                    proc.terminate()
                    rc = proc.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    bt.logging.info("validator did not exit after SIGTERM, killing…")
                    proc.kill()
                    rc = -9
                    break
    current_proc["proc"] = None
    dur = time.perf_counter() - start
    bt.logging.info(f"competition finished rc={rc} in {dur:.1f}s")
    return rc


def setup_and_check_registration():
    cfg = get_config()
    setup_logging(cfg)

    async def _check_reg():
        subtensor = bt.async_subtensor(network=cfg.network)
        await subtensor.initialize()
        wallet = bt.wallet(name=cfg.wallet.name, hotkey=cfg.wallet.hotkey)
        await check_registration(wallet, subtensor, cfg.netuid)

    asyncio.run(_check_reg())
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Validator scheduler")
    parser.add_argument("--test_mode", action="store_true", help="Trigger first run quickly for debugging")
    args, unknown = parser.parse_known_args()
    # one-time setup and registration check
    cfg = setup_and_check_registration()

    # graceful shutdown flags
    termination_requested = {"flag": False}  # graceful: SIGTERM (Watchtower)
    immediate_exit_requested = {"flag": False}  # immediate: SIGINT (Ctrl+C)
    is_running = {"flag": False}
    current_proc = {"proc": None}

    # background weights thread
    stop_event = threading.Event()
    weights_thread = threading.Thread(target=_weights_loop, args=(stop_event, cfg), name="weights", daemon=True)
    weights_thread.start()

    def _handle_term(signum, frame):
        termination_requested["flag"] = True
        stop_event.set()
        if is_running["flag"]:
            bt.logging.info("SIGTERM: will stop after current run finishes")
        else:
            bt.logging.info("SIGTERM: idle, exiting now")

    def _handle_int(signum, frame):
        immediate_exit_requested["flag"] = True
        stop_event.set()
        if is_running["flag"] and current_proc["proc"] is not None:
            bt.logging.info("SIGINT: user interrupt, will abort current run")
        else:
            bt.logging.info("SIGINT: idle, exiting now")

    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT, _handle_int)

    interval = _get_interval_seconds()
    bt.logging.info(f"scheduler started (interval={interval}s)")
    # Test mode: one-time initial delay to trigger first run quickly for debugging
    _first_cycle = bool(args.test_mode)
    _initial_delay_seconds = 10
    while True:
        now_ts = time.time()
        next_ts = _next_aligned_ts(now_ts, interval)
        if _first_cycle:
            next_ts = now_ts + _initial_delay_seconds
            _first_cycle = False
            bt.logging.info(f"test_mode: first run in {_format_duration_hms(int(_initial_delay_seconds))} (overrides alignment)")
        wait_s = max(0, int(next_ts - now_ts))
        next_utc = _format_utc(next_ts)
        bt.logging.info(f"next run at {next_utc} (in {_format_duration_hms(wait_s)})")

        # Sleep in small chunks to react to termination requests quickly
        while True:
            if termination_requested["flag"] and not is_running["flag"]:
                bt.logging.info("graceful shutdown: idle, exiting for update (SIGTERM)")
                return
            if immediate_exit_requested["flag"] and not is_running["flag"]:
                bt.logging.info("immediate shutdown: idle, exiting now (SIGINT)")
                return
            now = time.time()
            if now >= next_ts:
                break
            time.sleep(min(1.0, next_ts - now))

        if termination_requested["flag"] or immediate_exit_requested["flag"]:
            if termination_requested["flag"]:
                bt.logging.info("graceful shutdown: exiting before run start (SIGTERM)")
            else:
                bt.logging.info("immediate shutdown: user interrupt before run start (SIGINT)")
            # Received termination during wait; exit before starting a run
            return

        bt.logging.info("running competition…")
        is_running["flag"] = True
        try:
            run_competition(immediate_exit_requested, current_proc, extra_args=unknown)
        finally:
            is_running["flag"] = False
        if termination_requested["flag"] or immediate_exit_requested["flag"]:
            if termination_requested["flag"]:
                bt.logging.info("Completed run, exiting for update (SIGTERM)")
            else:
                bt.logging.info("immediate shutdown: aborted or completed run on user interrupt (SIGINT)")
            # Exit immediately after finishing current run
            return


if __name__ == "__main__":
    main()

