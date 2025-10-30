import time
import subprocess
from datetime import datetime, timezone
import asyncio
import signal

import bittensor as bt
from config.config_loader import load_config
from neurons.validator.setup import get_config, setup_logging, check_registration


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


def run_competition(immediate_exit_requested: dict, current_proc: dict) -> int:
    start = time.perf_counter()
    proc = subprocess.Popen([
        "python",
        "neurons/validator/validator.py",
    ])
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
    # one-time setup and registration check
    setup_and_check_registration()

    # graceful shutdown flags
    termination_requested = {"flag": False}  # graceful: SIGTERM (Watchtower)
    immediate_exit_requested = {"flag": False}  # immediate: SIGINT (Ctrl+C)
    is_running = {"flag": False}
    current_proc = {"proc": None}

    def _handle_term(signum, frame):
        termination_requested["flag"] = True
        if is_running["flag"]:
            bt.logging.info("SIGTERM: will stop after current run finishes")
        else:
            bt.logging.info("SIGTERM: idle, exiting now")

    def _handle_int(signum, frame):
        immediate_exit_requested["flag"] = True
        if is_running["flag"] and current_proc["proc"] is not None:
            bt.logging.info("SIGINT: user interrupt, will abort current run")
        else:
            bt.logging.info("SIGINT: idle, exiting now")

    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT, _handle_int)

    interval = _get_interval_seconds()
    bt.logging.info(f"scheduler started (interval={interval}s)")
    while True:
        now_ts = time.time()
        next_ts = _next_aligned_ts(now_ts, interval)
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
            run_competition(immediate_exit_requested, current_proc)
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

