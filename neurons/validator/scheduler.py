import time
import subprocess
from datetime import datetime, timezone
import asyncio

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


def run_competition() -> int:
    start = time.perf_counter()
    cp = subprocess.run([
        "python",
        "neurons/validator/validator.py",
    ], check=False)
    dur = time.perf_counter() - start
    bt.logging.info(f"competition finished rc={cp.returncode} in {dur:.1f}s")
    return cp.returncode


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

    interval = _get_interval_seconds()
    bt.logging.info(f"scheduler started (interval={interval}s)")
    while True:
        now_ts = time.time()
        next_ts = _next_aligned_ts(now_ts, interval)
        wait_s = max(0, int(next_ts - now_ts))
        next_utc = _format_utc(next_ts)
        bt.logging.info(f"next run at {next_utc} (in {_format_duration_hms(wait_s)})")
        time.sleep(wait_s)
        bt.logging.info("running competition…")
        run_competition()


if __name__ == "__main__":
    main()

