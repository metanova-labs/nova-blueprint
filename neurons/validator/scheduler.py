import time
import subprocess
import asyncio

import bittensor as bt
from config.config_loader import load_config
from neurons.validator.setup import get_config, setup_logging, check_registration


def _get_interval_seconds() -> int:
    cfg = load_config()
    val = int(cfg["competition_interval_seconds"])  # KeyError/ValueError if missing/invalid
    return val


def _next_aligned_ts(now_ts: float, interval: int) -> float:
    k = int(now_ts // interval)
    return (k + 1) * interval


def run_competition() -> None:
    subprocess.run([
        "python",
        "neurons/validator/validator.py",
    ], check=False)


def setup_and_check_registration():
    cfg = get_config()
    setup_logging(cfg)

    async def _check_reg():
        subtensor = bt.async_subtensor(network=cfg.network)
        await subtensor.initialize()
        wallet = bt.wallet(name=cfg.wallet.name, hotkey=cfg.wallet.hotkey)
        await check_registration(wallet, subtensor, cfg.netuid)

    try:
        asyncio.run(_check_reg())
    except Exception:
        pass
    return cfg


def main() -> None:
    setup_and_check_registration()
    interval = _get_interval_seconds()
    while True:
        now_ts = time.time()
        next_ts = _next_aligned_ts(now_ts, interval)
        time.sleep(max(0, int(next_ts - now_ts)))
        run_competition()


if __name__ == "__main__":
    main()

