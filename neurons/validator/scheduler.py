import time
import subprocess

from config.config_loader import load_config


def _get_interval_seconds() -> int:
    cfg = load_config()
    val = int(cfg["competition_interval_seconds"]) 
    return val


def _next_aligned_ts(now_ts: float, interval: int) -> float:
    k = int(now_ts // interval)
    return (k + 1) * interval


def run_competition() -> None:
    subprocess.run([
        "python",
        "neurons/validator/validator.py",
    ], check=False)


def main() -> None:
    interval = _get_interval_seconds()
    while True:
        now_ts = time.time()
        next_ts = _next_aligned_ts(now_ts, interval)
        time.sleep(max(0, int(next_ts - now_ts)))
        run_competition()


if __name__ == "__main__":
    main()

