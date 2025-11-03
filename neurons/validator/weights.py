import os
import time
from dotenv import load_dotenv
import bittensor as bt

NETUID = 68
MECH_ID = 1
BURN_RATE = 0.5


def apply_weights(target_uid: int) -> None:
    load_dotenv()

    wallet_name = os.getenv("BT_WALLET_COLD")
    wallet_hotkey = os.getenv("BT_WALLET_HOT")
    network = os.getenv("SUBTENSOR_NETWORK")

    missing = []
    if not wallet_name:
        missing.append("BT_WALLET_COLD")
    if not wallet_hotkey:
        missing.append("BT_WALLET_HOT")
    if not network:
        missing.append("SUBTENSOR_NETWORK")
    if missing:
        bt.logging.error(f"weights: missing required env: {', '.join(missing)}")
        return

    wallet = bt.wallet(name=wallet_name, hotkey=wallet_hotkey)
    subtensor = bt.subtensor(network=network)
    metagraph = subtensor.metagraph(NETUID)

    # Registration check
    hotkey_ss58 = wallet.hotkey.ss58_address
    if hotkey_ss58 not in metagraph.hotkeys:
        bt.logging.error(f"weights: hotkey {hotkey_ss58} not registered on netuid {NETUID}")
        return

    n = len(metagraph.uids)
    if n == 0:
        bt.logging.error("weights: empty metagraph")
        return
    if not (0 <= target_uid < n):
        bt.logging.error(f"weights: target_uid {target_uid} out of range [0, {n-1}]")
        return

    weights = [0.0] * n
    remainder = max(0.0, 1.0 - BURN_RATE)
    if target_uid == 0:
        weights[0] = BURN_RATE + remainder
    else:
        weights[0] = BURN_RATE
        weights[target_uid] = remainder

    max_retries = 5
    delay_s = 12
    for attempt in range(1, max_retries + 1):
        try:
            bt.logging.info(
                f"weights: attempt {attempt} apply uid={target_uid} burn={BURN_RATE}"
            )
            success, message = subtensor.set_weights(
                wallet=wallet,
                netuid=NETUID,
                mechid=MECH_ID,
                uids=metagraph.uids,
                weights=weights,
                wait_for_inclusion=True,
            )
            bt.logging.info(f"weights: set_weights success={success} message={message}")
            if success is True:
                return
            if attempt < max_retries:
                bt.logging.info(f"weights: retrying in {delay_s}s…")
                time.sleep(delay_s)
        except Exception as e:
            bt.logging.error(f"weights: error setting weights: {type(e).__name__}: {e}")
            if attempt < max_retries:
                bt.logging.info(f"weights: retrying in {delay_s}s…")
                time.sleep(delay_s)
    bt.logging.error("weights: failed after maximum retries")
