import argparse
import os
import sys
from dotenv import load_dotenv
import bittensor as bt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(BASE_DIR)

from config.config_loader import load_config

def get_config():
    """
    Parse command-line arguments to set up the configuration for the wallet
    and subtensor client.
    """
    load_dotenv()
    parser = argparse.ArgumentParser('Nova')
    bt.Wallet.add_args(parser)
    bt.Subtensor.add_args(parser)
    
    parser.add_argument('--test_mode', action='store_true', 
                       help='Run test validator without setting weights')
    parser.add_argument(
        '--contest_state_only',
        action='store_true',
        help='Ignored here; consumed by scheduler on first cycle only',
    )

    config = bt.Config(parser)

    parsed_args, _ = parser.parse_known_args()
    for _field in ("name", "hotkey", "path"):
        _val = getattr(parsed_args, f"wallet.{_field}", None)
        if _val is not None:
            setattr(config.wallet, _field, _val)
    config.netuid = 68
    config.network = os.environ.get("SUBTENSOR_NETWORK")
    with bt.Subtensor(network=config.network) as subtensor:
        config.epoch_length = subtensor.tempo(config.netuid) + 1

    # Load configuration options
    config.update(load_config())

    return config

def setup_logging(config):
    """
    bittensor 10.x defaults its console logger to WARNING+ (9.x emitted INFO);
    turn INFO back on so validator output reaches stdout (and Promtail).
    """
    bt.logging.enable_info()

    bt.logging.info(f"Running validator for subnet: {config.netuid} on network: {config.subtensor.network} with config:")
    bt.logging.info(config)

async def check_registration(wallet, subtensor, netuid):
    """
    Confirm that the wallet hotkey is in the metagraph for the specified netuid.
    Logs an error and exits if it's not registered. Warns if stake is less than 1000.
    """
    metagraph = await subtensor.metagraph(netuid=netuid)
    my_hotkey_ss58 = wallet.hotkey.ss58_address

    if my_hotkey_ss58 not in metagraph.hotkeys:
        bt.logging.error(f"Hotkey {my_hotkey_ss58} is not registered on netuid {netuid}.")
        bt.logging.error("Are you sure you've registered and staked?")
        sys.exit(1) 
    
    uid = metagraph.hotkeys.index(my_hotkey_ss58)
    myStake = metagraph.S[uid]
    bt.logging.info(f"Hotkey {my_hotkey_ss58} found with UID={uid} and stake={myStake}")

    if (myStake < 1000):
        bt.logging.warning(f"Hotkey has less than 1000 stake, unable to validate")

