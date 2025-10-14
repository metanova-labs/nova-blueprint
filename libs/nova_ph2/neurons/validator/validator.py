import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import asyncio
import sys
import bittensor as bt
import torch
import gc
import traceback

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(BASE_DIR)

from config.config_loader import load_config
from PSICHIC.wrapper import PsichicWrapper

from utils import get_challenge_params_from_blockhash
from neurons.validator.setup import get_config, setup_logging, check_registration
from neurons.validator.weights import set_weights
from neurons.validator.validity import validate_molecules_and_calculate_entropy
from neurons.validator.scoring import score_all_proteins_psichic, read_data_from_json
import neurons.validator.scoring as scoring_module
from neurons.validator.ranking import calculate_final_scores, determine_winner
from neurons.validator.monitoring import monitor_validator
from neurons.validator.save_data import submit_epoch_results

psichic = None

async def process_epoch(config, current_block, metagraph, subtensor, wallet):
    """
    Process a single epoch end-to-end.
    """
    global psichic
    try:
        start_block = current_block - config.epoch_length
        start_block_hash = await subtensor.determine_block_hash(start_block)
        final_block_hash = await subtensor.determine_block_hash(current_block)
        current_epoch = (current_block // config.epoch_length) - 1

        # Get challenge parameters for this epoch
        challenge_params = get_challenge_params_from_blockhash(
            block_hash=start_block_hash,
            weekly_target=config.weekly_target,
            num_antitargets=config.num_antitargets,
            include_reaction=config.random_valid_reaction,
        )
        target_proteins = challenge_params["targets"]
        antitarget_proteins = challenge_params["antitargets"]
        allowed_reaction = challenge_params.get("allowed_reaction")

        if allowed_reaction:
            bt.logging.info(f"Allowed reaction this epoch: {allowed_reaction}")

        bt.logging.info(f"Scoring using target proteins: {target_proteins}, antitarget proteins: {antitarget_proteins}")

        # Gather results
        uid_to_data = read_data_from_json(os.path.join(PARENT_DIR, "output.json"))

        if not uid_to_data:
            bt.logging.info("No valid submissions found this epoch.")
            return None

        # Initialize scoring structure
        score_dict = {
            uid: {
                "target_scores": [[] for _ in range(len(target_proteins))],
                "antitarget_scores": [[] for _ in range(len(antitarget_proteins))],
                "entropy": None,
                "block_submitted": None,
                "push_time": uid_to_data[uid].get("push_time", '')
            }
            for uid in uid_to_data
        }
        bt.logging.debug(f"uid_to_data: {uid_to_data}")

        # Validate molecules and calculate entropy
        valid_molecules_by_uid = validate_molecules_and_calculate_entropy(
            uid_to_data=uid_to_data,
            score_dict=score_dict,
            config=config,
            allowed_reaction=allowed_reaction
        )

        # Initialize and use PSICHIC model
        if psichic is None:
            psichic = PsichicWrapper()
            scoring_module.psichic = psichic
            bt.logging.info("PSICHIC model initialized successfully")
        
        # Score all target proteins then all antitarget proteins one protein at a time
        score_all_proteins_psichic(
            target_proteins=target_proteins,
            antitarget_proteins=antitarget_proteins,
            score_dict=score_dict,
            valid_molecules_by_uid=valid_molecules_by_uid,
            uid_to_data=uid_to_data,
            batch_size=32
        )

        # Calculate final scores
        score_dict = calculate_final_scores(
            score_dict, valid_molecules_by_uid, config, current_epoch
        )

        # Determine winner
        winner = determine_winner(score_dict)

        # Yield so ws heartbeats can run before the next RPC
        await asyncio.sleep(0)

        # Submit results to dashboard API if configured
        try:
            submit_url = os.environ.get('SUBMIT_RESULTS_URL')
            if submit_url:
                await submit_epoch_results(
                    submit_url=submit_url,
                    config=config,
                    metagraph=metagraph,
                    current_block=current_block,
                    start_block=start_block,
                    current_epoch=current_epoch,
                    target_proteins=target_proteins,
                    antitarget_proteins=antitarget_proteins,
                    uid_to_data=uid_to_data,
                    valid_molecules_by_uid=valid_molecules_by_uid,
                    score_dict=score_dict
                )
        except Exception as e:
            bt.logging.error(f"Failed to submit results to dashboard API: {e}")

        # Monitor validators
        if not bool(getattr(config, 'test_mode', False)):
            try:
                set_weights_call_block = await subtensor.get_current_block()
            except asyncio.CancelledError:
                bt.logging.info("Resetting subtensor connection.")
                subtensor = bt.async_subtensor(network=config.network)
                await subtensor.initialize()
                await asyncio.sleep(1)
                set_weights_call_block = await subtensor.get_current_block()
            monitor_validator(
                score_dict=score_dict,
                metagraph=metagraph,
                current_epoch=current_epoch,
                current_block=set_weights_call_block,
                validator_hotkey=wallet.hotkey.ss58_address,
                winning_uid=winner
            )

        return winner

    except Exception as e:
        bt.logging.error(f"Error processing epoch: {e}")
        return None

async def main(config):
    """
    Main validator loop
    """
    test_mode = bool(getattr(config, 'test_mode', False))
    
    # Initialize subtensor client
    subtensor = bt.async_subtensor(network=config.network)
    await subtensor.initialize()
    
    # Wallet + registration check (skipped in test mode)
    wallet = None
    if test_mode:
        bt.logging.info("TEST MODE: running without setting weights")
    else:
        try:
            wallet = bt.wallet(config=config)
            await check_registration(wallet, subtensor, config.netuid)
        except Exception as e:
            bt.logging.error(f"Wallet/registration check failed: {e}")
            sys.exit(1)


    # Main validator loop
    last_logged_blocks_remaining = None
    while True:
        try:
            metagraph = await subtensor.metagraph(config.netuid)
            current_block = await subtensor.get_current_block()

            if current_block % config.epoch_length != 0:
                # Epoch end - process and set weights
                winner = await process_epoch(config, current_block, metagraph, subtensor, wallet)
                if not test_mode:
                    await set_weights(winner, config)
                
            else:
                # Waiting for epoch
                blocks_remaining = config.epoch_length - (current_block % config.epoch_length)
                print(f"Current epoch: {(current_block // config.epoch_length) - 1}")
                if (blocks_remaining % 5 == 0) and (blocks_remaining != last_logged_blocks_remaining):
                    bt.logging.info(f"Waiting for epoch to end... {blocks_remaining} blocks remaining.")
                    last_logged_blocks_remaining = blocks_remaining
                await asyncio.sleep(1)
                
 
        except asyncio.CancelledError:
            bt.logging.info("Resetting subtensor connection.")
            subtensor = bt.async_subtensor(network=config.network)
            await subtensor.initialize()
            await asyncio.sleep(1)
            continue
        except Exception as e:
            bt.logging.error(f"Error in main loop: {e}")
            await asyncio.sleep(3)

if __name__ == "__main__":
    config = get_config()
    setup_logging(config)
    asyncio.run(main(config))
