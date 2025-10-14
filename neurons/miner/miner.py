import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import asyncio
import sys
import json
import traceback

import bittensor as bt
import pandas as pd
from rdkit import Chem

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

from nova_ph2.config.config_loader import load_config

from nova_ph2.utils import get_challenge_params_from_blockhash
from nova_ph2.neurons.validator.setup import get_config, setup_logging
from nova_ph2.neurons.validator.scoring import score_molecules_json
import nova_ph2.neurons.validator.scoring as scoring_module
from random_sampler import run_sampler
from nova_ph2.combinatorial_db.reactions import get_smiles_from_reaction

async def process_epoch(config, current_block, metagraph, subtensor):
    """
    Process a single epoch end-to-end.
    """
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

        bt.logging.info(f"Using target proteins: {target_proteins}, antitarget proteins: {antitarget_proteins}")

        output_dir = os.environ.get("OUTPUT_DIR", BASE_DIR)
        db_path = os.path.join(PARENT_DIR, "nova_ph2", "combinatorial_db", "molecules.sqlite")
        iterative_sampling_loop(
            n_samples=config.num_molecules * 5, # 5x the number of molecules to select top x 
            top_x=config.num_molecules,
            target_proteins=target_proteins,
            antitarget_proteins=antitarget_proteins,
            db_path=db_path,
            sampler_file_path=os.path.join(output_dir, "sampler_file.json"), 
            output_path=os.path.join(output_dir, "output.json"), 
            subnet_config=config,
            save_all_scores=True,
            current_epoch=current_epoch,
            )

    except Exception as e:
        bt.logging.error(f"Error processing epoch: {e}")
        bt.logging.error(traceback.format_exc())
        return None


def iterative_sampling_loop(
    n_samples: int,
    top_x: int,
    target_proteins: list[str],
    antitarget_proteins: list[str],
    db_path: str,
    sampler_file_path: str,
    output_path: str,
    subnet_config: dict,
    save_all_scores: bool = True,
    current_epoch: int = 0,
) -> None:
    """
    Infinite loop, runs until orchestrator kills it:
      1) Sample n molecules
      2) Score them
      3) Merge with previous top x, deduplicate, sort, select top x
      4) Write top x to file (overwrite) each iteration
    """

    top_pool = pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])

    iteration = 0
    while True:
        iteration += 1
        bt.logging.info(f"[Miner] Iteration {iteration}: sampling {n_samples} molecules")

        sampler_data = run_sampler(n_samples=n_samples, 
                        subnet_config=subnet_config, 
                        output_path=sampler_file_path,
                        save_to_file=True,
                        db_path=db_path,
                        )
        
        if not sampler_data:
            bt.logging.warning("[Miner] No valid molecules produced; continuing")
            continue

        score_dict = score_molecules_json(sampler_file_path, target_proteins, antitarget_proteins, subnet_config)
        
        if not score_dict:
            bt.logging.warning("[Miner] Scoring failed or mismatched; continuing")
            continue

        # Calculate final scores per molecule
        batch_scores = calculate_final_scores(score_dict, sampler_data, subnet_config, save_all_scores, current_epoch)

        # Merge, deduplicate, sort and take top x
        top_pool = pd.concat([top_pool, batch_scores])
        top_pool = top_pool.drop_duplicates(subset=["InChIKey"], keep="first")
        top_pool = top_pool.sort_values(by="score", ascending=False)
        top_pool = top_pool.head(top_x)

        # format to accepted format
        top_entries = {"molecules": top_pool["name"].tolist()}

        # write to file
        with open(output_path, "w") as f:
            json.dump(top_entries, f, ensure_ascii=False, indent=2)

        bt.logging.info(f"[Miner] Wrote {top_x} top molecules to {output_path}")
        bt.logging.info(f"[Miner] Average score: {top_pool['score'].mean()}")

def calculate_final_scores(score_dict: dict, 
        sampler_data: dict, 
        subnet_config: dict, 
        save_all_scores: bool = True,
        current_epoch: int = 0) -> pd.DataFrame:
    """
    Calculate final scores per molecule
    """

    names = sampler_data["molecules"]
    smiles = [get_smiles_from_reaction(name) for name in names]

    # Calculate InChIKey for each molecule to deduplicate molecules after merging
    inchikey_list = []
    
    for s in smiles:
        try:
            inchikey_list.append(Chem.MolToInchiKey(Chem.MolFromSmiles(s)))
        except Exception as e:
            bt.logging.error(f"Error calculating InChIKey for {s}: {e}")
            inchikey_list.append(None)

    # Calculate final scores for each molecule
    targets = score_dict[0]['target_scores']
    antitargets = score_dict[0]['antitarget_scores']
    final_scores = []
    for mol_idx in range(len(names)):
        # target average
        target_scores_for_mol = [target_list[mol_idx] for target_list in targets]
        avg_target = sum(target_scores_for_mol) / len(target_scores_for_mol)

        # antitarget average
        antitarget_scores_for_mol = [antitarget_list[mol_idx] for antitarget_list in antitargets]
        avg_antitarget = sum(antitarget_scores_for_mol) / len(antitarget_scores_for_mol)

        # final score
        score = avg_target - (subnet_config["antitarget_weight"] * avg_antitarget)
        final_scores.append(score)

    # Store final scores in dataframe
    batch_scores = pd.DataFrame({
        "name": names,
        "smiles": smiles,
        "InChIKey": inchikey_list,
        "score": final_scores
    })

    if save_all_scores:
        all_scores = {"scored_molecules": [(mol["name"], mol["score"]) for mol in batch_scores.to_dict(orient="records")]}
        
        if os.path.exists(os.path.join(BASE_DIR, f"all_scores_{current_epoch}.json")):
            with open(os.path.join(BASE_DIR, f"all_scores_{current_epoch}.json"), "r") as f:
                all_previous_scores = json.load(f)
            
            all_scores["scored_molecules"] = all_previous_scores["scored_molecules"] + all_scores["scored_molecules"]

        with open(os.path.join(BASE_DIR, f"all_scores_{current_epoch}.json"), "w") as f:
            json.dump(all_scores, f, ensure_ascii=False, indent=2)

    return batch_scores
        

async def main(config):
    """
    Main loop
    """
    test_mode = bool(getattr(config, 'test_mode', False))
    
    # Initialize subtensor client
    subtensor = bt.async_subtensor(network=config.network)
    await subtensor.initialize()

    # Main miner loop
    last_logged_blocks_remaining = None
    while True:
        try:
            metagraph = await subtensor.metagraph(config.netuid)
            current_block = await subtensor.get_current_block()

            await process_epoch(config, current_block, metagraph, subtensor)

            if current_block % config.epoch_length != 0:
                # Epoch end - process and set weights
                config.update(load_config())
                await process_epoch(config, current_block, metagraph, subtensor)
                
            # else:
            #     # Waiting for epoch
            #     blocks_remaining = config.epoch_length - (current_block % config.epoch_length)
            #     if (blocks_remaining % 5 == 0) and (blocks_remaining != last_logged_blocks_remaining):
            #         bt.logging.info(f"Waiting for epoch to end... {blocks_remaining} blocks remaining.")
            #         last_logged_blocks_remaining = blocks_remaining
            #     await asyncio.sleep(1)
                
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

def miner_main(payload):
    # Minimal shim: trigger existing pipeline and direct outputs to /output.
    out_dir = os.environ.get("OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)

    cfg = load_config()
    # Use provided target if present; otherwise keep miner's own config/selection.
    target = payload.get("target")
    target_proteins = [target] if isinstance(target, str) and target else []
    antitarget_proteins = []

    # Derive loop parameters from existing config
    n_samples = cfg["num_molecules"] * 5
    top_x = cfg["num_molecules"]

    sampler_file_path = os.path.join(out_dir, "sampler.json")
    output_path = os.path.join(out_dir, "output.json")

    iterative_sampling_loop(
        n_samples=n_samples,
        top_x=top_x,
        target_proteins=target_proteins,
        antitarget_proteins=antitarget_proteins,
        sampler_file_path=sampler_file_path,
        output_path=output_path,
        subnet_config=cfg,
        save_all_scores=True,
        current_epoch=0, # TODO: change to current epoch. Is it possivle to have it on payload?
    )
    return