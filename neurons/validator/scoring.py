"""
PSICHIC-based molecular scoring functionality
"""

import math
import os
import json
from typing import List, Dict
import asyncio

import pandas as pd
import bittensor as bt
import numpy as np

import sys
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
NOVA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
if NOVA_DIR not in sys.path:
    sys.path.append(NOVA_DIR)

from utils.proteins import get_sequence_from_protein_code, get_code_from_protein_sequence
from neurons.validator.validity import validate_molecules_and_calculate_entropy
from PSICHIC.wrapper import PsichicWrapper
from neurons.validator.ranking import calculate_final_scores, determine_winner
from neurons.validator.save_data import submit_epoch_results

# Global variable to store PSICHIC instance - will be set by validator.py
psichic = None

async def process_epoch(config, epoch: int, uid_to_data: dict):
    """
    Process a single epoch end-to-end.
    """
    global psichic
    try:
        current_epoch = epoch
        
        # get target and antitarget sequences from config
        target_sequences = config["target_sequences"]
        antitarget_sequences = config["antitarget_sequences"]
        allowed_reaction = config.get("allowed_reaction")

        target_codes = [get_code_from_protein_sequence(sequence) for sequence in target_sequences]
        antitarget_codes = [get_code_from_protein_sequence(sequence) for sequence in antitarget_sequences]

        config["target_codes"] = target_codes
        config["antitarget_codes"] = antitarget_codes

        if allowed_reaction:
            bt.logging.info(f"Allowed reaction this epoch: {allowed_reaction}")

        bt.logging.info(f"Scoring using target proteins: {target_codes}, antitarget proteins: {antitarget_codes}")

        if not uid_to_data:
            bt.logging.info("No valid submissions found this epoch.")
            return None

        # Initialize scoring structure
        score_dict = {
            uid: {
                "target_scores": [[] for _ in range(len(target_codes))],
                "antitarget_scores": [[] for _ in range(len(antitarget_codes))],
                "entropy": None,
                "github_data": uid_to_data[uid].get("github_data", None)
            }
            for uid in uid_to_data
        }

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
            bt.logging.info("PSICHIC model initialized successfully")
        
        # Score all target proteins then all antitarget proteins one protein at a time
        score_all_proteins_psichic(
            target_proteins=target_sequences,
            antitarget_proteins=antitarget_sequences,
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
        winner = determine_winner(score_dict, current_epoch)

        # Yield so ws heartbeats can run before the next RPC
        await asyncio.sleep(0)

        # Submit results to dashboard API if configured
        try:
            submit_url = os.environ.get('SUBMIT_RESULTS_URL')
            if submit_url:
                status = submit_epoch_results(
                    config=config,
                    epoch=current_epoch,
                    target_proteins=target_codes,
                    antitarget_proteins=antitarget_codes,
                    uid_to_data=uid_to_data,
                    valid_molecules_by_uid=valid_molecules_by_uid,
                    score_dict=score_dict
                )
                if status:
                    bt.logging.info("Submitted results to dashboard DB")
        except Exception as e:
            bt.logging.error(f"Failed to submit results to dashboard DB: {e}")

        # Monitor validators
        # if not bool(getattr(config, 'test_mode', False)):
        #     try:
        #         set_weights_call_block = await subtensor.get_current_block()
        #     except asyncio.CancelledError:
        #         bt.logging.info("Resetting subtensor connection.")
        #         subtensor = bt.async_subtensor(network=config.network)
        #         await subtensor.initialize()
        #         await asyncio.sleep(1)
        #         set_weights_call_block = await subtensor.get_current_block()
        #     monitor_validator(
        #         score_dict=score_dict,
        #         metagraph=metagraph,
        #         current_epoch=current_epoch,
        #         current_block=set_weights_call_block,
        #         validator_hotkey=wallet.hotkey.ss58_address,
        #         winning_uid=winner
        #     )

        if winner is not None:
            try:
                winner_score = float(score_dict[winner].get('ps_final_score'))
            except Exception:
                winner_score = None
        else:
            winner_score = None
        return winner, winner_score

    except Exception as e:
        bt.logging.error(f"Error processing epoch: {e}")
        return None, None

def score_all_proteins_psichic(
    target_proteins: list[str],
    antitarget_proteins: list[str],
    score_dict: dict[int, dict[str, list[list[float]]]],
    valid_molecules_by_uid: dict[int, dict[str, list[str]]],
    uid_to_data: dict = None,
    batch_size: int = 32
) -> None:
    """
    Score all molecules against all proteins using efficient batching.
    This replaces the need to call score_protein_for_all_uids multiple times.
    
    Args:
        target_proteins: List of target protein codes
        antitarget_proteins: List of antitarget protein codes
        score_dict: Dictionary to store scores
        valid_molecules_by_uid: Dictionary of valid molecules by UID
        uid_to_data: Original UID data (for fallback molecule counts)
        batch_size: Number of molecules to process in each batch
    """
    global psichic
    
    # Ensure psichic is initialized
    if psichic is None:
        bt.logging.error("PSICHIC model not initialized.")
        return
    
    all_proteins = target_proteins + antitarget_proteins
    
    # Process each protein
    for protein_idx, protein in enumerate(all_proteins):
        is_target = protein_idx < len(target_proteins)
        col_idx = protein_idx if is_target else protein_idx - len(target_proteins)
        
        # Initialize PSICHIC for this protein
        if len(protein) < 10:
            protein_code = protein
            protein_sequence = get_sequence_from_protein_code(protein_code)    
        else:
            protein_sequence = protein

        bt.logging.info(f'Initializing model for protein: {protein}')
        
        try:
            psichic.initialize_model(protein_sequence)
            bt.logging.info('Model initialized successfully.')
        except Exception as e:
            try:
                if BASE_DIR:
                    os.system(f"wget -O {os.path.join(BASE_DIR, 'PSICHIC/trained_weights/TREAT2/model.pt')} https://huggingface.co/Metanova/TREAT-2/resolve/main/model.pt")
                psichic.initialize_model(protein_sequence)
                bt.logging.info('Model initialized successfully.')
            except Exception as e:
                bt.logging.error(f'Error initializing model: {e}')
                # Set all scores to -inf for this protein
                for uid in score_dict:
                    num_molecules = len(valid_molecules_by_uid.get(uid, {}).get('smiles', []))
                    if num_molecules == 0 and uid_to_data:
                        num_molecules = len(uid_to_data.get(uid, {}).get("molecules", []))
                    score_dict[uid]["target_scores" if is_target else "antitarget_scores"][col_idx] = [-math.inf] * num_molecules
                continue
        
        # Collect all unique molecules across all UIDs
        unique_molecules = {}  # {smiles: [(uid, mol_idx), ...]}
        
        for uid, valid_molecules in valid_molecules_by_uid.items():
            if not valid_molecules.get('smiles'):
                # Set -inf scores for UIDs with no valid molecules
                num_molecules = 0
                if uid_to_data:
                    num_molecules = len(uid_to_data.get(uid, {}).get("molecules", []))
                score_dict[uid]["target_scores" if is_target else "antitarget_scores"][col_idx] = [-math.inf] * num_molecules
                continue
            
            for mol_idx, smiles in enumerate(valid_molecules['smiles']):
                if smiles not in unique_molecules:
                    unique_molecules[smiles] = []
                unique_molecules[smiles].append((uid, mol_idx))
        
        # Process unique molecules in batches
        unique_smiles_list = list(unique_molecules.keys())
        molecule_scores = {}  # {smiles: score}
        
        for batch_start in range(0, len(unique_smiles_list), batch_size):
            batch_end = min(batch_start + batch_size, len(unique_smiles_list))
            batch_molecules = unique_smiles_list[batch_start:batch_end]
            
            try:
                # Score the batch
                results_df = psichic.score_molecules(batch_molecules)
                
                if not results_df.empty and len(results_df) == len(batch_molecules):
                    for idx, smiles in enumerate(batch_molecules):
                        val = results_df.iloc[idx].get('predicted_binding_affinity')
                        score_value = float(val) if val is not None else -math.inf
                        molecule_scores[smiles] = score_value
                else:
                    bt.logging.warning(f"Unexpected results for batch, falling back to individual scoring")
                    for smiles in batch_molecules:
                        molecule_scores[smiles] = score_molecule_individually(smiles)
            except Exception as e:
                bt.logging.error(f"Error scoring batch: {e}")
                for smiles in batch_molecules:
                    molecule_scores[smiles] = score_molecule_individually(smiles)
        
        # Distribute scores to all UIDs
        for uid, valid_molecules in valid_molecules_by_uid.items():
            if not valid_molecules.get('smiles'):
                continue
            
            uid_scores = []
            for smiles in valid_molecules['smiles']:
                score = molecule_scores.get(smiles, -math.inf)
                uid_scores.append(score)
            
            if is_target:
                score_dict[uid]["target_scores"][col_idx] = uid_scores
            else:
                score_dict[uid]["antitarget_scores"][col_idx] = uid_scores
        
        bt.logging.info(f"Completed scoring for protein {protein}: {len(unique_molecules)} unique molecules")


def score_molecule_individually(smiles: str) -> float:
    """Helper function to score a single molecule."""
    global psichic
    
    if psichic is None:
        bt.logging.error("PSICHIC model not initialized.")
        return -math.inf
    
    try:
        results_df = psichic.score_molecules([smiles])
        if not results_df.empty:
            val = results_df.iloc[0].get('predicted_binding_affinity')
            return float(val) if val is not None else -math.inf
        else:
            return -math.inf
    except Exception as e:
        bt.logging.error(f"Error scoring molecule {smiles}: {e}")
        return -math.inf


def read_miner_output_from_json(path: str) -> pd.DataFrame:
    """
    Reads molecules from JSON file. Format must be {"uid": ..., 
    "block_number": ..., "owner": ..., "repo": ..., "branch": ..., "raw": ..., 
    "result": {"molecules": [...]
    }
    """
    if not os.path.exists(path):
        bt.logging.error(f"Could not find JSON file at '{path}'")
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # structure as in uid_to_data
    if isinstance(data, dict):
        data = [data]

    # if "uid" not in data[0], add it - for random sampler, assign uid=0
    # TODO: change this to accept specific format of the combined miner output JSON file
    if "uid" not in data[0]:
        data = [{"uid": 0, 
        "result": {"molecules": data[0]["molecules"]},
        "github_data": None,
        "coldkey": None,
        "hotkey": None
        }]

    # Every other miner will have a "uid" field
    uid_to_data = {item["uid"]: {"molecules": item["result"]["molecules"], 
        "github_data": item["raw"],
        "coldkey": item["coldkey"],
        "hotkey": item["hotkey"]} 
        for item in data
        }

    return uid_to_data

def score_molecules_json(
    input_path: str,
    target_proteins: list[str],
    antitarget_proteins: list[str],
    subnet_config: dict,
) -> dict:
    """
    End-to-end scoring:
    - Read molecules from JSON file
    - Score with PSICHIC (random target and antitargets)

    Returns score_dict.
    """
    global psichic

    uid_to_data = read_miner_output_from_json(input_path)
    if not uid_to_data:
        bt.logging.error("No molecules found in JSON file.")
        return None

    # Initialize scoring structure
    score_dict = {
        uid: {
            "target_scores": [[] for _ in range(len(target_proteins))],
            "antitarget_scores": [[] for _ in range(len(antitarget_proteins))],
            "entropy": None,
        }
        for uid in uid_to_data
    }

    # Check validity of submissions
    valid_molecules_by_uid = validate_molecules_and_calculate_entropy(uid_to_data, score_dict, subnet_config)
    
    # Score with PSICHIC
    if psichic is None:
        psichic = PsichicWrapper()
    score_all_proteins_psichic(target_proteins,
                                antitarget_proteins,
                                score_dict,
                                valid_molecules_by_uid,
                                uid_to_data,
                                32
                                )
    psichic.cleanup_model()
    psichic = None

    return score_dict

def calculate_histogram(score_list: list[float]) -> np.array:
    """
    Calculate the histogram of a list of scores.
    """
    return np.histogram(score_list, bins=100, range=(0, 10))


