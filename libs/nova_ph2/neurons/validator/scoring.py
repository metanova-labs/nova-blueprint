"""
PSICHIC-based molecular scoring functionality
"""

import math
import os
import json
from typing import List, Dict

import pandas as pd
import bittensor as bt
import numpy as np

import sys
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
NOVA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
if NOVA_DIR not in sys.path:
    sys.path.append(NOVA_DIR)

from config.config_loader import load_config
from utils.proteins import get_sequence_from_protein_code
from neurons.validator.validity import validate_molecules_and_calculate_entropy
from PSICHIC.wrapper import PsichicWrapper

# Global variable to store PSICHIC instance - will be set by validator.py
psichic = None

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
        bt.logging.info(f'Initializing model for protein: {protein}')
        #protein_sequence = get_sequence_from_protein_code(protein)
        protein_sequence = protein
        
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


def read_data_from_json(path: str) -> pd.DataFrame:
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
    if "uid" not in data[0]:
        data = [{"uid": 0, 
        "result": {"molecules": data[0]["molecules"]},
        "block_number": 0
        }]

    # Every other miner will have a "uid" field
    uid_to_data = {item["uid"]: {"molecules": item["result"]["molecules"], 
        "block_submitted": item["block_number"]} 
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

    uid_to_data = read_data_from_json(input_path)
    if not uid_to_data:
        bt.logging.error("No molecules found in JSON file.")
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


