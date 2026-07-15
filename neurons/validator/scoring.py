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
from neurons.validator.ranking import calculate_final_scores
from neurons.validator.contest import apply_contest_transition
from neurons.validator.save_data import submit_epoch_results

# Global variable to store PSICHIC instance - will be set by validator.py
psichic = None

def _build_thompson_benchmark_payload(
    *,
    epoch_number: int,
    config: dict,
    current_epoch: int,
    target_sequences: list[str],
    antitarget_sequences: list[str],
) -> list[dict]:

    try:
        jsonl_path = os.path.join(
            "/data/results", f"period_{int(epoch_number)}_results.jsonl"
        )
        if not os.path.exists(jsonl_path):
            return []

        # Pull TS molecules directly from the cumulative JSONL results file (uid=-2).
        molecules: list[str] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                try:
                    uid = int(rec.get("uid"))
                except Exception:
                    continue
                if uid != -2:
                    continue
                res = rec.get("result") or {}
                if isinstance(res, dict):
                    mols = res.get("molecules", [])
                    if isinstance(mols, list):
                        molecules = mols  # keep last occurrence if duplicated

        if not isinstance(molecules, list) or not molecules:
            return []

        # Use a negative UID that is never part of miner ranking/submissions.
        uid = -2
        github_data = None
        bench_uid_to_data = {uid: {"molecules": molecules, "raw": github_data}}
        bench_score_dict = {
            uid: {
                "ps_target_scores": [[] for _ in range(len(config.get("target_codes", [])))],
                "ps_antitarget_scores": [[] for _ in range(len(config.get("antitarget_codes", [])))],
                "entropy": None,
                "github_data": github_data,
            }
        }

        bench_valid = validate_molecules_and_calculate_entropy(
            uid_to_data=bench_uid_to_data,
            score_dict=bench_score_dict,
            config=config,
            allowed_reaction=config.get("allowed_reaction"),
        )
        if uid not in bench_valid:
            return []

        score_all_proteins_psichic(
            target_proteins=target_sequences,
            antitarget_proteins=antitarget_sequences,
            score_dict=bench_score_dict,
            valid_molecules_by_uid=bench_valid,
            uid_to_data=bench_uid_to_data,
            batch_size=32,
        )
        bench_score_dict = calculate_final_scores(
            bench_score_dict, bench_valid, config, current_epoch
        )

        names = bench_valid[uid].get("names", [])
        combined = bench_score_dict.get(uid, {}).get("ps_combined_molecule_scores", [])
        n = min(len(names), len(combined))
        scored_molecules = [[str(names[j]), float(combined[j])] for j in range(n)]
        if not scored_molecules:
            return []
        return [{"name": "thompson_sampling", "github_data": None, "scored_molecules": scored_molecules}]
    except Exception as e:
        bt.logging.warning(
            f"failed to build thompson_sampling benchmark payload: {type(e).__name__}: {e}"
        )
        return []


async def process_epoch(
    config,
    epoch_number: int,
    entries: dict,
    state: dict,
    scored_sample_path: str,
):
    """
    Score every entry (keyed by entry_id) and resolve the champion contest.

    Returns (new_state, champion_entry_id, champion_score).
    """
    global psichic
    try:
        current_epoch = epoch_number

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

        if not entries:
            bt.logging.info("No valid submissions found this epoch.")
            return state, None, None

        # Initialize scoring structure
        score_dict = {
            eid: {
                "ps_target_scores": [[] for _ in range(len(target_codes))],
                "ps_antitarget_scores": [[] for _ in range(len(antitarget_codes))],
                "entropy": None,
                "github_data": entries[eid].get("github_data"),
            }
            for eid in entries
        }

        # Validate molecules and calculate entropy
        valid_molecules_by_entry = validate_molecules_and_calculate_entropy(
            uid_to_data=entries,
            score_dict=score_dict,
            config=config,
            allowed_reaction=allowed_reaction,
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
            valid_molecules_by_uid=valid_molecules_by_entry,
            uid_to_data=entries,
            batch_size=32,
        )

        # Calculate final scores
        score_dict = calculate_final_scores(
            score_dict, valid_molecules_by_entry, config, current_epoch
        )

        # The margin a challenger must clear to beat the champion (stored on the competition row).
        config["threshold_to_win"] = float(config["improvement_margin"])

        # Resolve the champion contest
        new_state, champion_entry_id, champion_score = apply_contest_transition(
            score_dict=score_dict,
            entries=entries,
            state=state,
            cfg=config,
            epoch=epoch_number,
        )

        # Yield so ws heartbeats can run before the next RPC
        await asyncio.sleep(0)

        # Submit results to dashboard API if configured
        try:
            if os.environ.get("BACKEND_API_URL"):
                benchmarks_payload = _build_thompson_benchmark_payload(
                    epoch_number=epoch_number,
                    config=config,
                    current_epoch=current_epoch,
                    target_sequences=target_sequences,
                    antitarget_sequences=antitarget_sequences,
                )
                status = await submit_epoch_results(
                    config=config,
                    epoch_number=epoch_number,
                    target_proteins=target_codes,
                    antitarget_proteins=antitarget_codes,
                    entries=entries,
                    valid_molecules_by_entry=valid_molecules_by_entry,
                    score_dict=score_dict,
                    state=new_state,
                    scored_sample_path=scored_sample_path,
                    benchmarks=benchmarks_payload,
                )
                if status:
                    bt.logging.info("Submitted results to dashboard DB")
        except Exception as e:
            bt.logging.error(f"Failed to submit results to dashboard DB: {e}")

        return new_state, champion_entry_id, champion_score

    except Exception as e:
        bt.logging.error(f"Error processing epoch: {e}")
        return state, None, None

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
                # Download PSICHIC weights using wget into the standard PDBv2020_PSICHIC path
                os.system(f"wget -q -O {os.path.join(NOVA_DIR, 'PSICHIC/trained_weights/PDBv2020_PSICHIC/model.pt')} https://huggingface.co/Metanova/PSICHIC/resolve/main/model.pt")
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
                score_dict[uid]["ps_target_scores" if is_target else "ps_antitarget_scores"][col_idx] = [-math.inf] * num_molecules
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
                score_dict[uid]["ps_target_scores"][col_idx] = uid_scores
            else:
                score_dict[uid]["ps_antitarget_scores"][col_idx] = uid_scores
        
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
    Reads miner outputs from JSON or JSONL.
    Expected object keys: "uid", "raw", "coldkey", "hotkey", "result": {"molecules": [...]}
    """
    if not os.path.exists(path):
        bt.logging.error(f"Could not find JSON file at '{path}'")
        return None

    uid_to_data = {}

    # JSONL input
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception as e:
                    bt.logging.warning(f"Skipping malformed JSONL line {ln}: {e}")
                    continue

                uid = item.get("uid", 0)
                molecules = item.get("result", {}).get("molecules") or item.get("molecules", [])
                uid_to_data[uid] = {
                    "molecules": molecules,
                    "github_data": item.get("raw", None),
                    "coldkey": item.get("coldkey", None),
                    "hotkey": item.get("hotkey", None),
                }
        return uid_to_data

    # JSON input
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    if "uid" not in data[0]:
        data = [{
            "uid": -1,
            "result": {"molecules": data[0].get("molecules", [])},
            "raw": None,
            "coldkey": None,
            "hotkey": None,
        }]

    uid_to_data = {
        item["uid"]: {
            "molecules": item.get("result", {}).get("molecules", []),
            "github_data": item.get("raw", None),
            "coldkey": item.get("coldkey", None),
            "hotkey": item.get("hotkey", None),
        }
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
            "ps_target_scores": [[] for _ in range(len(target_proteins))],
            "ps_antitarget_scores": [[] for _ in range(len(antitarget_proteins))],
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


