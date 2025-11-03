import sqlite3
import random
import os
import json
from typing import List, Tuple, Optional
import bittensor as bt
from rdkit import Chem
from tqdm import tqdm

import sys
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
NOVA_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
if NOVA_DIR not in sys.path:
    sys.path.append(NOVA_DIR)

from nova_ph2.config.config_loader import load_config
from miner_utils import validate_molecules_sampler
from nova_ph2.combinatorial_db.reactions import (
    get_reaction_info, 
    get_smiles_from_reaction
)
from nova_ph2.utils import get_smiles, find_chemically_identical

def get_available_reactions(db_path: str = None) -> List[Tuple[int, str, int, int, int]]:
    """
    Get all available reactions from the database.
    
    Args:
        db_path: Path to the molecules database
        
    Returns:
        List of tuples (rxn_id, smarts, roleA, roleB, roleC)
    """
    if db_path is None:
        db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "combinatorial_db", "molecules.sqlite"))
    
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
        cursor = conn.cursor()
        cursor.execute("SELECT rxn_id, smarts, roleA, roleB, roleC FROM reactions")
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        bt.logging.error(f"Error getting available reactions: {e}")
        return []


def get_molecules_by_role(role_mask: int, db_path: str) -> List[Tuple[int, str, int]]:
    """
    Get all molecules that have the specified role_mask.
    
    Args:
        role_mask: The role mask to filter by
        db_path: Path to the molecules database
        
    Returns:
        List of tuples (mol_id, smiles, role_mask) for molecules that match the role
    """
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT mol_id, smiles, role_mask FROM molecules WHERE (role_mask & ?) = ?", 
            (role_mask, role_mask)
        )
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        bt.logging.error(f"Error getting molecules by role {role_mask}: {e}")
        return []



def generate_valid_random_molecules_batch(rxn_id: int, n_samples: int, db_path: str, subnet_config: dict, 
                                 batch_size: int = 200, seed: int = None) -> dict:
    """
    Efficiently generate n_samples valid molecules by generating them in batches and validating.
    
    Args:
        rxn_id: The reaction ID to use
        n_samples: Number of valid molecules to generate
        db_path: Path to the molecules database
        subnet_config: Configuration for validation
        batch_size: Number of molecules to generate per batch
        
    Returns:
        Dict of molecules for the sampler uid=0
    """
    # Pre-fetch molecule pools to avoid repeated database queries
    reaction_info = get_reaction_info(rxn_id, db_path)
    if not reaction_info:
        bt.logging.error(f"Could not get reaction info for rxn_id {rxn_id}")
        return {"molecules": [None] * n_samples}
    
    smarts, roleA, roleB, roleC = reaction_info
    is_three_component = roleC is not None and roleC != 0
    
    # Cache molecule pools to avoid repeated database queries
    molecules_A = get_molecules_by_role(roleA, db_path)
    molecules_B = get_molecules_by_role(roleB, db_path)
    molecules_C = get_molecules_by_role(roleC, db_path) if is_three_component else []
    
    if not molecules_A or not molecules_B or (is_three_component and not molecules_C):
        bt.logging.error(f"No molecules found for roles A={roleA}, B={roleB}, C={roleC}")
        return {"molecules": [None] * n_samples}
    
    valid_molecules = []
    seen_keys = set()
    iteration = 0

    progress_bar = tqdm(total=n_samples, desc="Creating valid molecules", unit="molecule")
    
    while len(valid_molecules) < n_samples:
        iteration += 1
        
        # Calculate how many molecules we still need
        needed = n_samples - len(valid_molecules)
        
        # Generate a batch of molecules (with some buffer for validation failures)
        batch_size_actual = min(batch_size, needed * 2)  # Generate 2x what we need to account for failures
        
        #bt.logging.debug(f"Iteration {iteration}: Generating {batch_size_actual} molecules, need {needed} more")
        
        # Generate batch molecules efficiently
        batch_molecules = generate_molecules_from_pools(
            rxn_id, batch_size_actual, molecules_A, molecules_B, molecules_C, is_three_component, seed
        )
        
        # Validate the batch
        batch_sampler_data = {"molecules": batch_molecules}
        batch_valid_molecules, batch_valid_smiles = validate_molecules_sampler(batch_sampler_data, subnet_config)

        # Deduplicate inside the batch (keep first per InChIKey)
        identical = find_chemically_identical(batch_valid_smiles)
        skip_indices = set()
        for indices in identical.values():
            for j in indices[1:]:
                skip_indices.add(j)

        # Add only chemically-unique molecules across all batches
        added = 0
        for i, name in enumerate(batch_valid_molecules):
            if i in skip_indices or not name:
                continue
            s = batch_valid_smiles[i] if i < len(batch_valid_smiles) else None
            if not s:
                continue
            try:
                mol = Chem.MolFromSmiles(s)
                if not mol:
                    continue
                key = Chem.MolToInchiKey(mol)
            except Exception:
                continue
            if key in seen_keys:
                continue

            seen_keys.add(key)
            valid_molecules.append(name)
            added += 1
        
        progress_bar.update(added)
        #bt.logging.debug(f"Batch validation: {len(batch_valid_molecules)}/{len(batch_molecules)} molecules passed")
    
    # Trim to exact number requested
    final_molecules = valid_molecules[:n_samples]
    progress_bar.close()

    return {"molecules": final_molecules}


def generate_molecules_from_pools(rxn_id: int, n: int, molecules_A: List[Tuple], molecules_B: List[Tuple], 
                                molecules_C: List[Tuple], is_three_component: bool, seed: int = None) -> List[str]:
    """
    Generate molecules using pre-fetched molecule pools to avoid database queries.
    
    Args:
        rxn_id: The reaction ID
        n: Number of molecules to generate
        molecules_A, molecules_B, molecules_C: Pre-fetched molecule pools
        is_three_component: Whether this is a 3-component reaction
        seed: Random seed for reproducibility (optional)
    Returns:
        List of molecule names
    """
    mol_ids = []

    if seed is not None:
        random.seed(seed)
    
    for i in range(n):
        try:
            # Randomly select molecules for each role
            mol_A = random.choice(molecules_A)
            mol_B = random.choice(molecules_B)
            
            mol_id_A, smiles_A, role_mask_A = mol_A
            mol_id_B, smiles_B, role_mask_B = mol_B
            
            if is_three_component:
                mol_C = random.choice(molecules_C)
                mol_id_C, smiles_C, role_mask_C = mol_C
                product_name = f"rxn:{rxn_id}:{mol_id_A}:{mol_id_B}:{mol_id_C}"
            else:
                product_name = f"rxn:{rxn_id}:{mol_id_A}:{mol_id_B}"
            
            mol_ids.append(product_name)
            
        except Exception as e:
            bt.logging.error(f"Error generating molecule {i+1}/{n}: {e}")
            mol_ids.append(None)
    
    return mol_ids


def run_sampler(n_samples: int = 1000, 
                seed: int = None, 
                subnet_config: dict = None, 
                output_path: str = None, 
                save_to_file: bool = True,
                db_path: str = None):
    reactions = get_available_reactions(db_path)
    if not reactions:
        bt.logging.error("No reactions found in the database, check db path and integrity.")
        return

    rxn_ids = [reactions[i][0] for i in range(len(reactions))]

    # TODO: change random reaction choice the one received from payload
    rxn_id = 4 #random.choice(rxn_ids)
    bt.logging.info(f"Generating {n_samples} random molecules for reaction {rxn_id}")

    # Generate molecules with validation in batches for efficiency
    sampler_data = generate_valid_random_molecules_batch(
        rxn_id, n_samples, db_path, subnet_config, batch_size=200, seed=seed
        )

    if save_to_file:
        with open(output_path, "w") as f:
            json.dump(sampler_data, f, ensure_ascii=False, indent=2)

    return sampler_data


if __name__ == "__main__":
    run_sampler()