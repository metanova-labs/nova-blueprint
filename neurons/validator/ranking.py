"""
Per-molecule and final scoring for the validator.
"""

import math
import numpy as np

import bittensor as bt


def calculate_final_scores(
    score_dict: dict[int, dict[str, list[list[float]]]],
    valid_molecules_by_uid: dict[int, dict[str, list[str]]],
    config: dict,
    current_epoch: int
) -> dict[int, dict[str, list[list[float]]]]:
    """
    Calculates final scores per molecule for each UID, considering target and antitarget scores.
    
    Args:
        score_dict: Dictionary containing scores for each UID
        valid_molecules_by_uid: Dictionary of valid molecules by UID
        config: Configuration dictionary
        current_epoch: Current epoch number
        
    Returns:
        Updated score_dict with final scores calculated
    """
    
    # Go through each UID scored
    for uid, data in valid_molecules_by_uid.items():
        targets = score_dict[uid]['ps_target_scores']
        antitargets = score_dict[uid]['ps_antitarget_scores']
        entropy = score_dict[uid]['entropy']

        # Replace None with -inf
        targets = [[-math.inf if not s else s for s in sublist] for sublist in targets]
        antitargets = [[-math.inf if not s else s for s in sublist] for sublist in antitargets]

        # Get number of molecules (length of any target score list)
        if not targets or not targets[0]:
            continue
        num_molecules = len(targets[0])

        # Calculate scores per molecule
        combined_molecule_scores = []
        
        for mol_idx in range(num_molecules):
            # Calculate average target score for this molecule
            target_scores_for_mol = [target_list[mol_idx] for target_list in targets]
            if any(score == -math.inf for score in target_scores_for_mol):
                combined_molecule_scores.append(-math.inf)
                continue
            avg_target = sum(target_scores_for_mol) / len(target_scores_for_mol)

            # Calculate average antitarget score for this molecule
            antitarget_scores_for_mol = [antitarget_list[mol_idx] for antitarget_list in antitargets]
            if any(score == -math.inf for score in antitarget_scores_for_mol):
                combined_molecule_scores.append(-math.inf)
                continue
            avg_antitarget = sum(antitarget_scores_for_mol) / len(antitarget_scores_for_mol)

            # Calculate score after target/antitarget combination
            mol_score = avg_target - (config['antitarget_weight'] * avg_antitarget)
            combined_molecule_scores.append(mol_score)
        
        # Store all score lists in score_dict
        score_dict[uid]['ps_combined_molecule_scores'] = combined_molecule_scores
        score_dict[uid]['ps_final_score'] = np.mean(combined_molecule_scores)
                
        # Log details
        # Prepare detailed log info
        smiles_list = data.get('smiles', [])
        names_list = data.get('names', [])
        # Transpose target/antitarget scores to get per-molecule lists
        target_scores_per_mol = list(map(list, zip(*targets))) if targets and targets[0] else []
        antitarget_scores_per_mol = list(map(list, zip(*antitargets))) if antitargets and antitargets[0] else []
        log_lines = [
            f"UID={uid}",
            f"  Molecule names: {names_list}",
            f"  SMILES: {smiles_list}",
            f"  Target scores per molecule: {target_scores_per_mol}",
            f"  Antitarget scores per molecule: {antitarget_scores_per_mol}",
            f"  Entropy: {entropy}",
            f"  Final score: {score_dict[uid]['ps_final_score']}"
        ]
        bt.logging.info("\n".join(log_lines))

    return score_dict

