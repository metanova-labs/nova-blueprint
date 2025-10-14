"""
Final scoring and winner determination functionality for the validator
"""

import math
import numpy as np
import datetime
from typing import Optional

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
        targets = score_dict[uid]['target_scores']
        antitargets = score_dict[uid]['antitarget_scores']
        entropy = score_dict[uid]['entropy']
        submission_block = score_dict[uid]['block_submitted']

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
        score_dict[uid]['combined_molecule_scores'] = combined_molecule_scores
        score_dict[uid]['final_score'] = np.mean(combined_molecule_scores)
                
        # TODO: decide what we want to do with entropy

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
            #f"  Entropy: {entropy}",
            f"  Final score: {score_dict[uid]['final_score']}"
        ]
        bt.logging.info("\n".join(log_lines))

    return score_dict


def determine_winner(score_dict: dict[int, dict[str, list[list[float]]]]) -> Optional[int]:
    """
    Determines the winning UID based on final score.
    In case of ties, earliest submission time is used as the tiebreaker.
    
    Args:
        score_dict: Dictionary containing final scores for each UID
        
    Returns:
        Optional[int]: Winning UID or None if no valid scores found
    """
    best_score = -math.inf
    best_uids = []

    def parse_timestamp(uid):
        ts = score_dict[uid].get('push_time', '')
        try:
            return datetime.datetime.fromisoformat(ts)
        except Exception as e:
            bt.logging.warning(f"Failed to parse timestamp '{ts}' for UID={uid}: {e}")
            return datetime.datetime.max.replace(tzinfo=datetime.timezone.utc)

    def tie_breaker(tied_uids: list[int], best_score: float):
        # Sort by block number first, then push time, then uid to ensure deterministic result
        winner = sorted(tied_uids, key=lambda uid: (
            score_dict[uid].get('block_submitted', float('inf')), 
            parse_timestamp(uid), 
            uid
        ))[0]
        
        winner_block = score_dict[winner].get('block_submitted')
        current_epoch = winner_block // 361 if winner_block else None
        push_time = score_dict[winner].get('push_time', '')
        
        tiebreaker_message = f"Epoch {current_epoch} tiebreaker winner: UID={winner}, score={best_score}, block={winner_block}"
        if push_time:
            tiebreaker_message += f", push_time={push_time}"
            
        bt.logging.info(tiebreaker_message)
            
        return winner
    
    # Find highest final score
    for uid, data in score_dict.items():
        if 'final_score' not in data:
            continue
        
        score = round(data['final_score'], 4)
        
        if score > best_score:
            best_score = score
            best_uids = [uid]
        elif score == best_score:
            best_uids.append(uid)
    
    if not best_uids:
        bt.logging.info("No valid winner found (all scores -inf or no submissions).")
        return None
    
    # Select winner from each model
    if best_uids:
        if len(best_uids) == 1:
            winner_block = score_dict[best_uids[0]].get('block_submitted')
            current_epoch = winner_block // 361 if winner_block else None
            bt.logging.info(f"Epoch {current_epoch} winner: UID={best_uids[0]}, winning_score={best_score}")
            winner = best_uids[0]
        else:
            winner = tie_breaker(best_uids, best_score)
    else:
        winner = None
    
    return winner
    
