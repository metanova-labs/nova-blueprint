import bittensor as bt
from rdkit import Chem
from rdkit.Chem import Descriptors

from utils import (
    get_smiles, 
    get_heavy_atom_count, 
    compute_maccs_entropy,
    find_chemically_identical,
    is_reaction_allowed
)


def validate_molecules_and_calculate_entropy(
    uid_to_data: dict[int, dict[str, list]],
    score_dict: dict[int, dict[str, list[list[float]]]],
    config: dict,
    allowed_reaction: str = None
) -> dict[int, dict[str, list[str]]]:
    """
    Validates molecules for all UIDs and calculates their MACCS entropy.
    Updates the score_dict with entropy values.
    
    Args:
        uid_to_data: Dictionary mapping UIDs to their data including molecules
        score_dict: Dictionary to store scores and entropy
        config: Configuration dictionary containing validation parameters
        allowed_reaction: Optional allowed reaction filter for this epoch
        
    Returns:
        Dictionary mapping UIDs to their list of valid SMILES strings
    """
    valid_molecules_by_uid = {}
    
    for uid, data in uid_to_data.items():
        valid_smiles = []
        valid_names = []
        
        # Check for duplicate molecules in submission
        if len(data["molecules"]) != len(set(data["molecules"])):
            bt.logging.error(f"UID={uid} submission contains duplicate molecules")
            score_dict[uid]["entropy"] = None
            score_dict[uid]["block_submitted"] = None
            continue
            
        for molecule in data["molecules"]:
            try:
                # Check if reaction is allowed this epoch (if filtering enabled)
                # (temporarily disabled):
                # if config.get('random_valid_reaction') and not is_reaction_allowed(molecule, allowed_reaction):
                #     bt.logging.warning(
                #         f"UID={uid}, molecule='{molecule}' uses disallowed reaction for this epoch (only {allowed_reaction} allowed)"
                #     )
                #     valid_smiles = []
                #     valid_names = []
                #     break

                # temporary: Always allow reactions 4 and 5, ignore config/random selection
                # allowed_ok = is_reaction_allowed(molecule, "rxn:4") or is_reaction_allowed(molecule, "rxn:5")
                # if not allowed_ok:
                #     bt.logging.warning(
                #         f"UID={uid}, molecule='{molecule}' uses disallowed reaction for this temporary window (only 4 or 5 allowed)"
                #     )
                #     valid_smiles = []
                #     valid_names = []
                #     break
                
                smiles = get_smiles(molecule)
                if not smiles:
                    bt.logging.error(f"No valid SMILES found for UID={uid}, molecule='{molecule}'")
                    valid_smiles = []
                    valid_names = []
                    break
                
                if get_heavy_atom_count(smiles) < config['min_heavy_atoms']:
                    bt.logging.warning(f"UID={uid}, molecule='{molecule}' has insufficient heavy atoms")
                    valid_smiles = []
                    valid_names = []
                    break

                try:
                    mol = Chem.MolFromSmiles(smiles)
                    num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
                    if num_rotatable_bonds < config['min_rotatable_bonds'] or num_rotatable_bonds > config['max_rotatable_bonds']:
                        bt.logging.warning(f"UID={uid}, molecule='{molecule}' has an invalid number of rotatable bonds")
                        valid_smiles = []
                        valid_names = []
                        break
                except Exception as e:
                    bt.logging.error(f"Molecule is not parseable by RDKit for UID={uid}, molecule='{molecule}': {e}")
                    valid_smiles = []
                    valid_names = []
                    break
     
                valid_smiles.append(smiles)
                valid_names.append(molecule)
            except Exception as e:
                bt.logging.error(f"Error validating molecule for UID={uid}, molecule='{molecule}': {e}")
                valid_smiles = []
                valid_names = []
                break
            
        # Check for chemically identical molecules
        if valid_smiles:
            try:
                identical_molecules = find_chemically_identical(valid_smiles)
                if identical_molecules:
                    duplicate_names = []
                    for inchikey, indices in identical_molecules.items():
                        molecule_names = [valid_names[idx] for idx in indices]
                        duplicate_names.append(f"{', '.join(molecule_names)} (same InChIKey: {inchikey})")
                    
                    bt.logging.warning(f"UID={uid} submission contains chemically identical molecules: {'; '.join(duplicate_names)}")
                    score_dict[uid]["entropy"] = None
                    score_dict[uid]["block_submitted"] = None
                    continue 
            except Exception as e:
                bt.logging.warning(f"Error checking for chemically identical molecules for UID={uid}: {e}")

        
        # Calculate entropy if we have valid molecules, or skip if below threshold
        if valid_smiles:
            try:
                entropy = compute_maccs_entropy(valid_smiles)
                if entropy > config['entropy_min_threshold']:
                    score_dict[uid]["entropy"] = entropy
                    valid_molecules_by_uid[uid] = {"smiles": valid_smiles, "names": valid_names}
                    if uid != 0:
                        score_dict[uid]["block_submitted"] = data["block_submitted"]     
                    else:
                        score_dict[uid]["block_submitted"] = None
                else:
                    bt.logging.warning(f"UID={uid} submission has entropy below threshold: {entropy}")
                    score_dict[uid]["entropy"] = None
                    score_dict[uid]["block_submitted"] = None
                    valid_smiles = []
                    valid_names = []

            except Exception as e:
                bt.logging.error(f"Error calculating entropy for UID={uid}: {e}")
                score_dict[uid]["entropy"] = None
                score_dict[uid]["block_submitted"] = None
                valid_smiles = []
                valid_names = []
        else:
            score_dict[uid]["entropy"] = None
            score_dict[uid]["block_submitted"] = None
            
    return valid_molecules_by_uid


def validate_molecules_sampler(
    sampler_data: dict[int, dict[str, list]],
    config: dict,
) -> tuple[list[str], list[str]]:
    """
    Validates molecules for random sampler (uid=0).    
    Doesn't interrupt the process if a molecule is invalid, removes it from the list instead. 
    Doesn't check allowed reactions (handled in random_sampler.py)
    
    Args:
        sampler_data: Dictionary in the format of {"molecules": list[str]}
        config: Configuration dictionary containing validation parameters
        
    Returns:
        valid names and smiles (list[str], list[str])
    """
    
    molecules = sampler_data["molecules"]

    valid_smiles = []
    valid_names = []
                
    for molecule in molecules:
        try:
            if molecule is None:
                continue
            
            smiles = get_smiles(molecule)
            if not smiles:
                continue
            
            if get_heavy_atom_count(smiles) < config['min_heavy_atoms']:
                continue

            try:    
                mol = Chem.MolFromSmiles(smiles)
                num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
                if num_rotatable_bonds < config['min_rotatable_bonds'] or num_rotatable_bonds > config['max_rotatable_bonds']:
                    continue
            except Exception as e:
                continue
    
            valid_smiles.append(smiles)
            valid_names.append(molecule)
        except Exception as e:
            continue
        
    return valid_names, valid_smiles