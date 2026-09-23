import bittensor as bt
from rdkit import Chem

from utils.molecules import (
    get_smiles,
    compute_fingerprint_entropy,
    find_chemically_identical,
    find_too_similar_pairs,
)
from utils.filters import check as check_filters, validate_thresholds
from utils.reactions import is_reaction_allowed


def validate_molecules_and_calculate_entropy(
    entries_by_id: dict[str, dict[str, list]],
    score_dict: dict[int, dict[str, list[list[float]]]],
    config: dict,
    allowed_reaction: str = None
) -> dict[int, dict[str, list[str]]]:
    """
    Validates molecules for every entry and calculates their fingerprint entropy.
    Updates the score_dict with entropy values.
    
    Args:
        entries_by_id: Dictionary mapping entry ids to their data including molecules
        score_dict: Dictionary to store scores and entropy
        config: Configuration dictionary containing validation parameters
        allowed_reaction: Optional allowed reaction filter for this epoch
        
    Returns:
        Dictionary mapping entry ids to their list of valid SMILES strings
    """
    valid_molecules_by_entry = {}
    filter_thresholds = validate_thresholds(config.get("filters"))

    for entry, data in entries_by_id.items():
        valid_smiles = []
        valid_names = []
        
        required_num = config.get("num_molecules")
        if required_num is not None and len(data.get("molecules", [])) != int(required_num):
            bt.logging.warning(
                f"entry={entry} submission has wrong molecule count: "
                f"got={len(data.get('molecules', []))}, required={int(required_num)}"
            )
            score_dict[entry]["entropy"] = None
            continue
        
        # Check for duplicate molecules in submission
        if len(data["molecules"]) != len(set(data["molecules"])):
            bt.logging.error(f"entry={entry} submission contains duplicate molecules")
            score_dict[entry]["entropy"] = None
            continue
            
        for molecule in data["molecules"]:
            try:
                # Check if reaction is allowed this epoch (if filtering enabled)
                if allowed_reaction is not None and not is_reaction_allowed(molecule, allowed_reaction):
                    bt.logging.warning(
                        f"entry={entry}, molecule='{molecule}' uses disallowed reaction for this epoch (only {allowed_reaction} allowed)"
                    )
                    valid_smiles = []
                    valid_names = []
                    break

                # temporary: Always allow reactions 4 and 5, ignore config/random selection
                # allowed_ok = is_reaction_allowed(molecule, "rxn:4") or is_reaction_allowed(molecule, "rxn:5")
                # if not allowed_ok:
                #     bt.logging.warning(
                #         f"entry={entry}, molecule='{molecule}' uses disallowed reaction for this temporary window (only 4 or 5 allowed)"
                #     )
                #     valid_smiles = []
                #     valid_names = []
                #     break
                
                smiles = get_smiles(molecule)
                if not smiles:
                    bt.logging.error(f"No valid SMILES found for entry={entry}, molecule='{molecule}'")
                    valid_smiles = []
                    valid_names = []
                    break
                
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        bt.logging.error(f"Molecule is not parseable by RDKit for entry={entry}, molecule='{molecule}'")
                        valid_smiles = []
                        valid_names = []
                        break
                    failures = check_filters(mol, filter_thresholds)
                    if failures:
                        bt.logging.warning(f"entry={entry}, molecule='{molecule}' rejected: {failures[0]}")
                        valid_smiles = []
                        valid_names = []
                        break
                except Exception as e:
                    bt.logging.error(f"Molecule is not parseable by RDKit for entry={entry}, molecule='{molecule}': {e}")
                    valid_smiles = []
                    valid_names = []
                    break
     
                valid_smiles.append(smiles)
                valid_names.append(molecule)
            except Exception as e:
                bt.logging.error(f"Error validating molecule for entry={entry}, molecule='{molecule}': {e}")
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
                    
                    bt.logging.warning(f"entry={entry} submission contains chemically identical molecules: {'; '.join(duplicate_names)}")
                    score_dict[entry]["entropy"] = None
                    continue 
            except Exception as e:
                bt.logging.warning(f"Error checking for chemically identical molecules for entry={entry}: {e}")

        # Pairwise diversity check: reject if any two molecules are too similar.
        if valid_smiles:
            tanimoto_max_threshold = config.get("tanimoto_max_threshold", 1.0)
            try:
                too_similar = find_too_similar_pairs(valid_smiles, tanimoto_max_threshold)
                if too_similar:
                    examples = "; ".join(
                        f"{valid_names[i]}~{valid_names[j]} (Tanimoto={s:.3f})"
                        for i, j, s in too_similar[:3]
                    )
                    bt.logging.warning(
                        f"entry={entry} submission rejected: {len(too_similar)} molecule pair(s) "
                        f"exceed Tanimoto threshold {tanimoto_max_threshold}. Examples: {examples}"
                    )
                    score_dict[entry]["entropy"] = None
                    continue
            except Exception as e:
                bt.logging.warning(f"Error running pairwise diversity check for entry={entry}: {e}")

        
        # Calculate entropy if we have valid molecules, or skip if below threshold
        if valid_smiles:
            try:
                entropy = compute_fingerprint_entropy(valid_smiles)
                if entropy > config['entropy_min_threshold']:
                    score_dict[entry]["entropy"] = entropy
                    valid_molecules_by_entry[entry] = {"smiles": valid_smiles, "names": valid_names}
                else:
                    bt.logging.warning(f"entry={entry} submission has entropy below threshold: {entropy}")
                    score_dict[entry]["entropy"] = None
                    valid_smiles = []
                    valid_names = []

            except Exception as e:
                bt.logging.error(f"Error calculating entropy for entry={entry}: {e}")
                score_dict[entry]["entropy"] = None
                valid_smiles = []
                valid_names = []
        else:
            score_dict[entry]["entropy"] = None
            
    return valid_molecules_by_entry
