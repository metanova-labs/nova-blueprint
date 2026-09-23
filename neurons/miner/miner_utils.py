from rdkit import Chem

from nova_miner.utils.filters import check as check_filters, validate_thresholds
from nova_miner.utils import get_smiles


def validate_molecules_sampler(
    sampler_data: dict[str, list],
    config: dict,
) -> tuple[list[str], list[str]]:
    """Drop sampled molecules the validator would reject, before they cost oracle time.

    Mirrors the per-molecule checks the validator applies in
    validate_molecules_and_calculate_entropy. Submission-level rules -- molecule
    count, duplicates, allowed reaction, chemical identity, diversity and entropy --
    belong to random_sampler.py.

    Returns (names, smiles) for the molecules that passed.
    """
    filter_thresholds = validate_thresholds(config.get("filters"))
    valid_names: list[str] = []
    valid_smiles: list[str] = []

    for molecule in sampler_data["molecules"]:
        if not molecule:
            continue
        try:
            smiles = get_smiles(molecule)
            if not smiles:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            if check_filters(mol, filter_thresholds):
                continue
        except Exception:
            continue
        valid_names.append(molecule)
        valid_smiles.append(smiles)

    return valid_names, valid_smiles
