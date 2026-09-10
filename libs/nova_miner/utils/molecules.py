import logging
import math

import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys

from ..combinatorial_db.reactions import get_smiles_from_reaction

log = logging.getLogger(__name__)


def get_smiles(product_name: str) -> str | None:
    """SMILES for a combinatorial molecule name, e.g. ``rxn:1:45499:31805``."""
    if not product_name:
        log.error("Product name is empty.")
        return None
    product_name = product_name.replace("'", "").replace('"', "")
    if not product_name.startswith("rxn:"):
        log.error("Not a combinatorial molecule name: %s", product_name)
        return None
    return get_smiles_from_reaction(product_name)


def get_heavy_atom_count(smiles: str) -> int:
    """
    Number of heavy atoms (atomic number > 1) in a SMILES string.

    Uses RDKit so aromatic lowercase atoms, bracketed elements, and Cl/Br
    match ``Mol.GetNumHeavyAtoms()``. Returns 0 if SMILES is empty or
    unparseable.
    """
    if not smiles:
        return 0
    mol = Chem.MolFromSmiles(smiles)
    return get_heavy_atom_count_from_mol(mol)


def get_heavy_atom_count_from_mol(mol) -> int:
    """Heavy-atom count from an already-parsed RDKit mol. 0 if mol is None."""
    if mol is None:
        return 0
    return mol.GetNumHeavyAtoms()


def compute_maccs_entropy(smiles_list: list[str]) -> float:
    """
    Computes fingerprint entropy from MACCS keys for a list of SMILES.

    Parameters:
        smiles_list (list of str): Molecules in SMILES format.

    Returns:
        avg_entropy (float): Average entropy per bit.
    """
    n_bits = 167  # RDKit uses 167 bits (index 0 is always 0)
    bit_counts = np.zeros(n_bits)
    valid_mols = 0

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = MACCSkeys.GenMACCSKeys(mol)
            arr = np.array(fp)
            bit_counts += arr
            valid_mols += 1

    if valid_mols == 0:
        raise ValueError("No valid molecules found.")

    probs = bit_counts / valid_mols
    entropy_per_bit = np.array([
        -p * math.log2(p) - (1 - p) * math.log2(1 - p) if 0 < p < 1 else 0
        for p in probs
    ])

    avg_entropy = np.mean(entropy_per_bit)

    return avg_entropy


def find_chemically_identical(smiles_list: list[str]) -> dict:
    """
    Check for identical molecules in a list of SMILES strings by converting to InChIKeys.
    """
    inchikey_to_indices = {}
    
    for i, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                inchikey = Chem.MolToInchiKey(mol)
                if inchikey not in inchikey_to_indices:
                    inchikey_to_indices[inchikey] = []
                inchikey_to_indices[inchikey].append(i)
        except Exception as e:
            log.warning(f"Error processing SMILES {smiles}: {e}")
    
    duplicates = {k: v for k, v in inchikey_to_indices.items() if len(v) > 1}
    
    return duplicates
