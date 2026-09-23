import os
import sys
import math

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetAtomPairGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity
import bittensor as bt
from combinatorial_db.reactions import get_smiles_from_reaction

FINGERPRINT_BITS = 2048
_ATOM_PAIR_GEN = GetAtomPairGenerator(fpSize=FINGERPRINT_BITS)


def get_smiles(product_name: str) -> str | None:
    """SMILES for a combinatorial molecule name, e.g. ``rxn:1:45499:31805``."""
    if not product_name:
        bt.logging.error("Product name is empty.")
        return None
    product_name = product_name.replace("'", "").replace('"', "")
    if not product_name.startswith("rxn:"):
        bt.logging.error(f"Not a combinatorial molecule name: {product_name}")
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


def compute_fingerprint_entropy(smiles_list: list[str]) -> float:
    """
    Computes fingerprint entropy from atom-pair fingerprints for a list of SMILES.

    Parameters:
        smiles_list (list of str): Molecules in SMILES format.

    Returns:
        avg_entropy (float): Average entropy per bit.
    """
    n_bits = FINGERPRINT_BITS
    bit_counts = np.zeros(n_bits)
    valid_mols = 0

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = _ATOM_PAIR_GEN.GetFingerprint(mol)
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
            bt.logging.warning(f"Error processing SMILES {smiles}: {e}")
    
    duplicates = {k: v for k, v in inchikey_to_indices.items() if len(v) > 1}
    
    return duplicates


def find_too_similar_pairs(
    smiles_list: list[str],
    threshold: float,
    radius: int = 2,
    n_bits: int = 2048,
) -> list[tuple[int, int, float]]:
    """
    Return all (i, j, similarity) pairs whose Morgan/ECFP4 Tanimoto similarity
    is >= ``threshold``. Pairs are reported with i < j.

    A submission is considered diverse iff this returns an empty list. Returning
    pairs (rather than a bool) lets callers log which molecules collided.

    Set ``threshold`` to 1.0 to effectively disable the check (only exact-FP
    duplicates would trigger, and those are already caught by the InChIKey check).
    """
    if threshold >= 1.0 + 1e-12 or len(smiles_list) < 2:
        return []

    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = []
    indices = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps.append(generator.GetFingerprint(mol))
        indices.append(i)

    offending: list[tuple[int, int, float]] = []
    for k in range(len(fps) - 1):
        sims = BulkTanimotoSimilarity(fps[k], fps[k + 1:])
        for off, s in enumerate(sims):
            if s >= threshold:
                offending.append((indices[k], indices[k + 1 + off], float(s)))
    return offending
