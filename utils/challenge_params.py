from typing import Dict, Any, List

from config.config_loader import load_config  
from utils.proteins import (
    get_challenge_params_from_blockhash,
    get_sequence_from_protein_code,
) 
from utils.reactions import get_total_reactions


def _extract_miner_config(cfg: dict) -> Dict[str, Any]:
    return {
        "antitarget_weight": cfg["antitarget_weight"],
        "entropy_min_threshold": cfg["entropy_min_threshold"],
        "min_heavy_atoms": cfg["min_heavy_atoms"],
        "min_rotatable_bonds": cfg["min_rotatable_bonds"],
        "max_rotatable_bonds": cfg["max_rotatable_bonds"],
        "num_molecules": cfg["num_molecules"],
    }


def build_challenge_params(block_hash: str) -> Dict[str, Any]:
    """
    Build the single challenge_params dict for one orchestrator run.
    - Loads YAML config
    - Derives target/antitarget codes from block hash
    - Fetches sequences only
    - Includes allowed_reaction if enabled
    """
    raw_cfg = load_config()
    miner_cfg = _extract_miner_config(raw_cfg)

    num_antitargets: int = int(raw_cfg["num_antitargets"])
    include_reaction: bool = bool(raw_cfg["random_valid_reaction"])

    params = get_challenge_params_from_blockhash(
        block_hash=block_hash,
        num_antitargets=num_antitargets,
        include_reaction=include_reaction,
    )

    target_code = params.get("target")
    antitarget_codes: List[str] = params.get("antitargets", [])  # type: ignore
    target_seq = get_sequence_from_protein_code(target_code) if isinstance(target_code, str) else None
    antitarget_seqs = [get_sequence_from_protein_code(code) for code in antitarget_codes]

    out: Dict[str, Any] = {
        "config": miner_cfg,
        "challenge": {
            "target_sequences": [target_seq],
            "antitarget_sequences": antitarget_seqs,
        },
    }
    allowed = params.get("allowed_reaction")
    if allowed:
        out["challenge"]["allowed_reaction"] = allowed
    return out


