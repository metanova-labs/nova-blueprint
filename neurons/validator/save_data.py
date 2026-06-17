import math
import os
import json

import bittensor as bt
from dotenv import load_dotenv

from neurons.validator.contest import entry_id

load_dotenv()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def _normalize_allowed_reaction(value):
    """Convert validator reaction identifiers like 'rxn:4' to integer 4."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if value.startswith("rxn:"):
            try:
                return int(value.split(":")[-1])
            except (TypeError, ValueError):
                return None
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _json_float(value):
    """Coerce to a JSON-safe float, mapping non-finite values to None."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isinf(out) or math.isnan(out):
        return None
    return out


def _build_entry_molecules(valid_entry: dict | None, scores: dict) -> list[dict]:
    """Pre-zip per-molecule scores into the backend's Molecule shape."""
    names = (valid_entry or {}).get("names", []) or []
    smiles = (valid_entry or {}).get("smiles", []) or []
    targets = scores.get("ps_target_scores", []) or []
    antitargets = scores.get("ps_antitarget_scores", []) or []
    combined = scores.get("ps_combined_molecule_scores", []) or []

    molecules = []
    for j in range(min(len(names), len(smiles))):
        molecules.append(
            {
                "name": str(names[j]),
                "smiles": str(smiles[j]),
                "ps_target_scores": [_json_float(col[j]) for col in targets if j < len(col)],
                "ps_antitarget_scores": [_json_float(col[j]) for col in antitargets if j < len(col)],
                "ps_final_score": _json_float(combined[j]) if j < len(combined) else None,
            }
        )
    return molecules


async def submit_epoch_results(
    config: dict,
    epoch_number: int,
    target_proteins: list[str],
    antitarget_proteins: list[str],
    entries: dict,
    valid_molecules_by_entry: dict,
    score_dict: dict,
    state: dict,
    scored_sample_path: str = os.path.join(BASE_DIR, "all_scores_0.json"),
    benchmarks: list[dict] | None = None,
) -> bool:
    """Submit one epoch of results to the backend as an entries[] payload."""
    try:
        from utils.BackendAPI import BackendAPI

        scored_sample_data = None
        if os.path.exists(scored_sample_path):
            try:
                with open(scored_sample_path, "r") as f:
                    scored_sample_data = json.load(f)
            except Exception as e:
                bt.logging.warning(
                    f"Failed to load scored sample data from {scored_sample_path}: {e}"
                )

        champion = (state or {}).get("champion")
        champion_eid = entry_id(champion["hotkey"], champion["snapshot_epoch"]) if champion else None
        challenger_wins = {
            entry_id(c["hotkey"], c["snapshot_epoch"]): int(c.get("wins", 0))
            for c in (state or {}).get("challengers", [])
        }

        entries_payload = []
        for eid, entry in entries.items():
            scores = score_dict.get(eid, {})
            if eid == champion_eid:
                kind, wins = "champion", None
            elif eid in challenger_wins:
                kind, wins = "challenger", challenger_wins[eid]
            else:
                kind, wins = "entrant", None

            snapshot_epoch = int(entry["snapshot_epoch"])
            entries_payload.append(
                {
                    "entry_id": eid,
                    "kind": kind,
                    "wins": wins,
                    "hotkey": entry.get("hotkey"),
                    "uid": entry.get("uid"),
                    "coldkey": entry.get("coldkey"),
                    "snapshot_epoch": snapshot_epoch,
                    "code_link": f"{snapshot_epoch}/{entry.get('hotkey')}",
                    "submission_name": entry.get("submission_name"),
                    "github_data": entry.get("github_data"),
                    "entropy": _json_float(scores.get("entropy")),
                    "ps_final_score": _json_float(scores.get("ps_final_score")),
                    "molecules": _build_entry_molecules(valid_molecules_by_entry.get(eid), scores),
                }
            )

        allowed_reaction = _normalize_allowed_reaction(config.get("allowed_reaction"))

        payload = {
            "epoch": epoch_number,
            "target_proteins": target_proteins,
            "antitarget_proteins": antitarget_proteins,
            "config": {
                "antitarget_weight": config.get("antitarget_weight", 1.0),
                "min_rotatable_bonds": config.get("min_rotatable_bonds", 0),
                "max_rotatable_bonds": config.get("max_rotatable_bonds", 0),
                "min_heavy_atoms": config.get("min_heavy_atoms", 0),
                "num_molecules": config.get("num_molecules", 0),
                "entropy_threshold": config.get("entropy_min_threshold", 0.0),
                "time_budget": config.get("time_budget_sec", 0),
                "threshold_to_win": config.get("threshold_to_win", 0),
                "wins_required": int(config.get("wins_required", 1)),
                "allowed_reaction": allowed_reaction,
            },
            "entries": entries_payload,
            "scored_sample_data": scored_sample_data,
        }
        if benchmarks:
            payload["benchmarks"] = benchmarks

        api = BackendAPI()
        success = await api.submit_epoch_results(payload)

        if success:
            bt.logging.info(f"Successfully submitted epoch {epoch_number} results to backend API")
        else:
            bt.logging.warning(f"Failed to submit epoch {epoch_number} results to backend API")
        return success

    except Exception as e:
        bt.logging.error(f"Error submitting epoch results: {e}", exc_info=True)
        return False
