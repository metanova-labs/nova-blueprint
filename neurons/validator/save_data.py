import os
import json
from pathlib import Path

import bittensor as bt
from dotenv import load_dotenv

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


def _get_prev_winner_snapshot_epoch(prev_winner_uid):
    """
    Return the snapshot_epoch for the previous winner UID from winner.json,
    or None if not available/mismatched.
    """
    if prev_winner_uid is None:
        return None
    try:
        winner_json_path = Path("/data/results/winner.json")
        if not winner_json_path.exists():
            return None
        with winner_json_path.open("r", encoding="utf-8") as f:
            prev = json.load(f)
        prev_snapshot = prev["winner_snapshot"]
        prev_uid = int(prev_snapshot["uid"])
        prev_snapshot_epoch_val = prev_snapshot.get("snapshot_epoch")
        if prev_uid == int(prev_winner_uid) and prev_snapshot_epoch_val is not None:
            return int(prev_snapshot_epoch_val)
    except Exception as e:
        bt.logging.warning(f"Failed to read previous winner snapshot_epoch: {e}")
    return None


async def submit_epoch_results(
    config: dict,
    epoch_number: int,
    target_proteins: list[str],
    antitarget_proteins: list[str],
    uid_to_data: dict,
    valid_molecules_by_uid: dict,
    score_dict: dict,
    scored_sample_path: str = os.path.join(BASE_DIR, "all_scores_0.json"),
    winner_uid=None,
    benchmarks: list[dict] | None = None,
) -> bool:
    """
    Submit epoch results to backend API via POST request.
    """
    try:
        from utils.BackendAPI import BackendAPI

        # load entire scored sample JSON file for backend
        scored_sample_data = None
        if os.path.exists(scored_sample_path):
            try:
                with open(scored_sample_path, "r") as f:
                    scored_sample_data = json.load(f)
            except Exception as e:
                bt.logging.warning(
                    f"Failed to load scored sample data from {scored_sample_path}: {e}"
                )

        # Convert integer keys to strings for JSON
        score_dict_str = {str(k): v for k, v in score_dict.items()}

        # Enrich uid_to_data with deterministic code_link.
        prev_winner_uid = config.get("prev_winner_uid")
        prev_winner_snapshot_epoch = _get_prev_winner_snapshot_epoch(prev_winner_uid)

        uid_to_data_enriched = {}
        for uid, data in uid_to_data.items():
            d = dict(data) if isinstance(data, dict) else data
            try:
                epoch_for_uid = epoch_number
                if (
                    prev_winner_uid is not None
                    and int(uid) == int(prev_winner_uid)
                    and prev_winner_snapshot_epoch is not None
                ):
                    epoch_for_uid = prev_winner_snapshot_epoch

                d["code_link"] = f"{int(epoch_for_uid)}/{int(uid)}"
            except Exception as e:
                bt.logging.warning(f"Failed to resolve snapshot key for uid={uid}: {e}")
            d["submission_name"] = d.get("submission_name")
            uid_to_data_enriched[uid] = d

        uid_to_data_str = {str(k): v for k, v in uid_to_data_enriched.items()}
        valid_molecules_str = {str(k): v for k, v in valid_molecules_by_uid.items()}
        allowed_reaction = _normalize_allowed_reaction(config.get("allowed_reaction"))

        # Build POST payload
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
                "allowed_reaction": allowed_reaction,
            },
            "score_dict": score_dict_str,
            "uid_to_data": uid_to_data_str,
            "valid_molecules_by_uid": valid_molecules_str,
            "scored_sample_data": scored_sample_data,
            "winner_uid": winner_uid,
        }
        if benchmarks:
            payload["benchmarks"] = benchmarks

        # Send via BackendAPI
        api = BackendAPI()
        success = await api.submit_epoch_results(payload)

        if success:
            bt.logging.info(
                f"Successfully submitted epoch {epoch_number} results to backend API"
            )
        else:
            bt.logging.warning(
                f"Failed to submit epoch {epoch_number} results to backend API"
            )
        return success

    except Exception as e:
        bt.logging.error(f"Error submitting epoch results: {e}", exc_info=True)
        return False
