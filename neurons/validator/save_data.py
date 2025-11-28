import os
import json
import bittensor as bt
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


async def submit_epoch_results(
    config: dict,
    epoch_number: int,
    target_proteins: list[str],
    antitarget_proteins: list[str],
    uid_to_data: dict,
    valid_molecules_by_uid: dict,
    score_dict: dict,
    scored_sample_path: str = os.path.join(BASE_DIR, "all_scores_0.json"),
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
        uid_to_data_str = {str(k): v for k, v in uid_to_data.items()}
        valid_molecules_str = {str(k): v for k, v in valid_molecules_by_uid.items()}

        # Build POST payload
        payload = {
            "epoch": epoch_number,
            "target_proteins": target_proteins,
            "antitarget_proteins": antitarget_proteins,
            "config": {
                "antitarget_weight": config.get("antitarget_weight", 1.0),
                "min_heavy_atoms": config.get("min_heavy_atoms", 0),
                "num_molecules": config.get("num_molecules", 0),
                "min_rotatable_bonds": config.get("min_rotatable_bonds", 0),
                "max_rotatable_bonds": config.get("max_rotatable_bonds", 0),
                "entropy_threshold": config.get("entropy_threshold", 0.0),
            },
            "score_dict": score_dict_str,
            "uid_to_data": uid_to_data_str,
            "valid_molecules_by_uid": valid_molecules_str,
            "scored_sample_data": scored_sample_data,
        }

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
