import os
import math
import json
import bittensor as bt
import pandas as pd
import numpy as np

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def _build_benchmark_payload(
    uid_to_data: dict,
    score_dict: dict,
    scored_sample_path: str = os.path.join(BASE_DIR, "all_scores_0.json"),
) -> dict or None:
    benchmark_uid = -1
    benchmark_name = "random_sample"
    ps_final_score = score_dict.get(benchmark_uid, {}).get("ps_final_score", -math.inf)
    ps_final_score_safe = ps_final_score

    if not os.path.exists(scored_sample_path):
        bt.logging.error(f"Scored sample path {scored_sample_path} does not exist")
        return None

    with open(scored_sample_path, "r") as f:
        scored_sample = json.load(f)
    scored_sample = scored_sample["scored_molecules"]
    df = pd.DataFrame(
        {
            "name": [score[0] for score in scored_sample],
            "score": [score[1] for score in scored_sample],
        }
    )

    curve = {
        "mean": df["score"].mean(),
        "stdv": df["score"].std(),
        "histogram": {
            "bounds": [
                -10,
                10,
            ],  # keep hardcoded for now, probably won't need to change until different scoring system
            "frequencies": np.histogram(df["score"], bins=200, range=(-10, 10))[
                0
            ].tolist(),
        },
    }

    benchmark = {
        "name": benchmark_name,
        "github_data": uid_to_data.get(benchmark_uid, {}).get("github_data", None),
        "ps_final_score": ps_final_score_safe,
        "curve": curve,
    }
    return benchmark


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

        # Build benchmark payload if available
        benchmark = _build_benchmark_payload(
            uid_to_data,
            score_dict,
            scored_sample_path,
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
        }

        # Add benchmark if available
        if benchmark:
            payload["benchmark"] = benchmark

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
