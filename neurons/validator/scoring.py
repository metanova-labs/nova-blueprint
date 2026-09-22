"""Boltz-2 molecular scoring, via the oracle."""

import math
import os
import json
from typing import List, Dict
import asyncio

import bittensor as bt

import sys
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
NOVA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
if NOVA_DIR not in sys.path:
    sys.path.append(NOVA_DIR)
LIBS_DIR = os.path.join(NOVA_DIR, "libs")
if LIBS_DIR not in sys.path:
    sys.path.append(LIBS_DIR)
from nova_miner.utils.oracle import FORMULA, combine

from utils.proteins import get_code_from_protein_sequence
from utils.molecules import get_heavy_atom_count
from neurons.validator.validity import validate_molecules_and_calculate_entropy
from sandbox.broker import score as oracle_score, MAX_PREDICTIONS

SCORING_MODEL = "boltz2"
from neurons.validator.ranking import calculate_final_scores
from neurons.validator.contest import apply_contest_transition
from neurons.validator.save_data import submit_epoch_results

def _build_thompson_benchmark_payload(
    *,
    epoch_number: int,
    config: dict,
    current_epoch: int,
    target_sequences: list[str],
    antitarget_sequences: list[str],
) -> list[dict]:

    try:
        jsonl_path = os.path.join(
            "/data/results", f"period_{int(epoch_number)}_results.jsonl"
        )
        if not os.path.exists(jsonl_path):
            return []

        # Pull TS molecules directly from the cumulative JSONL results file (uid=-2).
        molecules: list[str] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                try:
                    uid = int(rec.get("uid"))
                except Exception:
                    continue
                if uid != -2:
                    continue
                res = rec.get("result") or {}
                if isinstance(res, dict):
                    mols = res.get("molecules", [])
                    if isinstance(mols, list):
                        molecules = mols  # keep last occurrence if duplicated

        if not isinstance(molecules, list) or not molecules:
            return []

        # Use a negative UID that is never part of miner ranking/submissions.
        uid = -2
        github_data = None
        bench_entries = {uid: {"molecules": molecules, "raw": github_data}}
        bench_score_dict = {
            uid: {
                "target_scores": [[] for _ in range(len(config.get("target_codes", [])))],
                "antitarget_scores": [[] for _ in range(len(config.get("antitarget_codes", [])))],
                "target_metrics": [[] for _ in range(len(config.get("target_codes", [])))],
                "antitarget_metrics": [[] for _ in range(len(config.get("antitarget_codes", [])))],
                "entropy": None,
                "github_data": github_data,
            }
        }

        bench_valid = validate_molecules_and_calculate_entropy(
            entries_by_id=bench_entries,
            score_dict=bench_score_dict,
            config=config,
            allowed_reaction=config.get("allowed_reaction"),
        )
        if uid not in bench_valid:
            return []

        score_all_proteins_oracle(
            target_proteins=target_sequences,
            antitarget_proteins=antitarget_sequences,
            score_dict=bench_score_dict,
            valid_molecules_by_entry=bench_valid,
            entries_by_id=bench_entries,
            epoch=str(current_epoch),
        )
        bench_score_dict = calculate_final_scores(
            bench_score_dict, bench_valid, config, current_epoch
        )

        names = bench_valid[uid].get("names", [])
        combined = bench_score_dict.get(uid, {}).get("combined_molecule_scores", [])
        n = min(len(names), len(combined))
        scored_molecules = [[str(names[j]), float(combined[j])] for j in range(n)]
        if not scored_molecules:
            return []
        return [{"name": "thompson_sampling", "github_data": None, "scored_molecules": scored_molecules}]
    except Exception as e:
        bt.logging.warning(
            f"failed to build thompson_sampling benchmark payload: {type(e).__name__}: {e}"
        )
        return []


async def process_epoch(
    config,
    epoch_number: int,
    entries: dict,
    state: dict,
    scored_sample_path: str,
):
    """
    Score every entry (keyed by entry_id) and resolve the champion contest.

    Returns (new_state, champion_entry_id, champion_score).
    """
    try:
        current_epoch = epoch_number

        target_sequences = config["target_sequences"]
        antitarget_sequences = config["antitarget_sequences"]
        allowed_reaction = config.get("allowed_reaction")

        target_codes = [get_code_from_protein_sequence(sequence) for sequence in target_sequences]
        antitarget_codes = [get_code_from_protein_sequence(sequence) for sequence in antitarget_sequences]

        config["target_codes"] = target_codes
        config["antitarget_codes"] = antitarget_codes

        if allowed_reaction:
            bt.logging.info(f"Allowed reaction this epoch: {allowed_reaction}")

        bt.logging.info(f"Scoring using target proteins: {target_codes}, antitarget proteins: {antitarget_codes}")

        if not entries:
            bt.logging.info("No valid submissions found this epoch.")
            return state, None, None

        # Initialize scoring structure
        score_dict = {
            eid: {
                "target_scores": [[] for _ in range(len(target_codes))],
                "antitarget_scores": [[] for _ in range(len(antitarget_codes))],
                "target_metrics": [[] for _ in range(len(target_codes))],
                "antitarget_metrics": [[] for _ in range(len(antitarget_codes))],
                "entropy": None,
                "github_data": entries[eid].get("github_data"),
            }
            for eid in entries
        }

        # Validate molecules and calculate entropy
        valid_molecules_by_entry = validate_molecules_and_calculate_entropy(
            entries_by_id=entries,
            score_dict=score_dict,
            config=config,
            allowed_reaction=allowed_reaction,
        )

        score_all_proteins_oracle(
            target_proteins=target_sequences,
            antitarget_proteins=antitarget_sequences,
            score_dict=score_dict,
            valid_molecules_by_entry=valid_molecules_by_entry,
            entries_by_id=entries,
            epoch=str(current_epoch),
        )

        # Calculate final scores
        score_dict = calculate_final_scores(
            score_dict, valid_molecules_by_entry, config, current_epoch
        )

        # The margin a challenger must clear to beat the champion (stored on the competition row).
        config["threshold_to_win"] = float(config["improvement_margin"])

        # Resolve the champion contest
        new_state, champion_entry_id, champion_score = apply_contest_transition(
            score_dict=score_dict,
            entries=entries,
            state=state,
            cfg=config,
            epoch=epoch_number,
        )

        # Yield so ws heartbeats can run before the next RPC
        await asyncio.sleep(0)

        # Submit results to dashboard API if configured
        try:
            if os.environ.get("BACKEND_API_URL"):
                benchmarks_payload = _build_thompson_benchmark_payload(
                    epoch_number=epoch_number,
                    config=config,
                    current_epoch=current_epoch,
                    target_sequences=target_sequences,
                    antitarget_sequences=antitarget_sequences,
                )
                status = await submit_epoch_results(
                    config=config,
                    epoch_number=epoch_number,
                    target_proteins=target_codes,
                    antitarget_proteins=antitarget_codes,
                    scoring_model=SCORING_MODEL,
                    scoring_formula=FORMULA,
                    entries=entries,
                    valid_molecules_by_entry=valid_molecules_by_entry,
                    score_dict=score_dict,
                    state=new_state,
                    scored_sample_path=scored_sample_path,
                    benchmarks=benchmarks_payload,
                )
                if status:
                    bt.logging.info("Submitted results to dashboard DB")
        except Exception as e:
            bt.logging.error(f"Failed to submit results to dashboard DB: {e}")

        return new_state, champion_entry_id, champion_score

    except Exception as e:
        bt.logging.error(f"Error processing epoch: {e}")
        return state, None, None

def score_all_proteins_oracle(
    target_proteins: list[str],
    antitarget_proteins: list[str],
    score_dict: dict,
    valid_molecules_by_entry: dict,
    entries_by_id: dict = None,
    epoch: str = "",
) -> None:
    """Score every valid molecule against every protein and fill score_dict.

    One oracle request covers all proteins for a chunk of molecules, and a molecule
    submitted by several entries is predicted once. Alongside the combined value
    each prediction's raw metrics are kept, so the score stays reproducible.
    Anything the oracle could not score is -inf, which ranking.py treats as failed.
    """
    all_proteins = target_proteins + antitarget_proteins
    if not all_proteins:
        return

    def _blank(entry) -> int:
        n = 0
        if entries_by_id:
            n = len(entries_by_id.get(entry, {}).get("molecules", []))
        for col in range(len(target_proteins)):
            score_dict[entry]["target_scores"][col] = [-math.inf] * n
            score_dict[entry]["target_metrics"][col] = [None] * n
        for col in range(len(antitarget_proteins)):
            score_dict[entry]["antitarget_scores"][col] = [-math.inf] * n
            score_dict[entry]["antitarget_metrics"][col] = [None] * n
        return n

    unique: dict[str, None] = {}
    for entry, valid in valid_molecules_by_entry.items():
        smiles_list = valid.get("smiles") or []
        if not smiles_list:
            _blank(entry)
            continue
        for smiles in smiles_list:
            unique[smiles] = None

    if not unique:
        bt.logging.warning("No valid molecules to score this epoch.")
        return

    failed_values = [-math.inf] * len(all_proteins)
    failed_metrics = [None] * len(all_proteins)
    values: dict[str, list[float]] = {}
    metrics: dict[str, list[dict | None]] = {}
    smiles_list = list(unique)
    per_request = max(1, MAX_PREDICTIONS // len(all_proteins))

    for start in range(0, len(smiles_list), per_request):
        chunk = smiles_list[start:start + per_request]
        try:
            rows = oracle_score(all_proteins, chunk, epoch)
        except Exception as e:
            bt.logging.error(f"Oracle scoring failed for {len(chunk)} molecules: {e}")
            for smiles in chunk:
                values[smiles] = failed_values
                metrics[smiles] = failed_metrics
            continue
        for smiles, row in zip(chunk, rows):
            heavy = get_heavy_atom_count(smiles)
            values[smiles] = [combine(m, heavy) for m in row["scores"]]
            metrics[smiles] = list(row["scores"])

    for entry, valid in valid_molecules_by_entry.items():
        smiles_list = valid.get("smiles") or []
        if not smiles_list:
            continue
        for protein_idx in range(len(all_proteins)):
            value_col = [values.get(s, failed_values)[protein_idx] for s in smiles_list]
            metric_col = [metrics.get(s, failed_metrics)[protein_idx] for s in smiles_list]
            if protein_idx < len(target_proteins):
                score_dict[entry]["target_scores"][protein_idx] = value_col
                score_dict[entry]["target_metrics"][protein_idx] = metric_col
            else:
                col = protein_idx - len(target_proteins)
                score_dict[entry]["antitarget_scores"][col] = value_col
                score_dict[entry]["antitarget_metrics"][col] = metric_col

    bt.logging.info(
        f"Scored {len(unique)} unique molecules against {len(all_proteins)} proteins")
