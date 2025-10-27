import os
import math
import json
from typing import List, Dict, Any, Iterable, Tuple
import aiohttp
import bittensor as bt
import psycopg2
from psycopg2.extras import execute_values, Json
from psycopg2 import sql
import pandas as pd
import numpy as np

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def _connect_to_database() -> tuple[psycopg2.extensions.connection, psycopg2.extensions.cursor]:
    if not os.getenv('SUBMIT_RESULTS_URL'):
        bt.logging.error("SUBMIT_RESULTS_URL env variable not set; skipping submission.")
        return None, None
    conn = psycopg2.connect(os.getenv('SUBMIT_RESULTS_URL'))
    cursor = conn.cursor()
    return conn, cursor

def _insert_one(table_name: str, payload: dict) -> None:
    try:
        conn, cursor = _connect_to_database()
        columns = list(payload.keys())
        values = list(payload.values())
        cursor.execute(f'insert into {table_name} ({", ".join(columns)}) VALUES ({", ".join(["%s"] * len(columns))}) RETURNING id;', values)
        new_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
        return new_id
    except Exception as e:
        bt.logging.error(f"Error submitting payload {payload} to {table_name}: {e}")
        return None

def _insert_many(table_name: str, payloads: List[Dict[str, Any]]) -> List[int] | None:
    try:
        if not payloads:
            return []

        conn, cursor = _connect_to_database()

        # Ensure same columns for every payload and preserve order
        first_cols = list(payloads[0].keys())
        for p in payloads[1:]:
            if list(p.keys()) != first_cols:
                bt.logging.error("Columns of all payloads must be the same (and in the same order) for bulk insert.")
                cursor.close(); conn.close()
                return None

        # Build row tuples (NOT a flattened list!)
        rows: List[Tuple[Any, ...]] = [
            tuple(_pg_coerce(p[c]) for c in first_cols)
            for p in payloads
        ]

        query = sql.SQL("INSERT INTO {tbl} ({cols}) VALUES %s RETURNING id").format(
            tbl=sql.Identifier(table_name),
            cols=sql.SQL(", ").join(map(sql.Identifier, first_cols)),
        )

        # execute_values returns fetched rows when fetch=True
        returned = execute_values(
            cursor,
            query.as_string(cursor),
            rows,
            template=None,   # default "(%s)" repeated automatically
            fetch=True
        ) or []

        new_ids = [r[0] for r in returned]
        conn.commit()
        cursor.close()
        conn.close()
        return new_ids

    except Exception as e:
        bt.logging.error(f"Error submitting payloads {payloads} to {table_name}: {e}")
        return None

def _upsert_neurons_get_ids(neurons: list[dict]) -> list[tuple[int, str, str, int]]:
    """
    rows: iterable of (uid, coldkey, hotkey)
    returns: list of (uid, hotkey, coldkey, id) in the same order as input
    """
    conn, cursor = _connect_to_database()
    data = [(n['uid'], n['coldkey'], n['hotkey'], i) for i, n in enumerate(neurons)]
    if not data:
        return []

    sql = """
    WITH input(uid, coldkey, hotkey, ord) AS (
      VALUES %s
    ),
    ins AS (
      INSERT INTO public.neurons (uid, coldkey, hotkey)
      SELECT uid, coldkey, hotkey FROM input
      ON CONFLICT (uid, hotkey) DO NOTHING
      RETURNING id, uid, hotkey
    )
    SELECT i.uid, i.hotkey, i.coldkey, n.id
    FROM input i
    JOIN public.neurons n
      ON n.uid = i.uid AND n.hotkey = i.hotkey
    ORDER BY i.ord;
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, data, template="(%s,%s,%s,%s)")
        out = cur.fetchall()  # [(uid, hotkey, coldkey, id), ...] aligned to input order
    conn.commit()
    return out

def _pg_coerce(x: any) -> any:
    """Coerce numpy scalars and weird types to plain Python types for psycopg2."""
    try:
        # catches numpy scalar types: np.float64, np.int64, etc.
        if isinstance(x, np.generic):
            return x.item()
    except Exception:
        pass
    # simple fallbacks
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def _safe_num(x: float) -> float:
    return -999.99 if x == -math.inf else x

def _build_competition_payload(config, epoch: int, target_proteins: list[str], antitarget_proteins: list[str]) -> dict:
    return {
        "epoch": epoch,
        # start_block/end_block intentionally omitted; period is the sole timeline key
        "target_proteins": target_proteins,
        "antitarget_proteins": antitarget_proteins,
        "antitarget_weight": getattr(config, 'antitarget_weight', 1.0),
        "min_heavy_atoms": getattr(config, 'min_heavy_atoms', 0),
        "num_molecules": getattr(config, 'num_molecules', 0),
        "min_rotatable_bonds": getattr(config, 'min_rotatable_bonds', 0),
        "max_rotatable_bonds": getattr(config, 'max_rotatable_bonds', 0),
        "entropy_threshold": getattr(config, 'entropy_threshold', 0.0),
    }


def _build_submissions_payload(
        neurons_ids: list[tuple[int, str, str, int]],
        uid_to_data: dict, 
        valid_molecules_by_uid: dict, 
        score_dict: dict,
        competition_id: int,
        ) -> list[dict]:

    submissions = []
    for uid, data in uid_to_data.items():
        valid = valid_molecules_by_uid.get(uid, {})
        smiles_list = valid.get('smiles', [])
        names_list = valid.get('names', [])
        if not smiles_list:
            continue

        neuron_id = [n[3] for n in neurons_ids if n[0] == uid]
        if len(neuron_id) == 0:
            bt.logging.warning(f"Neuron ID not found for UID={uid}")
            continue

        entropy = score_dict.get(uid, {}).get('entropy')
        ps_final_score = score_dict.get(uid, {}).get('ps_final_score', -math.inf)
        ps_final_score_safe = _safe_num(ps_final_score)

        submissions.append({
            "neuron_id": neuron_id[0],
            "competition_id": competition_id,
            "github_data": data.get("github_data", None),
            "ps_final_score": ps_final_score_safe,
            "entropy": _safe_num(entropy) if entropy is not None else 0.0,
            "code_link": None, # TODO: add code link
        })

    return submissions

def _build_molecule_payload(
        valid_molecules_by_uid: dict, 
        score_dict: dict,
        uid: int,
        submission_id: int,
        ) -> list[dict]:

    molecules = []
    valid = valid_molecules_by_uid.get(uid, {})
    smiles_list = valid.get('smiles', [])
    names_list = valid.get('names', [])
    if not smiles_list:
        return None

    targets = score_dict.get(uid, {}).get('ps_target_scores', [])
    antitargets = score_dict.get(uid, {}).get('ps_antitarget_scores', [])
    combined_scores = score_dict.get(uid, {}).get('ps_combined_molecule_scores', [])

    for idx in range(len(smiles_list)):

        molecules.append({
            "submission_id": submission_id,
            "name": names_list[idx],
            "smiles": smiles_list[idx],
            "ps_target_scores": [ _safe_num(score) for score in ([t[idx] for t in targets] if targets else []) ],
            "ps_antitarget_scores": [ _safe_num(score) for score in ([a[idx] for a in antitargets] if antitargets else []) ],
            "ps_final_score": _safe_num(combined_scores[idx] if idx < len(combined_scores) else -math.inf),
        })

    return molecules

def _build_benchmark_payload(
        uid_to_data: dict,  
        score_dict: dict,
        competition_id: int,
        scored_sample_path: str = os.path.join(BASE_DIR, "all_scores_benchmark.json"),
        ) -> dict or None:

    benchmark_uid = 0
    benchmark_name = "random_sample"
    ps_final_score = score_dict.get(benchmark_uid, {}).get("ps_final_score", -math.inf)
    ps_final_score_safe = _safe_num(ps_final_score)

    if not os.path.exists(scored_sample_path):
        bt.logging.error(f"Scored sample path {scored_sample_path} does not exist")
        return None
    
    with open(scored_sample_path, "r") as f:
        scored_sample = json.load(f)
    scored_sample = scored_sample["scored_molecules"]
    df = pd.DataFrame({'name': [score[0] for score in scored_sample], 'score': [score[1] for score in scored_sample]})

    curve = {"mean": df['score'].mean(), 
            "stdv": df['score'].std(), 
            "histogram":
                {"bounds": [0, 10],  # keep hardcoded for now, probably won't need to change until different scoring system
                "frequencies": np.histogram(df['score'], bins=100, range=(0, 10))[0].tolist()}
                }
    
    benchmark = {
            "competition_id": competition_id,
            "name": benchmark_name,
            "github_data": Json(uid_to_data.get(benchmark_uid, {}).get("github_data", None)),
            "ps_final_score": _pg_coerce(ps_final_score_safe),
            "curve": Json(curve),
        }
    return benchmark

def _build_neurons_payload(
    uid_to_data: dict,
    score_dict: dict,
    ) -> list[dict]:

    neurons = []
    for uid, _ in score_dict.items():
        neurons.append({
            "uid": uid,
            "hotkey": uid_to_data.get(uid, {}).get("hotkey", "unknown"),
            "coldkey": uid_to_data.get(uid, {}).get("coldkey", "unknown"),
        })
    return neurons

def submit_epoch_results(
    config,
    epoch: int,
    target_proteins: list[str],
    antitarget_proteins: list[str],
    uid_to_data: dict,
    valid_molecules_by_uid: dict,
    score_dict: dict,
    scored_sample_path: str = os.path.join(BASE_DIR, "all_scores_benchmark.json"),
    ) -> bool:

    try:
        competition_payload = _build_competition_payload(config, epoch, target_proteins, antitarget_proteins)
        competition_id = _insert_one('competitions', competition_payload)
        bt.logging.debug(f"Competition ID: {competition_id}")

        neurons = _build_neurons_payload(uid_to_data, score_dict)
        neurons_ids = _upsert_neurons_get_ids(neurons)
        bt.logging.debug(f"Neurons IDs: {neurons_ids}")

        submissions = _build_submissions_payload(neurons_ids, uid_to_data, valid_molecules_by_uid, score_dict, competition_id)
        sub_ids = _insert_many('submissions', submissions)
        bt.logging.debug(f"Submissions IDs: {sub_ids}")

        for sub_id, uid in zip(sub_ids, score_dict.keys()):
            molecules = _build_molecule_payload(valid_molecules_by_uid, score_dict, uid, sub_id)
            molecule_ids = _insert_many('molecules', molecules)
            bt.logging.debug(f"Molecules IDs: {molecule_ids}")

        benchmark = _build_benchmark_payload(uid_to_data, score_dict, competition_id, scored_sample_path)
        benchmark_id = _insert_one('benchmark', benchmark)
        bt.logging.debug(f"Benchmark ID: {benchmark_id}")
        return True
    except Exception as e:
        bt.logging.error(f"Error submitting epoch results: {e}")
        return False

# if __name__ == "__main__":
#     neuron_data = fetch_all('benchmark')
#     print(neuron_data)
#     print(type(neuron_data))




