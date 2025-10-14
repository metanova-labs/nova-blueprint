import os
import math
import json
import aiohttp
import bittensor as bt
import psycopg2

from dotenv import load_dotenv

load_dotenv('../../.env')


def _connect_to_database() -> tuple[psycopg2.extensions.connection, psycopg2.extensions.cursor]:
    if not os.getenv('DATABASE_URL'):
        bt.logging.error("DATABASE_URL env variable not set; skipping submission.")
        return None, None
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
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

def _insert_many(table_name: str, payloads: list[dict]) -> list[int]:
    try:
        conn, cursor = _connect_to_database()
        columns = [d.keys() for d in payloads]

        if not all(c == columns[0] for c in columns):
            bt.logging.error(f"Columns of all payloads must be the same for bulk insert.")
            return None

        columns = columns[0]
        values = [p[c] for c in columns for p in payloads]
        template = "(" + ", ".join(["%s"] * len(columns)) + ")"
        command = sql.SQL("INSERT INTO competitions ({cols}) VALUES %s RETURNING id").format(
            cols=sql.SQL(", ").join(map(sql.Identifier, columns))
            )
        new_ids = []
        execute_values(cursor, command.as_string(cursor), values, template=template, fetch=True)
        new_ids = [t[0] for t in cursor.fetchall()] if cursor.rowcount else []
        conn.commit()
        cursor.close()
        conn.close()
        return new_ids
    except Exception as e:
        bt.logging.error(f"Error submitting payloads {payloads} to {table_name}: {e}")
        return None


def _safe_num(x: float) -> float:
    return -999.99 if x == -math.inf else x

def _build_competition_payload(config, start_block: int, target_proteins: list[str], antitarget_proteins: list[str]) -> dict:
    epoch_length = getattr(config, 'epoch_length', 361)
    end_block = start_block + epoch_length
    epoch_number = (end_block // epoch_length) - 1
    return {
        "epoch": epoch_number,
        "start_block": start_block,
        "end_block": end_block,
        "target_proteins": target_proteins,
        "antitarget_proteins": antitarget_proteins,
        "antitarget_weight": getattr(config, 'antitarget_weight', 1.0),
        "min_heavy_atoms": getattr(config, 'min_heavy_atoms', 0),
        "num_molecules": getattr(config, 'num_molecules', 0),
        "min_rotatable_bonds": getattr(config, 'min_rotatable_bonds', 0),
        "max_rotatable_bonds": getattr(config, 'max_rotatable_bonds', 0),
        "entropy_threshold": getattr(config, 'entropy_threshold', 0.0),
    }


# TODO: change db schema and adapt this function
def _build_submissions_payload(
        config, 
        metagraph, 
        current_block: int, 
        start_block: int, 
        uid_to_data: dict, 
        valid_molecules_by_uid: dict, 
        score_dict: dict
        ) -> list[dict]:
    submissions = []
    for uid, data in uid_to_data.items():
        valid = valid_molecules_by_uid.get(uid, {})
        smiles_list = valid.get('smiles', [])
        names_list = valid.get('names', [])
        if not smiles_list:
            continue

        hotkey = data.get('hotkey') or (metagraph.hotkeys[uid] if uid < len(metagraph.hotkeys) else "unknown")
        coldkey = metagraph.coldkeys[uid] if hasattr(metagraph, 'coldkeys') and uid < len(metagraph.coldkeys) else "unknown"

        entropy = score_dict.get(uid, {}).get('entropy')
        ps_final_score = score_dict.get(uid, {}).get('ps_final_score', -math.inf)
        ps_final_score_safe = _safe_num(ps_final_score)

        targets = score_dict.get(uid, {}).get('target_scores', [])
        antitargets = score_dict.get(uid, {}).get('antitarget_scores', [])
        combined_scores = score_dict.get(uid, {}).get('ps_combined_molecule_scores', [])

        molecule_details = []
        for idx in range(len(smiles_list)):

            molecule_details.append({
                "name": names_list[idx],
                "smiles": smiles_list[idx],
                "target_scores": [ _safe_num(score) for score in ([t[idx] for t in targets] if targets else []) ],
                "antitarget_scores": [ _safe_num(score) for score in ([a[idx] for a in antitargets] if antitargets else []) ],
                "combined_score": _safe_num(combined_scores[idx] if idx < len(combined_scores) else -math.inf),
            })

        submissions.append({
            "neuron": {
                "uid": uid,
                "hotkey": hotkey,
                "coldkey": coldkey,
            },
            "blocks_elapsed": blocks_elapsed,
            "molecules": molecule_details,
            "entropy": (entropy if (entropy is not None and final_score_safe > getattr(config, 'entropy_bonus_threshold', 0.0)) else 0),
            "final_score": final_score_safe,
            "final_boltz_score": (None if boltz_score is None else float(boltz_score)),
            "boltz_entropy": (None if boltz_entropy is None else float(boltz_entropy)),
        })

    return submissions




