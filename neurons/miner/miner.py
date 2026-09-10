import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import sys
import json
import traceback
import time

import logging
import pandas as pd
from rdkit import Chem
from pathlib import Path
import nova_miner

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(levelname)s %(name)s %(message)s")
log = logging.getLogger("miner")

from nova_miner.utils.oracle import Oracle, combine
from nova_miner.utils.molecules import get_heavy_atom_count
from random_sampler import run_sampler
from nova_miner.combinatorial_db.reactions import get_smiles_from_reaction

DB_PATH = str(Path(nova_miner.__file__).resolve().parent / "combinatorial_db" / "molecules.sqlite")

ORACLE = Oracle(os.environ["ORACLE_SOCKET"])
MAX_PREDICTIONS = 480   # per request, counting (molecule, target) pairs.
# The oracle round-robins a request across its 24 shards and waits for the
# slowest, so a request costs ceil(predictions / 24) rounds. 480 is a multiple
# of 24 and under the server's 512 cap, and leaves room for a whole batch in
# one request so it is not split into a full chunk plus a short one.


def score_against_proteins(smiles: list[str], proteins: list[str]) -> list[list[float]]:
    """-> one combined score per (molecule, protein), molecules in order."""
    unique = list(dict.fromkeys(smiles))
    per_request = max(1, MAX_PREDICTIONS // len(proteins))
    rows = []
    for start in range(0, len(unique), per_request):
        rows.extend(ORACLE.score(proteins, unique[start:start + per_request]))

    heavy = {s: get_heavy_atom_count(s) for s in unique}
    scored = {s: [combine(m, heavy[s]) for m in row["scores"]]
              for s, row in zip(unique, rows)}
    return [scored[s] for s in smiles]

def get_config(input_file: os.path = os.path.join(BASE_DIR, "input.json")):
    """
    Get config from input file
    """
    with open(input_file, "r") as f:
        d = json.load(f)
    config = {**d.get("config", {}), **d.get("challenge", {})}
    return config

def iterative_sampling_loop(
    db_path: str,
    sampler_file_path: str,
    output_path: str,
    config: dict,
    save_all_scores: bool = False
) -> None:
    """
    Infinite loop, runs until orchestrator kills it:
      1) Sample n molecules
      2) Score them
      3) Merge with previous top x, deduplicate, sort, select top x
      4) Write top x to file (overwrite) each iteration
    """
    # 20% over the submission size: headroom for dedup loss so the first iteration
    # already writes a full submission, and a run cut short loses one batch not five.
    n_samples = int(config["num_molecules"] * 1.2)

    top_pool = pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])

    iteration = 0
    while True:
        iteration += 1
        log.info(f"[Miner] Iteration {iteration}: sampling {n_samples} molecules")

        sampler_data = run_sampler(n_samples=n_samples, 
                        subnet_config=config, 
                        output_path=sampler_file_path,
                        save_to_file=True,
                        db_path=db_path,
                        )
        
        if not sampler_data:
            log.warning("[Miner] No valid molecules produced; continuing")
            continue

        batch_scores = calculate_final_scores(sampler_data, config, save_all_scores)

        # Merge, deduplicate, sort and take top x
        top_pool = pd.concat([top_pool, batch_scores])
        top_pool = top_pool.drop_duplicates(subset=["InChIKey"], keep="first")
        top_pool = top_pool.sort_values(by="score", ascending=False)
        top_pool = top_pool.head(config["num_molecules"])

        # format to accepted format
        top_entries = {"molecules": top_pool["name"].tolist()}

        # write to file
        with open(output_path, "w") as f:
            json.dump(top_entries, f, ensure_ascii=False, indent=2)

        log.info(f"[Miner] Wrote {config['num_molecules']} top molecules to {output_path}")
        log.info(f"[Miner] Average score: {top_pool['score'].mean()}")

def calculate_final_scores(sampler_data: dict,
        config: dict,
        save_all_scores: bool = True,
        current_epoch: int = 0) -> pd.DataFrame:
    """
    Calculate final scores per molecule
    """

    names = sampler_data["molecules"]
    smiles = [get_smiles_from_reaction(name) for name in names]

    # Calculate InChIKey for each molecule to deduplicate molecules after merging
    inchikey_list = []
    
    for s in smiles:
        try:
            inchikey_list.append(Chem.MolToInchiKey(Chem.MolFromSmiles(s)))
        except Exception as e:
            log.error(f"Error calculating InChIKey for {s}: {e}")
            inchikey_list.append(None)

    targets = config["target_sequences"]
    antitargets = config["antitarget_sequences"]
    per_molecule = score_against_proteins(smiles, targets + antitargets)

    final_scores = []
    for values in per_molecule:
        target_values = values[:len(targets)]
        antitarget_values = values[len(targets):]
        avg_target = sum(target_values) / len(target_values)
        avg_antitarget = (sum(antitarget_values) / len(antitarget_values)
                          if antitarget_values else 0.0)
        final_scores.append(avg_target - config["antitarget_weight"] * avg_antitarget)

    # Store final scores in dataframe
    batch_scores = pd.DataFrame({
        "name": names,
        "smiles": smiles,
        "InChIKey": inchikey_list,
        "score": final_scores
    })

    if save_all_scores:
        all_scores = {"scored_molecules": [(mol["name"], mol["score"]) for mol in batch_scores.to_dict(orient="records")]}
        all_scores_path = os.path.join(OUTPUT_DIR, f"all_scores_{current_epoch}.json")
        if os.path.exists(all_scores_path):
            with open(all_scores_path, "r") as f:
                all_previous_scores = json.load(f)
            all_scores["scored_molecules"] = all_previous_scores["scored_molecules"] + all_scores["scored_molecules"]
        with open(all_scores_path, "w") as f:
            json.dump(all_scores, f, ensure_ascii=False, indent=2)

    return batch_scores

def main(config: dict):
    iterative_sampling_loop(
        db_path=DB_PATH,
        sampler_file_path=os.path.join(OUTPUT_DIR, "sampler_file.json"),
        output_path=os.path.join(OUTPUT_DIR, "result.json"),
        config=config,
        save_all_scores=True,
    )
 

if __name__ == "__main__":
    config = get_config()
    main(config)
