import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import sys
import json
import traceback
import time

import bittensor as bt
import pandas as pd
from rdkit import Chem
from pathlib import Path

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

from nova_ph2.neurons.validator.scoring import score_molecules_json
import nova_ph2.neurons.validator.scoring as scoring_module
from random_sampler import run_sampler
from nova_ph2.combinatorial_db.reactions import get_smiles_from_reaction

#DB_PATH = str(Path(nova_ph2.__file__).resolve().parent / "combinatorial_db" / "molecules.sqlite")
DB_PATH = str(Path(BASE_DIR).resolve().parent / "nova_ph2" / "combinatorial_db" / "molecules.sqlite")

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
    n_samples = config["num_molecules"] * 5

    top_pool = pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])

    iteration = 0
    while True:
        iteration += 1
        bt.logging.info(f"[Miner] Iteration {iteration}: sampling {n_samples} molecules")

        sampler_data = run_sampler(n_samples=n_samples, 
                        subnet_config=config, 
                        output_path=sampler_file_path,
                        save_to_file=True,
                        db_path=db_path,
                        )
        
        if not sampler_data:
            bt.logging.warning("[Miner] No valid molecules produced; continuing")
            continue

        score_dict = score_molecules_json(sampler_file_path, 
                                         config["target_sequences"], 
                                         config["antitarget_sequences"], 
                                         config)
        
        if not score_dict:
            bt.logging.warning("[Miner] Scoring failed or mismatched; continuing")
            continue

        # Calculate final scores per molecule
        batch_scores = calculate_final_scores(score_dict, sampler_data, config, save_all_scores)

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

        bt.logging.info(f"[Miner] Wrote {config['num_molecules']} top molecules to {output_path}")
        bt.logging.info(f"[Miner] Average score: {top_pool['score'].mean()}")

def calculate_final_scores(score_dict: dict, 
        sampler_data: dict, 
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
            bt.logging.error(f"Error calculating InChIKey for {s}: {e}")
            inchikey_list.append(None)

    # Calculate final scores for each molecule
    targets = score_dict[0]['target_scores']
    antitargets = score_dict[0]['antitarget_scores']
    final_scores = []
    for mol_idx in range(len(names)):
        # target average
        target_scores_for_mol = [target_list[mol_idx] for target_list in targets]
        avg_target = sum(target_scores_for_mol) / len(target_scores_for_mol)

        # antitarget average
        antitarget_scores_for_mol = [antitarget_list[mol_idx] for antitarget_list in antitargets]
        avg_antitarget = sum(antitarget_scores_for_mol) / len(antitarget_scores_for_mol)

        # final score
        score = avg_target - (config["antitarget_weight"] * avg_antitarget)
        final_scores.append(score)

    # Store final scores in dataframe
    batch_scores = pd.DataFrame({
        "name": names,
        "smiles": smiles,
        "InChIKey": inchikey_list,
        "score": final_scores
    })

    if save_all_scores:
        all_scores = {"scored_molecules": [(mol["name"], mol["score"]) for mol in batch_scores.to_dict(orient="records")]}
        
        if os.path.exists(os.path.join(BASE_DIR, f"all_scores_{current_epoch}.json")):
            with open(os.path.join(BASE_DIR, f"all_scores_{current_epoch}.json"), "r") as f:
                all_previous_scores = json.load(f)
            
            all_scores["scored_molecules"] = all_previous_scores["scored_molecules"] + all_scores["scored_molecules"]

        with open(os.path.join(BASE_DIR, f"all_scores_{current_epoch}.json"), "w") as f:
            json.dump(all_scores, f, ensure_ascii=False, indent=2)

    return batch_scores

def main(config: dict):
    iterative_sampling_loop(
        db_path=DB_PATH,
        sampler_file_path=os.path.join(BASE_DIR, "sampler_file.json"),
        output_path=os.path.join(BASE_DIR, "output.json"),
        config=config,
        save_all_scores=True,
    )
 

if __name__ == "__main__":
    config = get_config()
    main(config)
