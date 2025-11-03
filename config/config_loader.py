import yaml
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_config(path: str = os.path.join(BASE_DIR, "config/config.yaml")):
    """
    Loads configuration from a YAML file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find config file at '{path}'")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Load configuration options
    num_antitargets = config["protein_selection"]["num_antitargets"]
    
    validation_config = config["molecule_validation"]
    antitarget_weight = validation_config["antitarget_weight"]
    min_heavy_atoms = validation_config["min_heavy_atoms"]
    min_rotatable_bonds = validation_config["min_rotatable_bonds"]
    max_rotatable_bonds = validation_config["max_rotatable_bonds"]
    num_molecules = validation_config["num_molecules"]
    entropy_min_threshold = validation_config["entropy_min_threshold"]

    # Load reaction filtering configuration
    reaction_config = config["reaction_filtering"]
    random_valid_reaction = reaction_config["random_valid_reaction"]

    # Run parameters
    run_cfg = config.get("run", {})
    time_budget_sec = int(run_cfg.get("time_budget_sec", 600))
    competition_interval_seconds = run_cfg.get("competition_interval_seconds")
    if competition_interval_seconds is not None:
        competition_interval_seconds = int(competition_interval_seconds)

    return {
        'num_antitargets': num_antitargets,
        'antitarget_weight': antitarget_weight,
        'min_heavy_atoms': min_heavy_atoms,
        'min_rotatable_bonds': min_rotatable_bonds,
        'max_rotatable_bonds': max_rotatable_bonds,
        'num_molecules': num_molecules,
        'entropy_min_threshold': entropy_min_threshold,
        'random_valid_reaction': random_valid_reaction,
        'time_budget_sec': time_budget_sec,
        'competition_interval_seconds': competition_interval_seconds,
    }


def load_time_budget_sec(path: str = os.path.join(BASE_DIR, "config/config.yaml")) -> int:
    """Fast path to read only time_budget_sec as an int."""
    with open(path, "r", encoding="utf-8") as f:
        return int(yaml.safe_load(f)["run"]["time_budget_sec"]) 


