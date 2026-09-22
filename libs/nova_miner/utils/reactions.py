import logging
import os
import sqlite3

log = logging.getLogger(__name__)


def get_total_reactions() -> int:
    """Query database for total number of reactions, add 1 for savi option."""
    try:
        db_path = os.path.join(os.path.dirname(__file__), "../combinatorial_db/molecules.sqlite")
        conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM reactions")
        count = cursor.fetchone()[0]
        conn.close()
        return count + 1  # +1 for savi option
    except Exception as e:
        raise RuntimeError(
            f"Could not query the combinatorial database reaction count: {e}"
        ) from e


def is_reaction_allowed(molecule: str, allowed_reaction: str = None) -> bool:
    """
    Check if molecule matches the allowed reaction for this epoch.
    - If allowed_reaction is None: all molecules allowed (filtering disabled)
    - If filtering enabled: only molecules matching the allowed_reaction type are allowed
    """
    if allowed_reaction is None:
        return True  
        
    if not molecule:
        return False 
    
    if molecule.startswith("rxn:"):
        try:
            parts = molecule.split(":")
            if len(parts) >= 2:
                rxn_id = int(parts[1])
                return allowed_reaction == f"rxn:{rxn_id}"
            return False  # Malformed rxn format
        except Exception as e:
            log.warning(f"Error parsing reaction molecule '{molecule}': {e}")
            return False
    else:
        # Not in reaction format, only allowed if savi is the allowed reaction
        return allowed_reaction == "savi"
