"""
Champion contest state and transition logic.

A frozen champion is re-run every epoch. A frozen challenger must beat the
current champion by `improvement_margin` for `wins_required` consecutive epochs
(single elimination) to be promoted. A live entrant that beats the champion is
admitted as a challenger with wins=1 (its qualifying beat is win #1).
"""

import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import bittensor as bt

CONTEST_STATE_PATH = Path("/data/results/contest_state.json")
WINNER_JSON_PATH = Path("/data/results/winner.json")

_IDENTITY_FIELDS = ("hotkey", "uid", "snapshot_epoch", "coldkey", "submission_name")


def entry_id(hotkey: str, snapshot_epoch: int) -> str:
    """Stable key for an entry, uniform for live entrants and frozen members."""
    return f"{hotkey}@{int(snapshot_epoch)}"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _identity(entry: dict) -> dict:
    return {k: entry.get(k) for k in _IDENTITY_FIELDS}


def _member_entry_id(member: dict) -> str:
    return entry_id(member["hotkey"], member["snapshot_epoch"])


def _finite_score(score_dict: dict, eid: str) -> Optional[float]:
    """Return the entry's final score if it ran and scored finitely, else None."""
    data = score_dict.get(eid)
    if not data:
        return None
    raw = data.get("ps_final_score")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isinf(value) or math.isnan(value):
        return None
    return value


def load_contest_state() -> dict:
    """Load contest state; bootstrap the champion from winner.json if absent."""
    try:
        if CONTEST_STATE_PATH.exists():
            with CONTEST_STATE_PATH.open("r", encoding="utf-8") as f:
                state = json.load(f)
            state.setdefault("champion", None)
            state.setdefault("challengers", [])
            return state
    except Exception as e:
        bt.logging.error(f"contest: failed to read contest_state.json: {type(e).__name__}: {e}")

    champion = _bootstrap_champion_from_winner_json()
    if champion is not None:
        bt.logging.info(f"contest: bootstrapped champion from winner.json hotkey={champion['hotkey']}")
    return {"champion": champion, "challengers": [], "updated_at": _now()}


def _bootstrap_champion_from_winner_json() -> Optional[dict]:
    try:
        if not WINNER_JSON_PATH.exists():
            return None
        with WINNER_JSON_PATH.open("r", encoding="utf-8") as f:
            snapshot = json.load(f)["winner_snapshot"]
        return {
            "hotkey": str(snapshot["hotkey"]),
            "uid": int(snapshot["uid"]),
            "snapshot_epoch": int(snapshot["snapshot_epoch"]),
            "coldkey": snapshot.get("coldkey"),
            "submission_name": snapshot.get("submission_name"),
        }
    except Exception as e:
        bt.logging.error(f"contest: failed to bootstrap champion from winner.json: {type(e).__name__}: {e}")
        return None


def save_contest_state(state: dict) -> None:
    """Atomically persist contest state."""
    state["updated_at"] = _now()
    out_dir = CONTEST_STATE_PATH.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir / "contest_state.json.tmp"
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, CONTEST_STATE_PATH)


def _promote(
    ready: list,
    survivors: list,
    score_dict: dict,
) -> Tuple[dict, str, Optional[float]]:
    """Promote the challenger with the best average score across its winning epochs; reset others to wins=0."""
    # Tied challengers share the same win count, so max score_sum ranks them by best average.
    winner = max(ready, key=lambda m: float(m.get("score_sum", 0.0)))
    new_challengers = []
    for member in survivors:
        if member is winner:
            continue
        reset = dict(member)
        reset["wins"] = 0
        reset["score_sum"] = 0.0
        new_challengers.append(reset)
    new_state = {"champion": _identity(winner), "challengers": new_challengers, "updated_at": _now()}
    weid = _member_entry_id(winner)
    return new_state, weid, _finite_score(score_dict, weid)


def apply_contest_transition(
    score_dict: dict,
    entries: Dict[str, dict],
    state: dict,
    cfg: dict,
    epoch: int,
) -> Tuple[dict, Optional[str], Optional[float]]:
    """
    Pure contest resolution. Returns (new_state, champion_entry_id, champion_score).

    Rules:
    - Cold start (no champion): crown the top finite scorer.
    - Champion infra-failure (no finite score): freeze transitions, keep state.
    - Each challenger beating champion by margin: wins += 1; failing: eliminated;
      infra-failing (no result): skipped, wins kept.
    - A challenger reaching wins_required is promoted (ties broken by best average score across winning epochs);
      surviving challengers reset to wins=0; old champion dropped; no admission.
    - Otherwise admit entrants beating champion by margin as challengers wins=1.
    """
    wins_required = int(cfg["wins_required"])
    margin = float(cfg["improvement_margin"])

    champion = state.get("champion")

    if champion is None:
        best_eid, best_score = None, -math.inf
        for eid in entries:
            score = _finite_score(score_dict, eid)
            if score is None:
                continue
            if round(score, 6) > round(best_score, 6):
                best_eid, best_score = eid, score
        if best_eid is None:
            return {"champion": None, "challengers": [], "updated_at": _now()}, None, None
        new_state = {"champion": _identity(entries[best_eid]), "challengers": [], "updated_at": _now()}
        return new_state, best_eid, best_score

    champ_eid = _member_entry_id(champion)
    champ_score = _finite_score(score_dict, champ_eid)
    if champ_score is None:
        bt.logging.info(f"contest: champion {champion['hotkey']} did not score; freezing transitions")
        return state, champ_eid, None

    threshold = champ_score + margin

    survivors: list = []
    for challenger in state.get("challengers", []):
        ceid = _member_entry_id(challenger)
        cscore = _finite_score(score_dict, ceid)
        if cscore is None:
            survivors.append(dict(challenger))  # infra fail: skip, keep wins
            continue
        if cscore >= threshold:
            advanced = dict(challenger)
            advanced["wins"] = int(challenger.get("wins", 0)) + 1
            advanced["score_sum"] = float(challenger.get("score_sum", 0.0)) + cscore
            survivors.append(advanced)
        # else: loss -> eliminated (dropped)

    ready = [m for m in survivors if int(m.get("wins", 0)) >= wins_required]
    if ready:
        return _promote(ready, survivors, score_dict)

    for eid, entry in entries.items():
        if entry.get("kind") != "entrant":
            continue
        score = _finite_score(score_dict, eid)
        if score is None or score < threshold:
            continue
        admitted = _identity(entry)
        admitted["wins"] = 1
        admitted["score_sum"] = score
        survivors.append(admitted)

    ready = [m for m in survivors if int(m.get("wins", 0)) >= wins_required]
    if ready:  # only reachable when wins_required == 1 (entry beat is win #1)
        return _promote(ready, survivors, score_dict)

    new_state = {"champion": champion, "challengers": survivors, "updated_at": _now()}
    return new_state, champ_eid, champ_score
