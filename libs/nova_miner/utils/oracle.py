"""Scoring client for the miner sandbox: HTTP over the broker's unix socket.

Standard library only. Refusals raise OracleError, an absent socket OSError.
"""
from __future__ import annotations

import http.client
import json
import math
import socket


class OracleError(RuntimeError):
    """A refused request. `status` is the HTTP code."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


class _UnixConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float):
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self._path)


class Oracle:
    # 900s matches the oracle's own request timeout.
    def __init__(self, socket_path: str, timeout: float = 900):
        self.socket_path = socket_path
        self.timeout = timeout

    def score(self, targets: list[str], smiles: list[str]) -> list[dict]:
        """results[i]["scores"][t] is smiles[i] against targets[t], or None."""
        body = json.dumps({"targets": list(targets), "smiles": list(smiles)})
        connection = _UnixConnection(self.socket_path, self.timeout)
        try:
            connection.request("POST", "/v1/score", body,
                               {"Content-Type": "application/json"})
            response = connection.getresponse()
            payload = response.read()
            if response.status != 200:
                raise OracleError(response.status, _detail(payload))
            return json.loads(payload)["results"]
        finally:
            connection.close()


def _detail(payload: bytes) -> str:
    try:
        return str(json.loads(payload)["detail"])
    except (ValueError, KeyError, TypeError):
        return payload.decode(errors="replace")[:200]


FORMULA = "(affinity_probability_binary - affinity_pred_value) / heavy_atom_count"


def combine(scores: dict | None, heavy_atom_count: int) -> float:
    """One number per (molecule, target), higher is better.

    The two affinity metrics differenced -- they point opposite ways -- then
    normalised by molecule size so bigger is not automatically better.
    """
    if not scores or not heavy_atom_count:
        return -math.inf
    return (scores["affinity_probability_binary"]
            - scores["affinity_pred_value"]) / heavy_atom_count
