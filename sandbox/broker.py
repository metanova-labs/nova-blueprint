"""Serves a unix socket for one sandbox run, stamping caller identity onto
each request and forwarding it to the oracle."""
from __future__ import annotations

import http.client
import json
import os
import socketserver
import sys
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

ORACLE_URL = os.environ.get("ORACLE_URL", "")
ORACLE_TOKEN = os.environ.get("ORACLE_TOKEN", "")

TIMEOUT_S = 900
REGISTER_TIMEOUT_S = 1800   # a new target needs an alignment search


def _post(path: str, body: dict, timeout: float) -> tuple[int, bytes]:
    url = urlparse(ORACLE_URL)
    opener = (http.client.HTTPSConnection if url.scheme == "https"
              else http.client.HTTPConnection)
    connection = opener(url.netloc, timeout=timeout)
    try:
        connection.request("POST", path, json.dumps(body), {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ORACLE_TOKEN}"})
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def register_proteins(sequences: list[str],
                      labels: list[str] | None = None) -> list[dict]:
    """Cache alignments and warm the oracle. Once per round, before any run."""
    body: dict = {"sequences": list(sequences)}
    if labels:
        body["labels"] = list(labels)
    status, payload = _post("/v1/proteins", body, REGISTER_TIMEOUT_S)
    if status != 200:
        raise RuntimeError(f"oracle refused proteins ({status}): {payload[:200]!r}")
    return json.loads(payload)["proteins"]


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        if self.path != "/v1/score":
            return self._send(404, b'{"detail":"not found"}')
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            targets, smiles = body["targets"], body["smiles"]
        except (ValueError, KeyError, TypeError):
            return self._send(400, b'{"detail":"bad request"}')

        # Identity comes from the run.
        status, payload = _post("/v1/score", {
            "miner": self.server.miner, "epoch": self.server.epoch,
            "targets": targets, "smiles": smiles}, TIMEOUT_S)
        self._send(status, payload)

    def _send(self, status: int, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:
        pass


class _Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def get_request(self):
        return self.socket.accept()[0], ("sandbox", 0)

    def handle_error(self, request, client_address) -> None:
        # A sandbox killed at its time budget drops the connection mid-reply.
        if not isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            super().handle_error(request, client_address)


class Broker:
    """One socket for one run:

        with Broker(path, miner=entry_id, epoch=period):
            ...
    """

    def __init__(self, socket_path: str | Path, miner: str, epoch: str):
        self.socket_path = str(socket_path)
        self.miner = miner
        self.epoch = epoch
        self._server: _Server | None = None

    def __enter__(self) -> "Broker":
        Path(self.socket_path).unlink(missing_ok=True)
        server = _Server(self.socket_path, _Handler)
        try:
            server.miner = self.miner
            server.epoch = self.epoch
            os.chmod(self.socket_path, 0o666)   # the sandbox runs as another uid
            threading.Thread(target=server.serve_forever, daemon=True).start()
        except BaseException:
            server.server_close()
            Path(self.socket_path).unlink(missing_ok=True)
            raise
        self._server = server
        return self

    def __exit__(self, *exc) -> None:
        try:
            self._server.shutdown()
        finally:
            try:
                self._server.server_close()
            finally:
                Path(self.socket_path).unlink(missing_ok=True)
