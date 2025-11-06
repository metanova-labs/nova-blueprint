## NOVA Blueprint - SN68

ML‑driven drug discovery on Bittensor.
NOVA Blueprint powers SN68 by running competitive cycles that explore vast chemical spaces, collect candidate molecules, and iteratively build on the best‑performing approaches.

This codebase implements the SN68 validator scheduler and the sandboxed miner runner: it pulls miner submissions from‑chain, generates per‑cycle challenges, executes miners in an isolated Docker sandbox within a fixed time budget, and collects `/output/result.json` for scoring.

### System Requirements (validators)
- Docker with docker compose plugin
- NVIDIA RTX 4090 GPU, NVIDIA driver + NVIDIA Container Toolkit installed on host
- Bittensor wallets on host (default: `$HOME/.bittensor/wallets`)

---

## For Validators

### Install and run
1) Clone the repo:
```bash
git clone https://github.com/metanova-labs/nova-blueprint.git
cd nova-blueprint
```

2) Create and fill `.env`:
```bash
cp .env.example .env
```

```
BT_WALLET_COLD=your_cold_wallet_name
BT_WALLET_HOT=your_hotkey_name
SUBTENSOR_NETWORK=finney
GITHUB_TOKEN=your_github_pat
# Optional if your wallets are not in the default location
# BT_WALLETS_DIR=$HOME/.bittensor/wallets 
```

3) Run the validator:
```bash
docker compose down && docker compose pull && docker compose up -d && docker compose logs -f
```

Notes:
- Artifacts/results are written under `./data/miner_runs` and `./data/results` by default.
- Auto‑update is handled by the bundled watchtower sidecar.

---

## For Miners 

Your miner repo is cloned and executed in a Docker sandbox (no network, read‑only root; use `/tmp`).

Must‑haves:
- `miner.py` at repo root is run as `python /workspace/miner.py`
- Read input from `/workspace/input.json`
- Write output to `/output/result.json` with reaction‑formatted molecules only (`rxn:*`)

Minimal example `result.json`:
```json
{
  "molecules": ["rxn:4:…", "rxn:5:…"]
}
```

**Note - Combinatorial SQLite DB**: open the provided database in read‑only mode to avoid write errors on a read‑only filesystem. Example: `sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)`.

Timing: You have a fixed time budget to generate the highest‑scoring set of molecules, at timeout molecules are collected to be scored against other submissions.  
Where it’s defined: `config/config.yaml` → `run.time_budget_sec` (default 1800s).  
Recommendation: keep `/output/result.json` up‑to‑date during the run so the latest results are captured when the sandbox exits.

---


