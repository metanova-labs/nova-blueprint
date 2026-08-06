## NOVA Blueprint - SN68

ML‑driven drug discovery on Bittensor.
NOVA Blueprint powers SN68 by running competitive cycles that explore vast chemical spaces, collect candidate molecules, and iteratively build on the best‑performing approaches.

This codebase implements the SN68 validator scheduler and the sandboxed miner runner: it pulls miner snapshots from the submission API + MinIO archive, generates per‑cycle challenges, executes miners in an isolated Docker sandbox within a fixed time budget, and collects `/output/result.json` for scoring.

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
SNAPSHOT_S3_ENDPOINT=s3.metanova-labs.ai
MINIO_ACCESS_KEY=your_minio_access_key
MINIO_SECRET_KEY=your_minio_secret_key
SUBMISSION_API_URL=https://submission-api.metanova-labs.ai
SUBMISSION_API_KEY=your_submission_read_api_key
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

### Miner submission SDK

Use the SDK to upload signed code submissions for the competition.

```python
from utils.submission_uploader import submit_from_local_path, submit_from_github_url

result = submit_from_local_path(
    local_path="/path/to/miner_project_dir", 
    wallet_name="my_wallet",
    hotkey_name="my_hotkey",
    submission_name="dock-sense2",  # name appears on the dashboard
)

# or
result = submit_from_github_url(
    github_url="https://github.com/<owner>/<repo>",
    wallet_name="my_wallet",
    hotkey_name="my_hotkey",
    submission_name="dock-sense2",  # name appears on the dashboard
)

print(result.status_code, result.ok, result.request_id, result.body)
```

Notes:
- Submissions must be signed by a hotkey that owns the submission (its coldkey receives emissions if you win). One active slot per hotkey per epoch; re-submitting with the same hotkey in the same epoch overwrites the previous code upload with no fee.
- Submission eligibility and instructions depend on the live submission mode; check `GET /quote?hotkey=<ss58>` and https://submission-api.metanova-labs.ai/docs
---


