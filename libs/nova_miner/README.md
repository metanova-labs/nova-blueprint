# nova_miner

The library available to miner code inside the sandbox. It ships in the sandbox
image at `site-packages/nova_miner`.

## Contents

- `utils.oracle` — `Oracle.score()` requests Boltz-2 affinity predictions over the
  broker's unix socket; `combine()` reduces one prediction to the NOVA score.
- `utils.molecules` — SMILES lookup, heavy-atom counts, MACCS entropy, chemical
  identity grouping.
- `utils.reactions` — reaction count and validity.
- `combinatorial_db` — the reaction database and SMILES construction from `rxn:` names.

## Scoring

```python
import os
from nova_miner.utils.oracle import Oracle, combine
from nova_miner.utils.molecules import get_heavy_atom_count

oracle = Oracle(os.environ["ORACLE_SOCKET"])
rows = oracle.score(targets=[target_sequence], smiles=[smiles])
score = combine(rows[0]["scores"][0], get_heavy_atom_count(smiles))
```

Higher is better. `combine` returns `-inf` where the oracle returned no prediction.
One request covers every (molecule, target) pair, so pass all targets at once.

## Runtime

The sandbox has no network and no GPU: `ORACLE_SOCKET` is the only route out, and
the broker stamps each request with the run's identity. Logging is the standard
library's — configure handlers in your entrypoint to see it.
